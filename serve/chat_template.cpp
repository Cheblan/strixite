#include "serve/chat_template.hpp"

#include "common/check.hpp"
#include "serve/unicode.hpp"

#include <cctype>

namespace strix {

namespace {

// A past turn's thinking that ends in a budget stop is rendered without that last paragraph. A budget stop: a
// paragraph of at most 300 bytes starting "Considering" that speaks of the user's time ("limited time", "time
// constraint", "user's time"), any case - the engine's kThinkingStop, or the model's own copies of it. Sent back in
// history (clients that return past reasoning), the injected sentence teaches the model to open its thinking with it
// and skip thinking, in drifting wordings ("Considering the user's limited time, I have to answer directly ...",
// "Considering the user's time constraints, I must answer directly from thinking right now."). The length bound keeps
// a real conclusion that happens to start the same way.
std::string drop_budget_stop(const std::string &reasoning) {
    static const std::string kOpen = "considering";
    static const char *const kTime[] = {"limited time", "time constraint", "user's time", "user\u2019s time"};
    constexpr size_t kMaxStop = 300;
    const size_t para = reasoning.rfind("\n\n");
    const size_t start = para == std::string::npos ? 0 : para + 2;
    if (reasoning.size() - start > kMaxStop) return reasoning;
    std::string last = reasoning.substr(start);
    for (char &ch : last) ch = (char)std::tolower((unsigned char)ch);
    if (last.compare(0, kOpen.size(), kOpen) != 0) return reasoning;
    bool about_time = false;
    for (const char *k : kTime) about_time = about_time || last.find(k) != std::string::npos;
    if (!about_time) return reasoning;
    return unicode::python_strip(reasoning.substr(0, para == std::string::npos ? 0 : para));
}

constexpr const char *kXhigh =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider "
    "plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
constexpr const char *kLow =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
    "without unnecessary elaboration.";
constexpr const char *kToolsIntro = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
constexpr const char *kToolsFormat =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in "
    "natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer "
    "the question like normal with your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";

bool starts_with(const std::string &s, const char *p) { return s.rfind(p, 0) == 0; }
bool ends_with(const std::string &s, const std::string &p) {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

// The template's render_content macro, then |trim. With vision on, an image part renders as one
// placeholder byte in `text` (0x01 - NOT 0x1f: Python's str.strip() treats 0x1c..0x1f as whitespace and would
// trim a placeholder at a content edge, while interior text is kept byte-exactly) and its offset goes in
// `images`; render_chat expands only the recorded offsets. The request's own text may hold 0x01 too (a tool
// result that printed a binary file): those bytes are plain text, never an image.
// With vision off the v1 refusal stands, word for word.
constexpr char kVisionPlaceholder = '\x01';
constexpr const char *kVisionTokens = "<|vision_start|><|image_pad|><|vision_end|>";

struct Content {
    std::string text;
    std::vector<size_t> images;  // byte offsets in text of the image placeholders, ascending
};

Content content_of(const json::Value &msg, const std::string &path, bool is_system, bool vision) {
    const json::Value *c = msg.find("content");
    if (!c || c->is_null()) return {};
    if (c->is_string()) return {unicode::python_strip(c->as_string(path)), {}};
    STRIX_CHECK(c->is_array(), "chat template: '", path, "' must be a string, null or an array of parts, got ",
                json::type_name(c->type()), " (Unexpected content type.)");
    std::string out;
    std::vector<size_t> images;
    const auto &parts = c->as_array(path);
    for (size_t k = 0; k < parts.size(); ++k) {
        const std::string pp = path + "[" + std::to_string(k) + "]";
        const json::Value &part = parts[k];
        STRIX_CHECK(part.is_object(), "chat template: '", pp, "' must be an object, got ", json::type_name(part.type()));
        const json::Value *type = part.find("type");
        const bool image = part.find("image") || part.find("image_url") || (type && type->is_string() && type->as_string(pp) == "image");
        const bool video = part.find("video") || (type && type->is_string() && type->as_string(pp) == "video");
        STRIX_CHECK(!(is_system && (image || video)), "chat template: '", pp, "': System message cannot contain images or videos.");
        STRIX_CHECK(!video, "chat template: '", pp, "' is a video part; videos are not served");
        if (image) {
            STRIX_CHECK(vision, "chat template: '", pp, "' is an image / video part; vision is not served (v1)");
            images.push_back(out.size());
            out += kVisionPlaceholder;
            continue;
        }
        const json::Value *text = part.find("text");
        STRIX_CHECK(text != nullptr, "chat template: '", pp, "' has no 'text' (Unexpected item type in content.)");
        out += text->as_string(pp + ".text");
    }
    std::string text = unicode::python_strip(out);
    if (images.empty()) return {std::move(text), {}};
    // The placeholders are not whitespace, so the strip removed only a leading and a trailing run around them:
    // the result starts at the first non-whitespace byte, where it is first found.
    const size_t lead = out.find(text);
    STRIX_CHECK(lead != std::string::npos && lead <= images.front() && images.back() < lead + text.size(),
                "chat template: '", path, "': the trimmed content (", text.size(), " of ", out.size(),
                " bytes, at ", lead, ") does not keep the image placeholders at ", images.front(), "..",
                images.back());
    for (size_t &at : images) {
        at -= lead;
        STRIX_CHECK(text[at] == kVisionPlaceholder, "chat template: '", path, "': image placeholder expected at byte ",
                    at, " of the trimmed content, found byte ", (int)(unsigned char)text[at]);
    }
    return {std::move(text), std::move(images)};
}

const std::string &role_of(const json::Value &msg, const std::string &path) {
    STRIX_CHECK(msg.is_object(), "chat template: '", path, "' must be an object, got ", json::type_name(msg.type()));
    const json::Value *r = msg.find("role");
    STRIX_CHECK(r != nullptr, "chat template: '", path, "' has no 'role'");
    return r->as_string(path + ".role");
}

// A tool call's arguments as the object the template iterates: an object as is, a JSON string parsed (it must
// hold an object). nullptr = render no parameters: absent, the empty string, or an object's JSON left open the way
// a streamed call that was cut off or malformed leaves it ("}" or "\"}" would close it), sent back as received.
std::optional<json::Value> arguments_of(const json::Value &call, const std::string &path) {
    const json::Value *a = call.find("arguments");
    if (!a) return std::nullopt;
    if (a->is_string()) {
        const std::string &s = a->as_string(path);
        if (s.empty()) return std::nullopt;
        json::Value v;
        try {
            v = json::Value::parse(s);
        } catch (const Error &e) {
            for (const char *close : {"}", "\"}"}) {
                try {
                    if (json::Value::parse(s + close).is_object()) return std::nullopt;
                } catch (const Error &) {
                }
            }
            STRIX_FAIL("chat template: '", path, ".arguments' is a string but not JSON: ", e.what());
        }
        STRIX_CHECK(v.is_object(), "chat template: '", path, ".arguments' must hold a JSON object, got ",
                    json::type_name(v.type()));
        return v;
    }
    STRIX_CHECK(a->is_object(), "chat template: '", path, ".arguments' must be an object or a JSON string, got ",
                json::type_name(a->type()));
    return *a;
}

}  // namespace

std::string render_chat(const json::Value &messages, const json::Value *tools, const ChatTemplateOptions &opt,
                        std::vector<std::pair<size_t, size_t>> *plain_spans, bool vision) {
    if (plain_spans) plain_spans->clear();
    const auto &msgs = messages.as_array("messages");
    STRIX_CHECK(!msgs.empty(), "chat template: No messages provided.");
    std::vector<std::string> roles;
    for (size_t k = 0; k < msgs.size(); ++k) roles.push_back(role_of(msgs[k], "messages[" + std::to_string(k) + "]"));

    std::string reasoning;
    if (!opt.enable_thinking.has_value() || *opt.enable_thinking) {
        const std::string effort = opt.reasoning_effort.value_or("xhigh");
        STRIX_CHECK(effort == "xhigh" || effort == "medium" || effort == "low", "chat template: Unexpected reasoning effort ",
                    effort, ". Supported types are xhigh (default), medium, and low.");
        if (effort == "xhigh") reasoning = kXhigh;
        else if (effort == "low") reasoning = kLow;
    }

    std::string out;
    // Appends request-supplied text, recording its span.
    // Request text, span-marked - except at the image offsets (content_of's `images`, ascending, each
    // holding the placeholder byte): the template's own vision tokens go in UNMARKED (a plain span means
    // "the user's text - never a control token"; these are the server's control tokens). Every other byte,
    // 0x01 included, is the request's text. With no images the lambda is today's, byte-identical.
    const auto plain = [&](const std::string &raw, const std::vector<size_t> &images = {}) {
        size_t at = 0;
        for (size_t k = 0; k <= images.size(); ++k) {
            const size_t nxt = k < images.size() ? images[k] : raw.size();
            STRIX_CHECK(nxt >= at && nxt <= raw.size() && (k == images.size() || raw[nxt] == kVisionPlaceholder),
                        "chat template: image offset ", k, " of ", images.size(), " is ", nxt,
                        " - expected ascending placeholder bytes inside the ", raw.size(), "-byte text (from ", at, ")");
            if (nxt > at) {
                if (plain_spans) plain_spans->emplace_back(out.size(), out.size() + (nxt - at));
                out.append(raw, at, nxt - at);
            }
            if (k == images.size()) break;
            out += kVisionTokens;
            at = nxt + 1;
        }
    };
    const bool have_tools = tools && tools->is_array() && !tools->as_array("tools").empty();
    if (have_tools) {
        out += "<|im_start|>system\n";
        if (!reasoning.empty()) out += reasoning + "\n\n";
        out += kToolsIntro;
        for (const json::Value &tool : tools->as_array("tools")) out += "\n", plain(tool.dump_python());
        out += "\n</tools>";
        out += kToolsFormat;
        if (roles[0] == "system") {
            const std::string c = content_of(msgs[0], "messages[0].content", true, vision).text;
            if (!c.empty()) out += "\n\n", plain(c);
        }
        out += "<|im_end|>\n";
    } else if (roles[0] == "system") {
        const std::string c = content_of(msgs[0], "messages[0].content", true, vision).text;
        if (!c.empty()) {
            out += "<|im_start|>system\n" + (reasoning.empty() ? "" : reasoning + "\n\n");
            plain(c);
            out += "<|im_end|>\n";
        }
        else if (!reasoning.empty()) out += "<|im_start|>system\n" + reasoning + "<|im_end|>\n";
    } else if (!reasoning.empty()) {
        out += "<|im_start|>system\n" + reasoning + "<|im_end|>\n";
    }

    // The last real user query (a user message that isn't just wrapped tool responses).
    long last_query = -1;
    for (size_t k = msgs.size(); k-- > 0;) {
        if (roles[k] != "user") continue;
        const std::string c = content_of(msgs[k], "messages[" + std::to_string(k) + "].content", false, vision).text;
        if (!(starts_with(c, "<tool_response>") && ends_with(c, "</tool_response>"))) {
            last_query = (long)k;
            break;
        }
    }
    STRIX_CHECK(last_query >= 0, "chat template: No user query found in messages.");

    for (size_t k = 0; k < msgs.size(); ++k) {
        const std::string path = "messages[" + std::to_string(k) + "]";
        const json::Value &m = msgs[k];
        const std::string &role = roles[k];
        const Content parsed = content_of(m, path + ".content", role == "system", vision);
        const std::string &content = parsed.text;
        if (role == "assistant")
            STRIX_CHECK(parsed.images.empty(), "chat template: '", path,
                        ".content': assistant messages cannot contain image parts");
        if (role == "system") {
            STRIX_CHECK(k == 0, "chat template: '", path, "': System message must be at the beginning.");
        } else if (role == "user") {
            out += "<|im_start|>user\n";
            // Tool results a client wrapped in a user message (the template's own test above): the wrapper stays markup.
            const std::string open = "<tool_response>", close = "</tool_response>";
            if (content.size() >= open.size() + close.size() && starts_with(content, open.c_str()) && ends_with(content, close)) {
                out += open;
                std::vector<size_t> inner = parsed.images;  // none sit in the wrapper: it is text at both ends
                for (size_t &at : inner) at -= open.size();
                plain(content.substr(open.size(), content.size() - open.size() - close.size()), inner);
                out += close;
            } else {
                plain(content, parsed.images);
            }
            out += "<|im_end|>\n";
        } else if (role == "assistant") {
            std::string reasoning_content;
            if (const json::Value *r = m.find("reasoning_content"); r && r->is_string())
                reasoning_content = drop_budget_stop(unicode::python_strip(r->as_string(path)));
            if (!opt.preserve_thinking.has_value() || *opt.preserve_thinking || (long)k > last_query)
                out += "<|im_start|>assistant\n<think>\n" + reasoning_content + "\n</think>\n\n" + content;
            else
                out += "<|im_start|>assistant\n" + content;
            const json::Value *calls = m.find("tool_calls");
            if (calls && calls->is_array()) {
                const auto &cs = calls->as_array(path + ".tool_calls");
                for (size_t j = 0; j < cs.size(); ++j) {
                    const std::string cp = path + ".tool_calls[" + std::to_string(j) + "]";
                    STRIX_CHECK(cs[j].is_object(), "chat template: '", cp, "' must be an object");
                    const json::Value *fn = cs[j].find("function");
                    const json::Value &call = fn ? *fn : cs[j];
                    const std::string fp = fn ? cp + ".function" : cp;
                    STRIX_CHECK(call.is_object(), "chat template: '", fp, "' must be an object");
                    const json::Value *name = call.find("name");
                    STRIX_CHECK(name != nullptr, "chat template: '", fp, "' has no 'name'");
                    const std::string &nm = name->as_string(fp + ".name");
                    if (j == 0) out += (content.empty() ? "" : "\n\n");
                    else out += "\n";
                    out += "<tool_call>\n<function=" + nm + ">\n";
                    if (const auto args = arguments_of(call, fp))
                        for (const auto &[an, av] : args->as_object(fp + ".arguments")) {
                            out += "<parameter=" + an + ">\n";
                            plain(av.is_string() ? av.as_string(an) : av.dump_python());
                            out += "\n</parameter>\n";
                        }
                    out += "</function>\n</tool_call>";
                }
            } else {
                STRIX_CHECK(!calls || calls->is_null(), "chat template: '", path, ".tool_calls' must be an array, got ",
                            json::type_name(calls->type()));
            }
            out += "<|im_end|>\n";
        } else if (role == "tool") {
            if (k > 0 && roles[k - 1] != "tool") out += "<|im_start|>user";
            out += "\n<tool_response>\n";
            plain(content, parsed.images);
            out += "\n</tool_response>";
            if (k + 1 == msgs.size() || roles[k + 1] != "tool") out += "<|im_end|>\n";
        } else {
            STRIX_FAIL("chat template: '", path, ".role' is '", role, "': Unexpected message role.");
        }
    }
    if (opt.add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        out += opt.enable_thinking.has_value() && !*opt.enable_thinking ? "<think>\n\n</think>\n\n" : "<think>\n";
    }
    return out;
}

}  // namespace strix
