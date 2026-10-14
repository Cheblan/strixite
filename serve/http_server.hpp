#pragma once

// The HTTP surface on cpp-httplib: POST /v1/chat/completions (JSON or SSE),
// GET /v1/models, /health, /metrics (Prometheus text), /cache (prefix-reuse counters). Requests are
// rendered and tokenized on the HTTP thread, then queued on the Engine; a streaming response relays the engine's
// events as they come, with SSE comment keep-alives while the request waits or prefills; a client that goes away
// cancels its request.

#include "serve/engine.hpp"
#include "serve/grammar_cache.hpp"
#include "serve/tokenizer.hpp"
#include "serve/vision_request.hpp"

#include <atomic>
#include <memory>
#include <string>

namespace httplib {
class Server;
}

namespace strix {

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 5300;
    std::string model_id = "qwen3.8-flash-next";
    SamplingParams defaults;  // generation_config.json's
    // Thinking for requests that don't say (no enable_thinking in any spelling, no reasoning_effort, no budget): false
    // renders them without the think block (config `thinking`).
    bool thinking_default = true;
    // thinking-cap: the cap on thinking tokens for requests that name no budget of their own (0 = off). A request's own
    // budget always wins. thinking-cap-grace: past the cap, at most this many more thinking tokens while the engine
    // waits for a sentence end (GenerationRequest::thinking_budget_grace).
    int64_t thinking_cap = 0, thinking_cap_grace = 0;
    double keepalive_s = 10;  // SSE comment interval while nothing else is sent
    size_t max_body_bytes = 64u << 20;
    // Vision (Phase D): non-null when the server loaded with vision on - the image parts parse, preprocess
    // (on the HTTP thread) and reach the engine; /health reports engine_.backend().has_vision().
    const vision::ServedVision *vision = nullptr;
};

class HttpServer {
public:
    HttpServer(Engine &engine, const Tokenizer &tok, ServerConfig cfg);
    ~HttpServer();
    // Binds (port 0 = any free port; returns the port) - then serve() blocks until stop().
    int bind();
    void serve();
    void stop();

private:
    Engine &engine_;
    const Tokenizer &tok_;
    ServerConfig cfg_;
    int64_t started_;
    std::atomic<int64_t> next_id_{0};  // request ids for the log ("req <id> ...")
    GrammarCache grammars_;            // compiled response formats (structured output), shared across requests
    std::unique_ptr<httplib::Server> svr_;
    void routes();
};

}  // namespace strix
