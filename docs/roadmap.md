# Roadmap

**Priorities, in order: stability, quality, performance.** A server that stays up comes first, then answers you can
trust, then speed.

What I'm working on, what comes next, and what I'm deliberately not doing. No dates: strixite is built by one person
around real use, and priorities move with what I measure. I update this page with each release.
The limit is rarely ideas: every change is tested and measured on the one machine I have, and that takes most of the
time. With fifteen Strix Halos, this page would be a lot shorter.

The focus stays the same throughout: a Strix Halo dedicated to serving a coding agent - fast at long context, stable
through long sessions.

Nothing here is set in stone. If there's a feature you need, or something on this page matters more to you than
where it sits, [open an issue](https://github.com/shawnshekari/strixite/issues/new) and say what you'd use it for -
real use cases are what move items up.

## Working on now

- **Less wasted thinking.** On agent benchmarks, most of the generated tokens are thinking, and much of it goes in
  circles. I'm measuring where it adds nothing to the answer, then tuning the thinking nudge and the think-block guard
  to cut it - the biggest speed lever left for agent work.
- **Faster prefill at long context and on short prompts.** v0.4.0 is ~3% faster at 32k-120k and reads cold n-gram
  rows ~1.25x faster; still to do: the sparse-attention scoring pass at depth, and the first chunk of a prompt waiting
  on n-gram table reads from the SSD.

## Next

- **A KV cache that grows with the conversation**: today the full context is reserved at startup, though most
  conversations use a fraction of it. Reserving the address space and backing it as the conversation grows would
  hand the unused part to the prompt cache, without slowing attention down. First a proof of concept that the GPU's
  virtual memory works for this on Strix Halo.
- **Say why a conversation was read again**: when a client changes the start of the prompt (a tool switched on, a
  date in the system prompt), the log and the response will name the cause, so it's clear the server didn't lose it
  ([why it happens](memory.md#when-the-client-changes-the-start-of-the-prompt)).
- **`tool_choice: required` and a named tool**, enforced the way structured output is.

## Considering

Ideas I want, but that wait for evidence that they're worth their cost, or for the items above.

- **Two conversations at once.** I built it and measured ~1.5x combined throughput with two conversations, but with
  several agents sharing the GPU each one got slower, so it's not in v0.4.0. What it needs first: drafting for all
  conversations in one pass instead of one after another, and the growing KV cache (Next) so a second conversation
  doesn't reserve memory it isn't using.
- **Changing settings without a restart**: an endpoint to change the server's settings while it runs - no 30-second
  reload of the weights, no emptied prompt cache.
- **Sleep mode**: after a configurable idle time, free the GPU memory and reload on the next request - for machines
  that do other things between sessions.
- **Smarter low-memory behaviour**: when memory is short, keep the prompt cache's RAM tier at whatever size fits
  instead of turning it off.
- **Smaller state**: the 8-bit KV cache shipped in v0.4.0; a BF16 recurrent state is next - more room for the
  prompt cache, if quality holds.
- **Responses API over WebSocket** ([#9](https://github.com/shawnshekari/strixite/issues/9)): a client sends each
  step's new items instead of the whole history, so long agent sessions upload and parse far less.
- **Calibrated 4-bit weights**: same size, possibly better quality - a new weights release if it measures better.
- **Easier setup**: a `models-dir` setting next to the `STRIX_MODELS_DIR` variable, and a one-command build per
  distribution.

## Not planned

- **Other GPUs, other operating systems, a CPU fallback.** strixite is written for one chip (gfx1151) and tested on
  one machine; a fallback would hide the bugs it should surface. Forks are welcome.
- **Other models**, for now - every kernel is written for this model's architecture.
- **Prebuilt binaries** beyond the container image.

## Helping

The most useful contributions are bug reports with the server's startup log (the `startup: memory:` line especially)
and changes that come with a measurement - a before / after on the same machine. See
[CONTRIBUTING.md](../CONTRIBUTING.md).
