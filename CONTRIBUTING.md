# Contributing to strixite

Thank you for wanting to help. Pull requests and issues are welcome.

## How a pull request lands

I test every change on my own Strix Halo before it goes in: the full test suite, and a before/after benchmark for
anything that touches speed. If it holds up, I merge it. It can take a few days.

strixite is published from the repository I work in day to day, so after a merge the next release may rearrange your
change a little to fit where the engine has moved since. It stays your change, and your name stays on it.

Contributions are accepted under the project's license, [AGPL-3.0](LICENSE).

Please don't include code, comments or docs copied from other projects. strixite is written from scratch, and I'd
like to keep it that way. Ideas from elsewhere are fine; say where they came from.

## What helps a change get in

- **Output identity:** run a few prompts at `temperature` 0 on both builds and compare the tokens. Identical output
  means the change only moves speed. If the output differs, say where and why - and keep in mind that decode speed
  with the draft head depends on the text being written, so comparing the speed of two different outputs mostly
  measures the difference in text. (I run the test suite, which checks every kernel against a reference, on my side.)
- **A before/after measurement from a Strix Halo**, with how you took it:
  - the command and the settings;
  - the old and the new build on the same machine, alternated (old, new, old, new, ...), at least four rounds each;
  - every round's number, or the spread, next to the difference. One run per side can't tell a gain of a few percent
    from run-to-run noise.
- **Where the time goes**, for a kernel change: the kernel's own time before and after (a micro-benchmark, or a
  trace with the `strix-roctx` preset), and roughly what share of a forward it takes. A kernel with a tiny share
  can't move end-to-end speed much, however much faster it gets.
- **A way to reproduce it**, for a bug fix: the request, the settings and the output that show the bug.
- **Small, focused changes.** One idea per pull request is much easier for me to verify.

## Pull requests written with an LLM

Fine with me - I use them too. Before you send one, please check its explanation against the code: the functions it
names should exist, the mechanism it describes should be what the code does, and every number should come from a run
you did. A wrong explanation costs us both time, even when the change itself is good.

## Scope

strixite targets one chip and one model: AMD Strix Halo (gfx1151) on Linux, and Qwen3.8-Flash-Next. I can't take on
ports to other platforms or models here, but forks are very welcome (see [Platform](README.md#platform)).
