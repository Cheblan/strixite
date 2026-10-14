# libwebp decoder subset, vendored

What: the WebP decoder of libwebp 1.6.0 (Google, BSD 3-clause + patent grant: `COPYING`, `PATENTS`, `AUTHORS`), for
image inputs (serve/vision_preprocess.cpp `decode_image`). Why this and not a single header: stb_image has no WebP;
the single-header decoders refuse extended (VP8X) files - most WebPs with transparency - so they would turn away much
of what people send; libwebp is the reference decoder and heavily fuzzed, which matters for bytes from the network.

Source: `https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz`,
sha256 `e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564`; its detached signature verified
(2026-10-14) - good signature from "WebP release signing key <webp-discuss@webmproject.org>", fingerprint
`6B0E 6B70 976D E303 EDF2 F601 F9C3 D6BD B823 2B5D`.

Contents, unmodified: the files of upstream `makefile.unix`'s `DEC_OBJS`, `DSP_DEC_OBJS` (generic, SSE2, SSE4.1 -
the MIPS / MSA / NEON files dropped: x86 only) and `UTILS_DEC_OBJS`, the headers of `src/dec`, `src/dsp`, `src/utils`
and `src/webp` (minus `mux.h` / `demux.h`, unused). No encoder, demux, tools or examples. Built by CMakeLists.txt's
`strix_webp` (C11, TheRock clang). To update: replace these files from a newer release's same lists, re-verify the
signature, update this file.
