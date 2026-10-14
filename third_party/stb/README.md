# stb - vendored single-header image codec

Unmodified copies from the official repository, https://github.com/nothings/stb (checked against master at
`2c980bb59875`, 2026-10-14):

- `stb_image.h` - v2.30, the image decoder (PNG, JPEG, GIF, BMP, TGA) used by `serve/vision_preprocess.cpp`.
  sha256 `594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3`
- `stb_image_write.h` - v1.16, an encoder used only by the tests (they write real image bytes and decode them
  back); the server never compiles against it. sha256 `cbd5f0ad7a9cf4468affb36354a1d2338034f2c12473cf1a8e32053cb6914a05`

Both are public domain (or MIT, at your choice - the license text is at the end of each file). The
`STB_IMAGE_IMPLEMENTATION` / `STB_IMAGE_WRITE_IMPLEMENTATION` definitions live in `third_party/stb_impl/stb_impl.cpp`,
so the implementation is compiled once and the rest of the tree includes only the declarations.
