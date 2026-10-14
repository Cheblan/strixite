# Images

strixite takes images in chat requests, the way the OpenAI API sends them: an `image_url` content part in a user
message. It's on by default and costs nothing when no image is sent.

## What it needs

Two files from the weights repo that earlier versions didn't use: `vision.strixw` (~0.9 GB, the vision encoder) next
to `weights.strixw`, and the checkpoint's `preprocessor_config.json`. A fresh download has both. If you downloaded
before v0.3, run the same `hf download` command again - it fetches only the new files.

If either file is missing the server refuses to start and names it - it never quietly serves text only. To run text
only instead, set `vision = off` in `strix-server.conf` (or start with `--vision off`): image parts are then a 400 and
nothing extra is loaded.

## What it accepts

| | |
|---|---|
| formats | PNG, JPEG, WebP, GIF (the first frame), BMP, TGA |
| colors | RGB; grayscale and transparency are converted the way the model's own image processor does it - gray copied to all three channels, transparency dropped (not blended onto a background) |
| how they arrive | `data:` URLs only (`data:image/png;base64,...`). The server never fetches a remote URL |
| where | user messages; several images per message and per conversation are fine. Images in system or assistant messages are a 400 |
| not supported | video, HEIC / AVIF, TIFF, SVG - each is a 400 that says what it got |

## Size and cost

Each image is resized to a multiple of 32 pixels on each side, keeping its shape, with the same bicubic,
antialiased filter the model's processor uses. Images larger than about 2.1 megapixels are scaled down to fit -
8,192 patches of 16 x 16, so a 1920 x 1080 screenshot is read as 1920 x 1088 and a 4K image is halved. Every 32 x 32
pixels become one token of the prompt: a 1080p screenshot is about 2,000 tokens, a 448 x 448 image about 200.

The vision encoder runs once per new image (about 0.2 s for a small image, 0.9 s for a full-size screenshot), then the image's
tokens go through the model like any other prompt. Follow-up turns in the same conversation reuse them from the prompt
cache, so a screenshot is paid for once, not every turn.
