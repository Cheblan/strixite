// The stb implementations, compiled once (third_party/stb/README.md). The engine TU (serve/vision_preprocess.cpp)
// includes only the declarations; tests/ includes the write header's declarations for its encode->decode round-trip.

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>  // third_party/stb (on the include path)

#ifdef STRIX_STB_IMPL_TEST_ONLY
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#endif
