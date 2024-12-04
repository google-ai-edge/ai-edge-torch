#ifndef THIRD_PARTY_PY_AI_EDGE_TORCH_GENERATIVE_EXAMPLES_CPP_UTILS_H_
#define THIRD_PARTY_PY_AI_EDGE_TORCH_GENERATIVE_EXAMPLES_CPP_UTILS_H_

#include <cstddef>

#include "tensorflow/lite/util.h"

namespace ai_edge_torch::examples {

// A minimal check macro.
#define MINIMAL_CHECK(x)                                     \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

// TF Lite requires all buffers (including external buffers used for KV cache
// here) be `tflite::kDefaultTensorAlignment` aligned. To ensure that, we use
// this custom allocator. Please use with caution as different platforms may
// have different alignment requirements.
template <typename T>
class AlignedAllocator {
 public:
  using value_type = T;

  T* allocate(std::size_t n) {
    void* ptr;
    std::size_t size = n * sizeof(T);
    std::size_t padding = tflite::kDefaultTensorAlignment -
                          (size % tflite::kDefaultTensorAlignment);
    size += padding;
    int ret = posix_memalign(&ptr, tflite::kDefaultTensorAlignment, size);
    if (ret != 0) {
      return nullptr;
    }
    return static_cast<T*>(ptr);
  };

  void deallocate(T* ptr, std::size_t n) { free(ptr); }
};

}  // namespace ai_edge_torch::examples

#endif  // THIRD_PARTY_PY_AI_EDGE_TORCH_GENERATIVE_EXAMPLES_CPP_UTILS_H_
