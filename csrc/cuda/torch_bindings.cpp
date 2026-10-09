// Copyright (c) 2026 BAAI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "registration.h"

namespace vllm_fl {

torch::Tensor weak_ref_tensor_cuda(torch::Tensor& tensor);
bool cuda_eventfd_completion_supported();
int64_t enqueue_cuda_eventfd_completion(int64_t stream_ptr, int64_t event_fd);

}  // namespace vllm_fl

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

// vLLM may have already populated _C. Reuse its weak-ref operator instead of
// defining the namespace/schema twice when loading the completion extension.
#define TORCH_LIBRARY_FRAGMENT_EXPAND(NAME, MODULE) TORCH_LIBRARY_FRAGMENT(NAME, MODULE)
TORCH_LIBRARY_FRAGMENT_EXPAND(TORCH_EXTENSION_NAME, ops) {
  if (!c10::Dispatcher::singleton().findSchema({"_C::weak_ref_tensor", ""})) {
    ops.def("weak_ref_tensor(Tensor input) -> Tensor");
    ops.impl("weak_ref_tensor", c10::kCUDA, &vllm_fl::weak_ref_tensor_cuda);
  }
}

// Completion notification has no Tensor argument, so register a catch-all
// implementation in a plugin-owned namespace instead of relying on dispatch.
TORCH_LIBRARY_FRAGMENT(vllm_fl, ops) {
  ops.def("cuda_eventfd_completion_supported() -> bool");
  ops.impl("cuda_eventfd_completion_supported",
           &vllm_fl::cuda_eventfd_completion_supported);
  ops.def("enqueue_cuda_eventfd_completion(int stream_ptr, int event_fd) -> int");
  ops.impl("enqueue_cuda_eventfd_completion",
           &vllm_fl::enqueue_cuda_eventfd_completion);
}
