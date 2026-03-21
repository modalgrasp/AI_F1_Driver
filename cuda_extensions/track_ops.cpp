#include <torch/extension.h>

torch::Tensor pairwise_distance_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor reward_kernel_cuda(torch::Tensor progress, torch::Tensor offtrack, torch::Tensor collision);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pairwise_distance_cuda", &pairwise_distance_cuda, "Pairwise distance CUDA");
  m.def("reward_kernel_cuda", &reward_kernel_cuda, "Reward kernel CUDA");
}
