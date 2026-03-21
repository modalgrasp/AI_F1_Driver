#include <torch/extension.h>
#include <vector>

torch::Tensor pairwise_distance_cuda(torch::Tensor a, torch::Tensor b) {
  // Minimal baseline implementation. Replace with custom kernel for production tuning.
  return torch::cdist(a, b);
}

torch::Tensor reward_kernel_cuda(torch::Tensor progress, torch::Tensor offtrack, torch::Tensor collision) {
  return 2.0 * progress - 3.0 * offtrack.to(torch::kFloat) - 8.0 * collision.to(torch::kFloat);
}
