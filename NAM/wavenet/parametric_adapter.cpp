#include "parametric_adapter.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace nam
{
namespace wavenet
{

ParametricAdapter::ParametricAdapter(int C, int P)
: _C(C)
, _P(P)
, _gamma_w(C, P)
, _gamma_b(C)
, _beta_w(C, P)
, _beta_b(C)
, _one_plus_gamma(C)
, _beta(C)
{
  _gamma_w.setZero();
  _gamma_b.setZero();
  _beta_w.setZero();
  _beta_b.setZero();
  // Default scratch: identity (1+0)*z + 0 = z
  _one_plus_gamma.setOnes();
  _beta.setZero();
}

void ParametricAdapter::load_weights_(std::vector<float>::iterator& it)
{
  // gamma_w: C×P row-major
  for (int r = 0; r < _C; ++r)
    for (int c = 0; c < _P; ++c)
      _gamma_w(r, c) = *(it++);
  // gamma_b: C
  for (int r = 0; r < _C; ++r)
    _gamma_b(r) = *(it++);
  // beta_w: C×P row-major
  for (int r = 0; r < _C; ++r)
    for (int c = 0; c < _P; ++c)
      _beta_w(r, c) = *(it++);
  // beta_b: C
  for (int r = 0; r < _C; ++r)
    _beta_b(r) = *(it++);
}

void ParametricAdapter::Recompute(std::span<const float> params)
{
  if ((int)params.size() != _P)
    throw std::invalid_argument(
      "ParametricAdapter::Recompute: expected " + std::to_string(_P) + " params, got "
      + std::to_string((int)params.size()));
  Eigen::Map<const Eigen::VectorXf> p(params.data(), _P);
  _one_plus_gamma = _gamma_w * p + _gamma_b;
  _one_plus_gamma.array() += 1.0f;
  _beta = _beta_w * p + _beta_b;
}

void ParametricAdapter::Apply(Eigen::Ref<Eigen::MatrixXf> z, int num_frames) const
{
  // Per-channel broadcast, allocation-free: multiply then add bias.
  for (int f = 0; f < num_frames; ++f)
  {
    z.col(f).array() *= _one_plus_gamma.array();
    z.col(f).array() += _beta.array();
  }
}

std::vector<int> parametric_distinct_channel_sizes(const std::vector<LayerArrayParams>& layer_array_params)
{
  std::vector<int> sizes;
  for (const auto& arr : layer_array_params)
  {
    const int bottleneck = arr.bottleneck;
    for (const auto& mode : arr.gating_modes)
    {
      const int C = (mode == GatingMode::NONE) ? bottleneck : 2 * bottleneck;
      sizes.push_back(C);
    }
  }
  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
  return sizes;
}

int parametric_adapter_weight_count(const std::vector<int>& distinct_channel_sizes, int param_dim)
{
  int total = 0;
  for (int C : distinct_channel_sizes)
    total += 2 * C * param_dim + 2 * C;
  return total;
}

} // namespace wavenet
} // namespace nam
