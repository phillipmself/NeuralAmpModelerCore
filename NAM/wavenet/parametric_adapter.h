#pragma once

#include <span>
#include <vector>

#include <Eigen/Dense>

#include "params.h"

namespace nam
{
namespace wavenet
{

/// \brief Per-channel affine adapter: z' = (1 + gamma(p)) * z + beta(p)
///
/// Holds gamma_w (C×P), gamma_b (C), beta_w (C×P), beta_b (C) from the adapter-tail
/// weight blob (AD-C6 layout). Pre-sized scratch vectors make Recompute/Apply
/// allocation-free after construction.
class ParametricAdapter
{
public:
  /// \brief Construct and pre-size storage for channel count C and param dim P.
  ParametricAdapter(int C, int P);

  /// \brief Load weights from iterator; advances by 2*C*P + 2*C floats exactly.
  ///
  /// Weight order (AD-C6): gamma_w (C×P) row-major, gamma_b (C),
  ///                        beta_w (C×P) row-major,  beta_b (C).
  void load_weights_(std::vector<float>::iterator& it);

  /// \brief Recompute scratch from params. Throws std::invalid_argument if params.size() != P.
  ///
  /// Computes: one_plus_gamma = 1 + gamma_w * p + gamma_b
  ///                     beta = beta_w * p + beta_b
  void Recompute(std::span<const float> params);

  /// \brief Apply in-place to first num_frames columns: z = (1+gamma)*z + beta.
  ///
  /// Per-channel broadcast; no allocation proportional to num_frames.
  void Apply(Eigen::Ref<Eigen::MatrixXf> z, int num_frames) const;

  int C() const { return _C; }
  int P() const { return _P; }

private:
  int _C, _P;
  Eigen::MatrixXf _gamma_w; // C×P
  Eigen::VectorXf _gamma_b; // C
  Eigen::MatrixXf _beta_w;  // C×P
  Eigen::VectorXf _beta_b;  // C
  Eigen::VectorXf _one_plus_gamma; // C  (scratch)
  Eigen::VectorXf _beta;           // C  (scratch)
};

/// \brief Returns sorted, de-duplicated per-layer channel sizes across all layer arrays.
///
/// C = 2*bottleneck for GATED or BLENDED layers, bottleneck for NONE.
std::vector<int> parametric_distinct_channel_sizes(const std::vector<LayerArrayParams>& layer_array_params);

/// \brief Total adapter weight count: sum over distinct C of (2*C*param_dim + 2*C).
int parametric_adapter_weight_count(const std::vector<int>& distinct_channel_sizes, int param_dim);

} // namespace wavenet
} // namespace nam
