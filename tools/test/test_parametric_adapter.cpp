// C2.2a tests for ParametricAdapter module and helper functions.
//
// Covers: weight-count formula, distinct-channel derivation (ungated/gated/blended + de-dup),
// zero-adapter identity, known-weights hand-computed result, exact iterator consumption,
// and Recompute dimension-mismatch rejection.

#include <cassert>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "NAM/wavenet/parametric_adapter.h"
#include "NAM/wavenet/params.h"

namespace test_parametric_adapter
{
namespace
{

// ---- helpers ----------------------------------------------------------------

static nam::wavenet::_FiLMParams film_inactive()
{
  return nam::wavenet::_FiLMParams(false, false);
}

// Build a minimal LayerArrayParams with n layers all sharing the same bottleneck
// and gating_mode. Everything else is set to a reasonable non-crashing default.
static nam::wavenet::LayerArrayParams make_lap(int bottleneck, nam::wavenet::GatingMode mode, int n_layers = 1)
{
  const int channels = bottleneck; // layer1x1 inactive → channels == bottleneck
  nam::activations::ActivationConfig act = nam::activations::ActivationConfig::simple(
    nam::activations::ActivationType::ReLU);
  nam::activations::ActivationConfig empty{};

  std::vector<int> kernel_sizes(n_layers, 1);
  std::vector<int> dilations(n_layers, 1);
  std::vector<nam::activations::ActivationConfig> act_cfgs(n_layers, act);
  std::vector<nam::wavenet::GatingMode> modes(n_layers, mode);
  std::vector<nam::activations::ActivationConfig> sec_cfgs(n_layers, empty);

  auto fp = film_inactive();
  return nam::wavenet::LayerArrayParams(
    /*input_size=*/1, /*condition_size=*/1, /*head_size=*/1, /*head_kernel_size=*/1,
    channels, bottleneck,
    std::move(kernel_sizes), std::move(dilations),
    std::move(act_cfgs), std::move(modes),
    /*head_bias=*/false, /*groups_input=*/1, /*groups_input_mixin=*/1,
    nam::wavenet::Layer1x1Params(false, 1),
    nam::wavenet::Head1x1Params(false, 1, 1),
    std::move(sec_cfgs),
    fp, fp, fp, fp, fp, fp, fp, fp);
}

// ---- test_weight_count_formula ----------------------------------------------

void test_weight_count_formula()
{
  // distinct C = [4, 8], param_dim = 2
  // expected = (2*4*2 + 2*4) + (2*8*2 + 2*8) = 24 + 48 = 72
  std::vector<int> distinct_C = {4, 8};
  int result = nam::wavenet::parametric_adapter_weight_count(distinct_C, 2);
  assert(result == 72);

  // distinct C = [3], param_dim = 1
  // expected = 2*3*1 + 2*3 = 6 + 6 = 12
  result = nam::wavenet::parametric_adapter_weight_count({3}, 1);
  assert(result == 12);

  // empty → 0
  result = nam::wavenet::parametric_adapter_weight_count({}, 5);
  assert(result == 0);
}

// ---- test_distinct_channel_sizes --------------------------------------------

void test_distinct_channel_sizes()
{
  using nam::wavenet::GatingMode;

  // One ungated layer (bottleneck=4) → C=4
  // One gated layer (bottleneck=4) → C=8
  // One blended layer (bottleneck=4) → C=8  (duplicate — should be de-duped)
  std::vector<nam::wavenet::LayerArrayParams> laps;
  laps.push_back(make_lap(4, GatingMode::NONE));
  laps.push_back(make_lap(4, GatingMode::GATED));
  laps.push_back(make_lap(4, GatingMode::BLENDED));

  auto sizes = nam::wavenet::parametric_distinct_channel_sizes(laps);
  // Expect sorted, de-duped: [4, 8]
  assert(sizes.size() == 2);
  assert(sizes[0] == 4);
  assert(sizes[1] == 8);
}

void test_distinct_channel_sizes_multi_layer_array()
{
  using nam::wavenet::GatingMode;

  // Array 0: 2 NONE layers at bottleneck=6 → C=6, C=6
  // Array 1: 1 GATED layer at bottleneck=3 → C=6  (same value, de-dup)
  // Array 2: 1 NONE layer at bottleneck=2 → C=2
  std::vector<nam::wavenet::LayerArrayParams> laps;
  laps.push_back(make_lap(6, GatingMode::NONE, 2));
  laps.push_back(make_lap(3, GatingMode::GATED, 1));
  laps.push_back(make_lap(2, GatingMode::NONE, 1));

  auto sizes = nam::wavenet::parametric_distinct_channel_sizes(laps);
  // Expect sorted, de-duped: [2, 6]
  assert(sizes.size() == 2);
  assert(sizes[0] == 2);
  assert(sizes[1] == 6);
}

// ---- test_zero_adapter_identity ---------------------------------------------

void test_zero_adapter_identity()
{
  // All-zero weights: gamma=0, beta=0 → one_plus_gamma=1, beta=0 → z unchanged.
  const int C = 3, P = 2;
  nam::wavenet::ParametricAdapter adapter(C, P);

  // Build zero weight vector and load
  std::vector<float> weights(2 * C * P + 2 * C, 0.0f);
  auto it = weights.begin();
  adapter.load_weights_(it);
  assert(it == weights.end());

  // Recompute with arbitrary params
  std::vector<float> params = {1.5f, -0.7f};
  adapter.Recompute(std::span<const float>(params));

  // Apply to a known matrix
  Eigen::MatrixXf z(C, 4);
  z << 1.0f, 2.0f, 3.0f, 4.0f,
       5.0f, 6.0f, 7.0f, 8.0f,
       9.0f, 10.0f, 11.0f, 12.0f;
  Eigen::MatrixXf z_orig = z;

  adapter.Apply(z, 4);
  assert(z.isApprox(z_orig, 1e-5f));
}

// ---- test_known_weights -----------------------------------------------------

void test_known_weights()
{
  // C=2, P=2
  // gamma_w = [[1,0],[0,1]] row-major → weights: 1,0,0,1
  // gamma_b = [0.1, 0.2]
  // beta_w  = [[0.5,0],[0,0.5]] row-major → weights: 0.5,0,0,0.5
  // beta_b  = [0.3, 0.4]
  //
  // params = [1.0, 2.0]
  // one_plus_gamma = 1 + [[1,0],[0,1]]*[1,2] + [0.1,0.2]
  //               = 1 + [1,2] + [0.1,0.2] = [2.1, 3.2]
  // beta           = [[0.5,0],[0,0.5]]*[1,2] + [0.3,0.4]
  //               = [0.5, 1.0] + [0.3, 0.4] = [0.8, 1.4]
  //
  // z = [[3.0],[2.0]]  (2×1)
  // after Apply col 0:
  //   row 0: 2.1*3.0 + 0.8 = 7.1
  //   row 1: 3.2*2.0 + 1.4 = 7.8

  const int C = 2, P = 2;
  nam::wavenet::ParametricAdapter adapter(C, P);

  std::vector<float> weights = {
    1.0f, 0.0f,   // gamma_w row 0
    0.0f, 1.0f,   // gamma_w row 1
    0.1f, 0.2f,   // gamma_b
    0.5f, 0.0f,   // beta_w row 0
    0.0f, 0.5f,   // beta_w row 1
    0.3f, 0.4f    // beta_b
  };
  auto it = weights.begin();
  adapter.load_weights_(it);
  assert(it == weights.end());

  std::vector<float> params = {1.0f, 2.0f};
  adapter.Recompute(std::span<const float>(params));

  Eigen::MatrixXf z(2, 1);
  z(0, 0) = 3.0f;
  z(1, 0) = 2.0f;

  adapter.Apply(z, 1);

  assert(std::fabs(z(0, 0) - 7.1f) < 1e-5f);
  assert(std::fabs(z(1, 0) - 7.8f) < 1e-5f);
}

// ---- test_iterator_consumption ----------------------------------------------

void test_iterator_consumption()
{
  // Verify load_weights_ consumes exactly 2*C*P + 2*C floats and leaves
  // a sentinel value immediately after the adapter slice untouched.
  const int C = 3, P = 2;
  const int adapter_weights = 2 * C * P + 2 * C; // 12 + 6 = 18

  constexpr float sentinel = 99999.0f;
  std::vector<float> weights(adapter_weights + 1, 0.0f);
  weights.back() = sentinel;

  nam::wavenet::ParametricAdapter adapter(C, P);
  auto it = weights.begin();
  adapter.load_weights_(it);

  // Iterator must now point to the sentinel
  assert(it == weights.begin() + adapter_weights);
  assert(*it == sentinel);
}

// ---- test_recompute_dim_mismatch --------------------------------------------

void test_recompute_dim_mismatch()
{
  nam::wavenet::ParametricAdapter adapter(4, 3);
  std::vector<float> wrong_params = {1.0f, 2.0f}; // size 2, expected 3
  bool threw = false;
  try
  {
    adapter.Recompute(std::span<const float>(wrong_params));
  }
  catch (const std::invalid_argument&)
  {
    threw = true;
  }
  assert(threw);
}

} // namespace
} // namespace test_parametric_adapter

void run_parametric_adapter_tests()
{
  test_parametric_adapter::test_weight_count_formula();
  test_parametric_adapter::test_distinct_channel_sizes();
  test_parametric_adapter::test_distinct_channel_sizes_multi_layer_array();
  test_parametric_adapter::test_zero_adapter_identity();
  test_parametric_adapter::test_known_weights();
  test_parametric_adapter::test_iterator_consumption();
  test_parametric_adapter::test_recompute_dim_mismatch();
}
