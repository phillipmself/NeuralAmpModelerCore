// CP5 synthetic tests: zero-adapter parity + known-adapter behavior change (C2.2d).
// CP6 seam tests: IParametricControl runtime parameter-control seam (C2.1).
//
// CP5 proves that the parametric adapter hook is behaviorally inert when all
// adapter weights are zero (identity for any param vector), and that a known
// nonzero adapter produces a hand-computable, deterministic output change.
//
// CP5a/b use the A2 nano inner config and deterministic random inner weights W:
//   reference = generic WaveNet from parse_config_json + WaveNetConfig::create(W)
//   test      = ParametricWaveNet from W ++ zeros(adapter_count)
//   assertion = bit-exact sample equality for any param vector (zero adapter is
//               identity for all p: z' = (1+0)*z + 0 = z).
//
// CP5c/d use a tiny 1-layer, 1-channel, K=1, P=1 fixture whose forward pass is
// hand-traceable:
//   weights: rechannel=1, conv=(1,0), mixin=1, 1x1=(1,0), head_rechannel=1, head_scale=1
//   output for input x (x > 0): 2x  (no adapter)
//   with adapter beta_w = w_b, param p: 2x + w_b * p
//
// CP6 uses a local TestParametricDSP to prove IParametricControl seam semantics
// independently of real WaveNet inference.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "NAM/wavenet/model.h"
#include "allocation_tracking.h"
#include "parametric_test_fixtures.h"

namespace test_parametric_dsp
{
namespace
{

// ============================================================================
// Shared helpers
// ============================================================================

struct A2ParityModels
{
  std::unique_ptr<nam::DSP> ref_dsp;
  std::unique_ptr<nam::DSP> param_dsp;
};

std::vector<float> make_deterministic_weights(const int count, const uint32_t seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
  std::vector<float> w(count);
  for (auto& x : w)
    x = dist(rng);
  return w;
}

std::vector<NAM_SAMPLE> make_random_input(const int count, const uint32_t seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<NAM_SAMPLE> dist(-0.25f, 0.25f);
  std::vector<NAM_SAMPLE> input(count);
  for (auto& sample : input)
    sample = dist(rng);
  return input;
}

// Reset the DSP (triggers prewarm), then run input through in blocks of block_size.
// Returns the full output vector. Both models must be given the same block_size for
// their prewarm and processing to be comparable.
std::vector<NAM_SAMPLE> run_blocks(nam::DSP& dsp, const std::vector<NAM_SAMPLE>& input, int block_size)
{
  dsp.Reset(48000.0, block_size);
  std::vector<NAM_SAMPLE> out(input.size(), static_cast<NAM_SAMPLE>(0));
  int pos = 0;
  const int total = static_cast<int>(input.size());
  while (pos < total)
  {
    const int n = std::min(block_size, total - pos);
    NAM_SAMPLE* in_ptr = const_cast<NAM_SAMPLE*>(input.data() + pos);
    NAM_SAMPLE* out_ptr = out.data() + pos;
    NAM_SAMPLE* in_arr[] = {in_ptr};
    NAM_SAMPLE* out_arr[] = {out_ptr};
    dsp.process(in_arr, out_arr, n);
    pos += n;
  }
  return out;
}

A2ParityModels make_zero_adapter_a2_models()
{
  const int inner_count = test_parametric_fixtures::compute_a2_inner_weight_count();
  const auto inner_weights = make_deterministic_weights(inner_count, 42);

  auto ref_config = nam::wavenet::parse_config_json(test_parametric_fixtures::build_a2_inner_config(), 48000.0);
  auto ref_dsp = ref_config.create(std::vector<float>(inner_weights), 48000.0);

  std::vector<float> full_weights = inner_weights;
  full_weights.resize(
    static_cast<size_t>(inner_count + test_parametric_fixtures::kA2AdapterCount), 0.0f);
  auto param_dsp = nam::get_dsp(test_parametric_fixtures::build_a2_parametric_fixture(full_weights));

  return {.ref_dsp = std::move(ref_dsp), .param_dsp = std::move(param_dsp)};
}

void assert_sample_exact_equal(const std::vector<NAM_SAMPLE>& expected, const std::vector<NAM_SAMPLE>& actual)
{
  assert(expected.size() == actual.size());
  for (size_t i = 0; i < expected.size(); i++)
    assert(expected[i] == actual[i]);
}

// ============================================================================
// Tiny 1-layer, 1-channel, K=1, ReLU, P=1 fixture (CP5c/d)
//
// Weight layout (8 inner floats + 4 adapter floats):
//   inner: rechannel(1.0), conv(1.0, 0.0), mixin(1.0), 1x1(1.0, 0.0),
//          head_rechannel(1.0), head_scale(1.0)
//   adapter for C=1, P=1: [gamma_w, gamma_b, beta_w, beta_b]
//
// Forward pass for input x > 0 (K=1 → memoryless, no causal buffer effect):
//   rechannel_out = 1*x
//   _conv out = 1*(rechannel_out) + 0 = x
//   _mixin out = 1*x (condition = raw audio)
//   _z = x + x = 2x
//   [adapter: _z += beta(p) = beta_w * p  (gamma=0 everywhere)]
//   z_act = ReLU(2x + beta_w * p) = 2x + beta_w * p  (positive)
//   head_inputs = z_act  (_skip_head_copy=true: no head1x1, GatingMode::NONE)
//   head_rechannel_out = 1 * z_act = z_act  (K=1, no bias)
//   output = head_scale * z_act = z_act
//   => output = 2x + beta_w * p
// ============================================================================

// Inner weights for the tiny fixture (8 floats, all weights=1, biases=0):
//   rechannel(1.0), conv_w(1.0), conv_b(0.0), mixin(1.0),
//   1x1_w(1.0), 1x1_b(0.0), head_rechannel(1.0), head_scale(1.0)
static std::vector<float> make_tiny_inner_weights()
{
  return {1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
}

// Build a JSON-loaded ParametricWaveNet from the tiny fixture with the given beta_w.
// This keeps the hand-computable behavior test on the real public load path, including
// weight splitting and adapter-tail ordering.
static std::unique_ptr<nam::DSP> make_tiny_parametric(float beta_w)
{
  using nlohmann::json;

  json film_inactive = {{"active", false}, {"shift", true}, {"groups", 1}};
  json layer;
  layer["input_size"] = 1;
  layer["condition_size"] = 1;
  layer["channels"] = 1;
  layer["bottleneck"] = 1;
  layer["kernel_sizes"] = json::array({1});
  layer["dilations"] = json::array({1});
  layer["activation"] = json::array({{{"type", "ReLU"}}});
  layer["gating_mode"] = json::array({"none"});
  layer["secondary_activation"] = json::array({nullptr});
  layer["head"] = {{"out_channels", 1}, {"kernel_size", 1}, {"bias", false}};
  layer["head1x1"] = {{"active", false}, {"out_channels", 1}, {"groups", 1}};
  layer["layer1x1"] = {{"active", true}, {"groups", 1}};
  layer["conv_pre_film"] = film_inactive;
  layer["conv_post_film"] = film_inactive;
  layer["input_mixin_pre_film"] = film_inactive;
  layer["input_mixin_post_film"] = film_inactive;
  layer["activation_pre_film"] = film_inactive;
  layer["activation_post_film"] = film_inactive;
  layer["layer1x1_post_film"] = film_inactive;
  layer["head1x1_post_film"] = film_inactive;
  layer["groups_input"] = 1;
  layer["groups_input_mixin"] = 1;

  json config;
  config["layers"] = json::array({layer});
  config["head_scale"] = 1.0f;
  config["param_names"] = json::array({"gain"});
  config["param_dim"] = 1;
  config["nominal_params"] = json::array({0.0f});

  std::vector<float> weights = make_tiny_inner_weights();
  weights.insert(weights.end(), {0.0f, 0.0f, beta_w, 0.0f});

  json model_json;
  model_json["version"] = "0.7.0";
  model_json["architecture"] = "ParametricWaveNet";
  model_json["config"] = config;
  model_json["weights"] = weights;
  model_json["sample_rate"] = 48000;
  return nam::get_dsp(model_json);
}

// ============================================================================
// TestParametricDSP (CP6 helper — unchanged)
// ============================================================================

class TestParametricDSP : public nam::DSP, public nam::IParametricControl
{
public:
  TestParametricDSP(int param_dim, const std::vector<float>& nominal_params)
  : nam::DSP(1, 1, 48000.0)
  , param_dim_(param_dim)
  , current_params_(nominal_params)
  {
    assert((int)nominal_params.size() == param_dim);
  }

  void SetParams(std::span<const float> params) override
  {
    if ((int)params.size() != param_dim_)
      throw std::invalid_argument(
        "SetParams: expected " + std::to_string(param_dim_)
        + " params, got " + std::to_string((int)params.size()));
    std::copy(params.begin(), params.end(), current_params_.begin());
  }

  std::span<const float> GetParams() const override { return current_params_; }
  int ParamDim() const override { return param_dim_; }

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames) override
  {
    const float offset = param_dim_ > 0 ? current_params_[0] : 0.0f;
    for (int f = 0; f < num_frames; f++)
      output[0][f] = input[0][f] + static_cast<NAM_SAMPLE>(offset);
  }

private:
  int param_dim_;
  std::vector<float> current_params_;
};

} // namespace

// ============================================================================
// CP5a: zero-adapter parity — nominal params
//
// Load W as a plain WaveNet (generic path via parse_config_json + create) and
// W ++ zeros(adapter_count) as a ParametricWaveNet (via get_dsp). The zero
// adapter is identity for all p, so the outputs must be bit-exact.
// ============================================================================

void test_cp5a_zero_adapter_parity_nominal()
{
  auto models = make_zero_adapter_a2_models();

  // Build a realistic multi-block input.
  constexpr int kTotal = 1024, kBlock = 64;
  const auto input = make_random_input(kTotal, 7);

  // run_blocks resets each DSP (triggering identical prewarm with zero input)
  // then processes the same input. Zero adapter is identity regardless of params.
  const auto ref_out = run_blocks(*models.ref_dsp, input, kBlock);
  const auto param_out = run_blocks(*models.param_dsp, input, kBlock);
  assert_sample_exact_equal(ref_out, param_out);
}

// ============================================================================
// CP5b: zero-adapter parity — arbitrary non-nominal params
//
// Same zero-adapter ParametricWaveNet as CP5a, but params are changed to an
// arbitrary vector before the run. The zero adapter remains identity for all p:
// gamma(p) = 0*p + 0 = 0, beta(p) = 0*p + 0 = 0, so z' = z still.
// ============================================================================

void test_cp5b_zero_adapter_parity_after_setparams()
{
  auto models = make_zero_adapter_a2_models();

  // Reset both with identical block size to synchronize prewarm state.
  constexpr int kBlock = 64;
  models.ref_dsp->Reset(48000.0, kBlock);
  models.param_dsp->Reset(48000.0, kBlock);

  // Push a non-nominal param vector. Zero adapter weights → still identity.
  auto* ctrl = dynamic_cast<nam::IParametricControl*>(models.param_dsp.get());
  assert(ctrl != nullptr);
  const float arbitrary_params[] = {1.5f};
  ctrl->SetParams(arbitrary_params);

  // Build input and run both manually (no extra Reset).
  constexpr int kTotal = 1024;
  const auto input = make_random_input(kTotal, 7);

  std::vector<NAM_SAMPLE> ref_out(kTotal, 0), param_out(kTotal, 0);
  int pos = 0;
  while (pos < kTotal)
  {
    const int n = std::min(kBlock, kTotal - pos);
    NAM_SAMPLE* in_ptr = const_cast<NAM_SAMPLE*>(input.data() + pos);
    NAM_SAMPLE* ref_ptr = ref_out.data() + pos;
    NAM_SAMPLE* par_ptr = param_out.data() + pos;
    NAM_SAMPLE* in_arr[] = {in_ptr};
    NAM_SAMPLE* ref_arr[] = {ref_ptr};
    NAM_SAMPLE* par_arr[] = {par_ptr};
    models.ref_dsp->process(in_arr, ref_arr, n);
    models.param_dsp->process(in_arr, par_arr, n);
    pos += n;
  }

  // Zero adapter is identity for all p: still bit-exact vs. reference.
  assert_sample_exact_equal(ref_out, param_out);
}

// ============================================================================
// CP5c: known adapter matches hand-computed output
//
// Tiny fixture, beta_w = 2.0, p = 0.5, input x = 1.0 (constant, positive).
// Hand-computed: output = 2*x + beta_w * p = 2.0 + 2.0*0.5 = 3.0 per frame.
// All arithmetic is exact in FP32 (powers of 2, representable values).
// ============================================================================

void test_cp5c_known_adapter_matches_hand_computed_output()
{
  constexpr float kBetaW = 2.0f;
  constexpr float kP = 0.5f;
  constexpr float kX = 1.0f;
  // Hand-computed expected output = 2*x + beta_w * p = 2.0 + 1.0 = 3.0
  constexpr float kExpected = 2.0f * kX + kBetaW * kP; // = 3.0f

  auto model = make_tiny_parametric(kBetaW);

  constexpr int kBlock = 32;
  model->Reset(48000.0, kBlock);

  auto* ctrl = dynamic_cast<nam::IParametricControl*>(model.get());
  assert(ctrl != nullptr);
  const float params[] = {kP};
  ctrl->SetParams(params);

  std::vector<NAM_SAMPLE> input(kBlock, static_cast<NAM_SAMPLE>(kX));
  std::vector<NAM_SAMPLE> output(kBlock, 0.0f);
  NAM_SAMPLE* in_arr[] = {input.data()};
  NAM_SAMPLE* out_arr[] = {output.data()};
  model->process(in_arr, out_arr, kBlock);

  constexpr float kEps = 1e-5f;
  for (int i = 0; i < kBlock; i++)
    assert(std::abs(static_cast<float>(output[i]) - kExpected) < kEps);
}

// ============================================================================
// CP5d: known adapter — param change affects subsequent block
//
// Same tiny fixture. Process block 1 with p_a, then SetParams to p_b, process
// block 2. Hand-computed:
//   block 1: 2*x + beta_w * p_a = 3.0
//   block 2: 2*x + beta_w * p_b = 4.0
// ============================================================================

void test_cp5d_known_adapter_param_change_affects_subsequent_block()
{
  constexpr float kBetaW = 2.0f;
  constexpr float kX = 1.0f;
  constexpr float kPa = 0.5f; // block-1 params
  constexpr float kPb = 1.0f; // block-2 params (different)
  constexpr float kExpected1 = 2.0f * kX + kBetaW * kPa; // = 3.0
  constexpr float kExpected2 = 2.0f * kX + kBetaW * kPb; // = 4.0

  auto model = make_tiny_parametric(kBetaW);

  constexpr int kBlock = 32;
  model->Reset(48000.0, kBlock);

  auto* ctrl = dynamic_cast<nam::IParametricControl*>(model.get());
  assert(ctrl != nullptr);

  std::vector<NAM_SAMPLE> input(kBlock, static_cast<NAM_SAMPLE>(kX));
  std::vector<NAM_SAMPLE> out1(kBlock, 0.0f), out2(kBlock, 0.0f);
  NAM_SAMPLE* in_arr[] = {input.data()};
  NAM_SAMPLE* out_arr1[] = {out1.data()};
  NAM_SAMPLE* out_arr2[] = {out2.data()};

  // Block 1 with p_a.
  const float params_a[] = {kPa};
  ctrl->SetParams(params_a);
  model->process(in_arr, out_arr1, kBlock);

  // Block 2 with p_b.
  const float params_b[] = {kPb};
  ctrl->SetParams(params_b);
  model->process(in_arr, out_arr2, kBlock);

  assert(std::abs(kExpected1 - kExpected2) > 1e-4f); // blocks must actually differ

  constexpr float kEps = 1e-5f;
  for (int i = 0; i < kBlock; i++)
  {
    assert(std::abs(static_cast<float>(out1[i]) - kExpected1) < kEps);
    assert(std::abs(static_cast<float>(out2[i]) - kExpected2) < kEps);
  }
}

// ============================================================================
// CP6: IParametricControl seam tests (unchanged from C2.1)
// ============================================================================

// CP6a: nominal_params seed the initial parameter state at construction.
void test_cp6a_nominal_params_seed_initial_state()
{
  const std::vector<float> nominal = {0.5f, -0.25f};
  TestParametricDSP dsp(2, nominal);
  const auto& params = dsp.GetParams();
  assert((int)params.size() == 2);
  assert(std::abs(params[0] - 0.5f) < 1e-6f);
  assert(std::abs(params[1] - (-0.25f)) < 1e-6f);
}

// CP6b: the last committed param vector is what GetParams() returns.
void test_cp6b_last_committed_params()
{
  TestParametricDSP dsp(1, {0.5f});
  assert(std::abs(dsp.GetParams()[0] - 0.5f) < 1e-6f);

  const float p1[] = {1.0f};
  dsp.SetParams(p1);
  assert(std::abs(dsp.GetParams()[0] - 1.0f) < 1e-6f);

  const float p2[] = {0.0f};
  dsp.SetParams(p2);
  assert(std::abs(dsp.GetParams()[0] - 0.0f) < 1e-6f);
}

// CP6c: changing params between two process() calls changes the output.
void test_cp6c_param_change_affects_output()
{
  TestParametricDSP dsp(1, {0.5f});

  NAM_SAMPLE in_buf[4] = {};
  NAM_SAMPLE out1[4] = {};
  NAM_SAMPLE out2[4] = {};
  NAM_SAMPLE* inputs[1] = {in_buf};
  NAM_SAMPLE* outputs1[1] = {out1};
  NAM_SAMPLE* outputs2[1] = {out2};

  // First block: params = nominal (0.5)
  dsp.process(inputs, outputs1, 4);

  // Update params before second block
  const float new_params[] = {0.0f};
  dsp.SetParams(new_params);

  // Second block: params = 0.0
  dsp.process(inputs, outputs2, 4);

  // All samples in block 1 must differ from block 2
  for (int f = 0; f < 4; f++)
    assert(out1[f] != out2[f]);
  // Spot-check expected values (input is 0; output = 0 + offset)
  assert(std::abs(static_cast<float>(out1[0]) - 0.5f) < 1e-5f);
  assert(std::abs(static_cast<float>(out2[0]) - 0.0f) < 1e-5f);
}

// CP6d: repeated SetParams + process() calls are allocation-free after construction.
void test_cp6d_no_allocation_during_steady_state()
{
  TestParametricDSP dsp(2, {0.5f, 0.5f});

  NAM_SAMPLE in_buf[8] = {};
  NAM_SAMPLE out_buf[8] = {};
  NAM_SAMPLE* inputs[1] = {in_buf};
  NAM_SAMPLE* outputs[1] = {out_buf};

  // Pre-allocate param storage and do a warmup round before tracking starts.
  const float params_a[] = {0.3f, 0.1f};
  const float params_b[] = {0.7f, -0.2f};
  dsp.SetParams(params_a);
  dsp.process(inputs, outputs, 8);

  allocation_tracking::run_allocation_test_no_allocations(
    nullptr,
    [&]() {
      dsp.SetParams(params_a);
      dsp.process(inputs, outputs, 8);
      dsp.SetParams(params_b);
      dsp.process(inputs, outputs, 8);
    },
    nullptr,
    "CP6d: SetParams + process() steady-state");
}

// CP6e: SetParams with wrong dimension throws std::invalid_argument.
void test_cp6e_dim_mismatch_throws()
{
  TestParametricDSP dsp(2, {0.5f, 0.5f});
  bool caught = false;
  try
  {
    const float bad[] = {0.5f}; // 1 instead of 2
    dsp.SetParams(bad);
    assert(false); // must not reach here
  }
  catch (const std::invalid_argument& e)
  {
    const std::string msg = e.what();
    assert(msg.find("2") != std::string::npos);
    assert(msg.find("1") != std::string::npos);
    caught = true;
  }
  assert(caught);
}

// CP6f: IParametricControl is discoverable via dynamic_cast from a DSP pointer.
void test_cp6f_dynamic_cast_discovery()
{
  std::unique_ptr<nam::DSP> dsp = std::make_unique<TestParametricDSP>(1, std::vector<float>{0.5f});
  auto* ctrl = dynamic_cast<nam::IParametricControl*>(dsp.get());
  assert(ctrl != nullptr);
  assert(ctrl->ParamDim() == 1);
  assert(std::abs(ctrl->GetParams()[0] - 0.5f) < 1e-6f);
}

} // namespace test_parametric_dsp

void run_parametric_dsp_tests()
{
  // CP5: synthetic zero-adapter parity and known-adapter behavior-change evidence (C2.2d)
  test_parametric_dsp::test_cp5a_zero_adapter_parity_nominal();
  test_parametric_dsp::test_cp5b_zero_adapter_parity_after_setparams();
  test_parametric_dsp::test_cp5c_known_adapter_matches_hand_computed_output();
  test_parametric_dsp::test_cp5d_known_adapter_param_change_affects_subsequent_block();
  // CP6: IParametricControl seam tests (C2.1)
  test_parametric_dsp::test_cp6a_nominal_params_seed_initial_state();
  test_parametric_dsp::test_cp6b_last_committed_params();
  test_parametric_dsp::test_cp6c_param_change_affects_output();
  test_parametric_dsp::test_cp6d_no_allocation_during_steady_state();
  test_parametric_dsp::test_cp6e_dim_mismatch_throws();
  test_parametric_dsp::test_cp6f_dynamic_cast_discovery();
}
