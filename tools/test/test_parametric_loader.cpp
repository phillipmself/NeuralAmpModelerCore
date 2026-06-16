// C2.2c: ParametricWaveNet loader tests.
//
// CP1: parser registration guard (has() == true).
// CP2: config-shape negative cases (malformed param_names / param_dim / nominal_params).
// CP3: adapter-tail weight-count validation (negative: too-short blob; positive: exact size).
// CP4: real round-trip — get_dsp(), Reset(), process(), IParametricControl discovery.
//
// See docs/parametric-a2-core/test-plan.md for full CP descriptions.

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

#include "json.hpp"

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "NAM/model_config.h"

namespace test_parametric_loader
{
namespace
{

// A2 nano shape constants — inlined from NAM/wavenet/a2_fast.h so this test
// does not require NAM_ENABLE_A2_FAST to compile.
constexpr int kA2NumLayers = 23;
constexpr int kA2HeadKernelSize = 16;
constexpr float kA2LeakySlope = 0.01f;
constexpr int kA2KernelSizes[kA2NumLayers] = {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6};
constexpr int kA2Dilations[kA2NumLayers] = {1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239};

// Minimal A2 nano (3-channel) inner WaveNet config.
// Matches test_a2_fast::build_a2_config(3) so the fixture shape is stable across tests.
nlohmann::json build_a2_inner_config()
{
  using nlohmann::json;

  const int channels = 3;

  json activation = json::array();
  json gating_mode = json::array();
  json secondary = json::array();
  json kernel_sizes_arr = json::array();
  json dilations_arr = json::array();
  for (int i = 0; i < kA2NumLayers; i++)
  {
    activation.push_back({{"type", "LeakyReLU"}, {"negative_slope", kA2LeakySlope}});
    gating_mode.push_back("none");
    secondary.push_back(nullptr);
    kernel_sizes_arr.push_back(kA2KernelSizes[i]);
    dilations_arr.push_back(kA2Dilations[i]);
  }

  json film_inactive = {{"active", false}, {"shift", true}, {"groups", 1}};

  json layer;
  layer["input_size"] = 1;
  layer["condition_size"] = 1;
  layer["channels"] = channels;
  layer["bottleneck"] = channels;
  layer["kernel_sizes"] = kernel_sizes_arr;
  layer["dilations"] = dilations_arr;
  layer["activation"] = activation;
  layer["gating_mode"] = gating_mode;
  layer["secondary_activation"] = secondary;
  layer["head"] = {{"out_channels", 1}, {"kernel_size", kA2HeadKernelSize}, {"bias", true}};
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
  config["head_scale"] = 0.01f;
  return config;
}

// Exact inner weight count for the A2 nano fixture with channels=3.
// Mirrors test_a2_fast::a2_weight_count(3) using the same constants.
//
// Layout per LayerArray:
//   rechannel Conv1x1(1→3, no bias)      = channels         = 3
//   per layer:
//     conv1d  Conv1D(3→3, K, bias)       = channels*bn*K+bn = 9K+3
//     mixin   Conv1x1(1→bn, no bias)     = bn               = 3
//     layer1x1 Conv1x1(bn→3, bias)       = bn*channels+channels = 12
//                                          per-layer total  = 9K+18
//   head_rechannel Conv1D(bn→1, K=16, bias) = 1*3*16+1      = 49
//   head_scale (1 trailing float)                            = 1
int compute_a2_inner_weight_count()
{
  const int channels = 3;
  const int bn = channels; // bottleneck == channels, gating=none
  int total = channels;    // rechannel: Conv1x1(1, channels, false) = channels*1 weights
  for (int i = 0; i < kA2NumLayers; i++)
  {
    const int K = kA2KernelSizes[i];
    total += bn * channels * K + bn; // conv1d weights + bias
    total += bn;                      // input mixin (no bias)
    total += channels * bn + channels; // layer1x1 + bias
  }
  total += channels * kA2HeadKernelSize + 1; // head rechannel (Conv1D): 3*16+1 = 49
  total += 1;                                // head_scale (read by WaveNet::set_weights_)
  return total;
}

// Adapter weight count for param_dim=1 with distinct_Cs=[3].
// Formula: 2*C*P + 2*C per distinct C.
// For C=3, P=1: 2*3*1 + 2*3 = 12.
constexpr int kAdapterCount = 12;

// Build a JSON fixture with the A2 nano inner config and parametric metadata.
// weights is provided by the caller so CP3/CP4 can control the size.
nlohmann::json build_parametric_fixture(const std::vector<float>& weights)
{
  using nlohmann::json;

  json config = build_a2_inner_config();
  config["param_names"] = json::array({"gain"});
  config["param_dim"] = 1;
  config["nominal_params"] = json::array({0.5});

  json j;
  j["version"] = "0.7.0";
  j["architecture"] = "ParametricWaveNet";
  j["config"] = config;
  j["weights"] = weights;
  j["sample_rate"] = 48000;
  return j;
}

} // namespace

// CP1: "ParametricWaveNet" is registered in the ConfigParserRegistry.
// create() now constructs a real DSP; the not-implemented throw is gone (see CP4 for round-trip).
void test_cp1_parser_is_registered()
{
  assert(nam::ConfigParserRegistry::instance().has("ParametricWaveNet") == true);
}

// --- CP3: adapter-tail weight-count validation (C2.2c) ---

// CP3 negative: a weight blob shorter than the required adapter tail must throw a
// std::runtime_error whose message names both the expected adapter count and the actual total.
void test_cp3_adapter_tail_too_short()
{
  // Provide 5 weights total (far less than kAdapterCount=12 needed for adapter tail).
  const std::vector<float> tiny(5, 0.0f);
  const nlohmann::json j = build_parametric_fixture(tiny);

  bool caught = false;
  try
  {
    nam::get_dsp(j);
    assert(false); // must not reach here
  }
  catch (const std::runtime_error& e)
  {
    const std::string msg = e.what();
    // Message must contain both the expected adapter count (12) and the actual total (5).
    assert(msg.find("12") != std::string::npos);
    assert(msg.find("5") != std::string::npos);
    caught = true;
  }
  assert(caught);
}

// CP3 positive: a correctly-sized weight blob constructs successfully and the model
// is discoverable as IParametricControl.
void test_cp3_correct_size_constructs()
{
  const int inner_count = compute_a2_inner_weight_count();
  const int total_count = inner_count + kAdapterCount;
  const std::vector<float> weights(total_count, 0.0f);
  const nlohmann::json j = build_parametric_fixture(weights);

  auto dsp = nam::get_dsp(j);
  assert(dsp != nullptr);
  auto* ctrl = dynamic_cast<nam::IParametricControl*>(dsp.get());
  assert(ctrl != nullptr);
  assert(ctrl->ParamDim() == 1);
}

// CP4: full round-trip — load, Reset, discover IParametricControl, check nominal params,
// and run one process() block to completion.
// Replaces the Phase 1 "create() throws not-yet-implemented" stub.
void test_cp4_real_load_and_process()
{
  const int inner_count = compute_a2_inner_weight_count();
  const int total_count = inner_count + kAdapterCount;
  const std::vector<float> weights(total_count, 0.0f);
  const nlohmann::json j = build_parametric_fixture(weights);

  // Load: must not throw.
  auto dsp = nam::get_dsp(j);
  assert(dsp != nullptr);

  // Reset: sets up buffers and prewarming (prewarm calls process() internally).
  constexpr int kBlockSize = 64;
  dsp->Reset(48000.0, kBlockSize);

  // IParametricControl is discoverable from the DSP* handle.
  auto* ctrl = dynamic_cast<nam::IParametricControl*>(dsp.get());
  assert(ctrl != nullptr);
  assert(ctrl->ParamDim() == 1);

  // Nominal params are seeded from nominal_params = [0.5].
  const auto params = ctrl->GetParams();
  assert(static_cast<int>(params.size()) == 1);
  assert(std::abs(params[0] - 0.5f) < 1e-6f);

  // One process() block must complete without throwing.
  NAM_SAMPLE in_buf[kBlockSize] = {};
  NAM_SAMPLE out_buf[kBlockSize] = {};
  NAM_SAMPLE* inputs[1] = {in_buf};
  NAM_SAMPLE* outputs[1] = {out_buf};
  dsp->process(inputs, outputs, kBlockSize);
  // No sample-value assertions here (C2.2d owns parity / behavior-change evidence).
}

// --- CP2: config-shape negative cases (C1.3) — unchanged ---

// Wraps get_dsp() and asserts a runtime_error whose message contains `needle`.
void assert_parse_error(nlohmann::json j, const std::string& needle)
{
  bool caught = false;
  try
  {
    nam::get_dsp(j);
    assert(false); // must not reach here
  }
  catch (const std::runtime_error& e)
  {
    const std::string msg = e.what();
    assert(msg.find(needle) != std::string::npos);
    caught = true;
  }
  assert(caught);
}

void test_cp2a_missing_param_names()
{
  nlohmann::json j = build_parametric_fixture({0.0f});
  j["config"].erase("param_names");
  assert_parse_error(j, "param_names");
}

void test_cp2b_missing_param_dim()
{
  nlohmann::json j = build_parametric_fixture({0.0f});
  j["config"].erase("param_dim");
  assert_parse_error(j, "param_dim");
}

void test_cp2c_missing_nominal_params()
{
  nlohmann::json j = build_parametric_fixture({0.0f});
  j["config"].erase("nominal_params");
  assert_parse_error(j, "nominal_params");
}

void test_cp2d_param_dim_names_mismatch()
{
  // param_dim=2 but param_names has 1 entry → mismatch; error must mention both values.
  nlohmann::json j = build_parametric_fixture({0.0f});
  j["config"]["param_dim"] = 2;
  assert_parse_error(j, "param_dim");
  // Also verify both values appear in the message.
  bool caught = false;
  try
  {
    nam::get_dsp(j);
  }
  catch (const std::runtime_error& e)
  {
    const std::string msg = e.what();
    assert(msg.find("2") != std::string::npos);
    assert(msg.find("1") != std::string::npos);
    caught = true;
  }
  assert(caught);
}

void test_cp2e_param_dim_nominal_mismatch()
{
  // param_dim=1, nominal_params has 2 entries → mismatch; error must mention both values.
  nlohmann::json j = build_parametric_fixture({0.0f});
  j["config"]["nominal_params"] = nlohmann::json::array({0.5f, 0.5f});
  assert_parse_error(j, "param_dim");
  bool caught = false;
  try
  {
    nam::get_dsp(j);
  }
  catch (const std::runtime_error& e)
  {
    const std::string msg = e.what();
    assert(msg.find("1") != std::string::npos);
    assert(msg.find("2") != std::string::npos);
    caught = true;
  }
  assert(caught);
}

} // namespace test_parametric_loader

void run_parametric_loader_tests()
{
  // CP1: registration
  test_parametric_loader::test_cp1_parser_is_registered();
  // CP3: weight-count validation (C2.2c)
  test_parametric_loader::test_cp3_adapter_tail_too_short();
  test_parametric_loader::test_cp3_correct_size_constructs();
  // CP4: real round-trip (C2.2c)
  test_parametric_loader::test_cp4_real_load_and_process();
  // CP2: config-shape negatives (C1.3)
  test_parametric_loader::test_cp2a_missing_param_names();
  test_parametric_loader::test_cp2b_missing_param_dim();
  test_parametric_loader::test_cp2c_missing_nominal_params();
  test_parametric_loader::test_cp2d_param_dim_names_mismatch();
  test_parametric_loader::test_cp2e_param_dim_nominal_mismatch();
}
