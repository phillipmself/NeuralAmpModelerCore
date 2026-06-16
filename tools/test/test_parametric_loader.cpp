// CP1 (flipped, C1.2): "ParametricWaveNet" is now a registered architecture.
// The parser composes wavenet::parse_config_json for the inner config and stores
// parametric metadata. create() throws "ParametricWaveNet inference not yet
// implemented" (AD-C4). See docs/parametric-a2-core/test-plan.md CP1/CP4.

#include <cassert>
#include <string>

#include "json.hpp"

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
// Matches test_a2_fast::build_a2_config(3) so C1.2 can reuse this fixture unchanged.
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

// Minimal valid ParametricWaveNet fixture.
// Includes all four keys get_dsp validates before the registry lookup (AD-C11):
// version, architecture, config, weights. The weights array need not be
// correctly sized in Phase 1 because create() throws before consuming them.
nlohmann::json build_parametric_fixture()
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
  j["weights"] = json::array({0.0f});
  j["sample_rate"] = 48000;
  return j;
}

} // namespace

// CP1 (C1.2): "ParametricWaveNet" is registered and get_dsp() reaches create(),
// which throws the deliberate not-implemented error.
void test_cp1_registered_parser_throws_not_implemented()
{
  assert(nam::ConfigParserRegistry::instance().has("ParametricWaveNet") == true);

  nlohmann::json j = build_parametric_fixture();
  bool caught = false;
  try
  {
    nam::get_dsp(j);
    assert(false); // must not reach here — create() must throw
  }
  catch (const std::runtime_error& e)
  {
    const std::string msg = e.what();
    assert(msg.find("ParametricWaveNet inference not yet implemented") != std::string::npos);
    caught = true;
  }
  assert(caught);
}

// CP4: a valid parametric fixture parses successfully, then create() throws.
// In Phase 1 this is observationally identical to CP1 (AD-C11); kept as a
// named test so Phase 2 can extend it into a real weight-consuming round-trip.
void test_cp4_parse_then_create_throws()
{
  nlohmann::json j = build_parametric_fixture();
  bool caught = false;
  try
  {
    nam::get_dsp(j);
    assert(false);
  }
  catch (const std::runtime_error& e)
  {
    const std::string msg = e.what();
    assert(msg.find("ParametricWaveNet inference not yet implemented") != std::string::npos);
    caught = true;
  }
  assert(caught);
}

// --- CP2: config-shape negative cases (C1.3) ---
//
// Each helper strips one field (or introduces a mismatch) from the valid fixture
// and asserts that create_parametric_config throws a std::runtime_error whose
// message names the offending field / both values. Weight-count validation is
// intentionally NOT tested here (deferred to Phase 2, AD-C10).

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
  nlohmann::json j = build_parametric_fixture();
  j["config"].erase("param_names");
  assert_parse_error(j, "param_names");
}

void test_cp2b_missing_param_dim()
{
  nlohmann::json j = build_parametric_fixture();
  j["config"].erase("param_dim");
  assert_parse_error(j, "param_dim");
}

void test_cp2c_missing_nominal_params()
{
  nlohmann::json j = build_parametric_fixture();
  j["config"].erase("nominal_params");
  assert_parse_error(j, "nominal_params");
}

void test_cp2d_param_dim_names_mismatch()
{
  // param_dim=2 but param_names has 1 entry → mismatch; error must mention both values.
  nlohmann::json j = build_parametric_fixture();
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
  nlohmann::json j = build_parametric_fixture();
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
  test_parametric_loader::test_cp1_registered_parser_throws_not_implemented();
  test_parametric_loader::test_cp4_parse_then_create_throws();
  // CP2: config-shape negative cases (C1.3)
  test_parametric_loader::test_cp2a_missing_param_names();
  test_parametric_loader::test_cp2b_missing_param_dim();
  test_parametric_loader::test_cp2c_missing_nominal_params();
  test_parametric_loader::test_cp2d_param_dim_names_mismatch();
  test_parametric_loader::test_cp2e_param_dim_nominal_mismatch();
}
