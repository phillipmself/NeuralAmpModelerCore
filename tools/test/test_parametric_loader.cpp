// CP1: pre-registration fail-loud test for "ParametricWaveNet".
// Before any parser is registered, a syntactically valid ParametricWaveNet
// JSON must throw the generic registry error (AD-C2), proving the fixture is
// well-formed but the architecture is intentionally unsupported.
// See docs/parametric-a2-core/test-plan.md CP1.

#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "json.hpp"

#include "NAM/get_dsp.h"

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
// correctly sized in C1.1 because the registry error fires first.
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

void test_cp1_unregistered_architecture_throws_registry_error()
{
  nlohmann::json j = build_parametric_fixture();
  bool caught = false;
  try
  {
    nam::get_dsp(j);
    assert(false); // must not reach here before the parser is registered
  }
  catch (const std::runtime_error& e)
  {
    const std::string msg = e.what();
    // TODO C1.2: flip this assertion to expect "ParametricWaveNet inference not yet implemented"
    assert(msg.find("No config parser registered for architecture: ParametricWaveNet") != std::string::npos);
    caught = true;
  }
  assert(caught);
}

} // namespace test_parametric_loader

void run_parametric_loader_tests()
{
  test_parametric_loader::test_cp1_unregistered_architecture_throws_registry_error();
}
