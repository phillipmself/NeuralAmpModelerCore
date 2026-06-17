// C2.2c + C2.3: ParametricWaveNet loader tests.
//
// CP1: parser registration guard (has() == true).
// RG1: unknown architecture is rejected with registry-error message (C2.3).
// CP2: config-shape negatives for the self-describing `params` array (missing array,
//      malformed entry, default out of [min,max], non-finite, empty) + multi-param order.
// CP3: adapter-tail weight-count validation (negative: too-short blob; positive: exact size).
// CP4: real round-trip — get_dsp(), Reset(), process(), IParametricControl discovery,
//      and per-parameter spec (name/min/max/default) readback.
//
// See docs/parametric-a2-core/test-plan.md for full CP descriptions.

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "NAM/model_config.h"
#include "parametric_test_fixtures.h"

namespace test_parametric_loader
{
namespace
{

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
  // Provide 5 weights total (far less than kA2AdapterCount=12 needed for adapter tail).
  const std::vector<float> tiny(5, 0.0f);
  const nlohmann::json j = test_parametric_fixtures::build_a2_parametric_fixture(tiny);

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
  const int inner_count = test_parametric_fixtures::compute_a2_inner_weight_count();
  const int total_count = inner_count + test_parametric_fixtures::kA2AdapterCount;
  const std::vector<float> weights(total_count, 0.0f);
  const nlohmann::json j = test_parametric_fixtures::build_a2_parametric_fixture(weights);

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
  const int inner_count = test_parametric_fixtures::compute_a2_inner_weight_count();
  const int total_count = inner_count + test_parametric_fixtures::kA2AdapterCount;
  const std::vector<float> weights(total_count, 0.0f);
  const nlohmann::json j = test_parametric_fixtures::build_a2_parametric_fixture(weights);

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

  // Nominal params are seeded from the spec defaults = [0.5].
  const auto params = ctrl->GetParams();
  assert(static_cast<int>(params.size()) == 1);
  assert(std::abs(params[0] - 0.5f) < 1e-6f);

  // Per-parameter specs (name/min/max/default) are surfaced in positional order.
  const auto& specs = ctrl->GetParamSpecs();
  assert(specs.size() == 1);
  assert(specs[0].name == "gain");
  assert(std::abs(specs[0].min - 0.0f) < 1e-6f);
  assert(std::abs(specs[0].max - 1.0f) < 1e-6f);
  assert(std::abs(specs[0].defaultValue - 0.5f) < 1e-6f);

  // One process() block must complete without throwing.
  NAM_SAMPLE in_buf[kBlockSize] = {};
  NAM_SAMPLE out_buf[kBlockSize] = {};
  NAM_SAMPLE* inputs[1] = {in_buf};
  NAM_SAMPLE* outputs[1] = {out_buf};
  dsp->process(inputs, outputs, kBlockSize);
  // No sample-value assertions here (C2.2d owns parity / behavior-change evidence).
}

// RG1: an explicitly unknown architecture must throw the registry-error message.
// Use an otherwise valid-shaped fixture so this isolates registry lookup instead
// of depending on unrelated validation order.
void test_rg1_unknown_architecture_throws()
{
  const int inner_count = test_parametric_fixtures::compute_a2_inner_weight_count();
  const int total_count = inner_count + test_parametric_fixtures::kA2AdapterCount;
  nlohmann::json j = test_parametric_fixtures::build_a2_parametric_fixture(
    std::vector<float>(total_count, 0.0f));
  j["architecture"] = "DefinitelyUnknownArchitecture";

  bool caught = false;
  try
  {
    nam::get_dsp(j);
    assert(false); // must not reach here
  }
  catch (const std::runtime_error& e)
  {
    const std::string msg = e.what();
    assert(msg.find("No config parser registered for architecture: DefinitelyUnknownArchitecture")
           != std::string::npos);
    caught = true;
  }
  assert(caught);
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

// Missing 'params' on a ParametricWaveNet file → clear, schema-naming error.
void test_cp2a_missing_params()
{
  nlohmann::json j = test_parametric_fixtures::build_a2_parametric_fixture({0.0f});
  j["config"].erase("params");
  assert_parse_error(j, "missing 'params' array");
}

// A param entry that omits a required key (here: 'max') → error names the offending index.
void test_cp2b_param_entry_missing_key()
{
  nlohmann::json j = test_parametric_fixtures::build_a2_parametric_fixture({0.0f});
  j["config"]["params"] = nlohmann::json::array({
    {{"name", "gain"}, {"min", 0.0}, {"default", 0.5}}, // no "max"
  });
  assert_parse_error(j, "name/min/max/default");
}

// default outside [min, max] → error names the parameter and the offending values.
void test_cp2c_default_out_of_range()
{
  nlohmann::json j = test_parametric_fixtures::build_a2_parametric_fixture({0.0f});
  j["config"]["params"] = nlohmann::json::array({
    {{"name", "gain"}, {"min", 0.0}, {"max", 1.0}, {"default", 5.0}}, // default > max
  });
  assert_parse_error(j, "min <= default <= max");
  // The parameter name must appear in the message.
  bool caught = false;
  try
  {
    nam::get_dsp(j);
  }
  catch (const std::runtime_error& e)
  {
    assert(std::string(e.what()).find("gain") != std::string::npos);
    caught = true;
  }
  assert(caught);
}

// Non-finite numeric value → rejected.
void test_cp2d_non_finite_value()
{
  nlohmann::json j = test_parametric_fixtures::build_a2_parametric_fixture({0.0f});
  j["config"]["params"] = nlohmann::json::array({
    {{"name", "gain"}, {"min", 0.0}, {"max", INFINITY}, {"default", 0.5}},
  });
  assert_parse_error(j, "non-finite");
}

// Empty 'params' array → rejected.
void test_cp2e_empty_params()
{
  nlohmann::json j = test_parametric_fixtures::build_a2_parametric_fixture({0.0f});
  j["config"]["params"] = nlohmann::json::array();
  assert_parse_error(j, "at least one parameter");
}

// Multi-parameter file → specs parsed in order with per-entry metadata preserved.
void test_cp2f_multi_param_order_preserved()
{
  const int inner_count = test_parametric_fixtures::compute_a2_inner_weight_count();
  // Two params over the same single channel size C=3 → adapter tail count uses param_dim=2.
  // distinct C = {3}; adapter weights = 2*C*P + 2*C = 2*3*2 + 2*3 = 18.
  const int adapter_count = 2 * 3 * 2 + 2 * 3;
  nlohmann::json j = test_parametric_fixtures::build_a2_parametric_fixture(
    std::vector<float>(inner_count + adapter_count, 0.0f));
  j["config"]["params"] = nlohmann::json::array({
    {{"name", "gain"}, {"min", 0.0}, {"max", 10.0}, {"default", 5.0}},
    {{"name", "bright"}, {"min", 0.0}, {"max", 1.0}, {"default", 0.5}},
  });

  auto dsp = nam::get_dsp(j);
  auto* ctrl = dynamic_cast<nam::IParametricControl*>(dsp.get());
  assert(ctrl != nullptr);
  assert(ctrl->ParamDim() == 2);

  const auto& specs = ctrl->GetParamSpecs();
  assert(specs.size() == 2);
  assert(specs[0].name == "gain");
  assert(std::abs(specs[0].max - 10.0f) < 1e-6f);
  assert(std::abs(specs[0].defaultValue - 5.0f) < 1e-6f);
  assert(specs[1].name == "bright");
  assert(std::abs(specs[1].max - 1.0f) < 1e-6f);
  assert(std::abs(specs[1].defaultValue - 0.5f) < 1e-6f);

  // Nominal params seeded from defaults, in order.
  const auto params = ctrl->GetParams();
  assert(params.size() == 2);
  assert(std::abs(params[0] - 5.0f) < 1e-6f);
  assert(std::abs(params[1] - 0.5f) < 1e-6f);
}

} // namespace test_parametric_loader

void run_parametric_loader_tests()
{
  // CP1: registration
  test_parametric_loader::test_cp1_parser_is_registered();
  // RG1: unknown architecture is rejected with registry-error message (C2.3)
  test_parametric_loader::test_rg1_unknown_architecture_throws();
  // CP3: weight-count validation (C2.2c)
  test_parametric_loader::test_cp3_adapter_tail_too_short();
  test_parametric_loader::test_cp3_correct_size_constructs();
  // CP4: real round-trip (C2.2c)
  test_parametric_loader::test_cp4_real_load_and_process();
  // CP2: config-shape negatives for the self-describing `params` array
  test_parametric_loader::test_cp2a_missing_params();
  test_parametric_loader::test_cp2b_param_entry_missing_key();
  test_parametric_loader::test_cp2c_default_out_of_range();
  test_parametric_loader::test_cp2d_non_finite_value();
  test_parametric_loader::test_cp2e_empty_params();
  test_parametric_loader::test_cp2f_multi_param_order_preserved();
}
