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
  nlohmann::json j = test_parametric_fixtures::build_a2_parametric_fixture({0.0f});
  j["config"].erase("param_names");
  assert_parse_error(j, "param_names");
}

void test_cp2b_missing_param_dim()
{
  nlohmann::json j = test_parametric_fixtures::build_a2_parametric_fixture({0.0f});
  j["config"].erase("param_dim");
  assert_parse_error(j, "param_dim");
}

void test_cp2c_missing_nominal_params()
{
  nlohmann::json j = test_parametric_fixtures::build_a2_parametric_fixture({0.0f});
  j["config"].erase("nominal_params");
  assert_parse_error(j, "nominal_params");
}

void test_cp2d_param_dim_names_mismatch()
{
  // param_dim=2 but param_names has 1 entry → mismatch; error must mention both values.
  nlohmann::json j = test_parametric_fixtures::build_a2_parametric_fixture({0.0f});
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
  nlohmann::json j = test_parametric_fixtures::build_a2_parametric_fixture({0.0f});
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
