// CP6 tests for the IParametricControl runtime parameter-control seam (C2.1).
//
// The seam: a stateful full-vector setter consumed block-by-block.
// Host contract: call SetParams() between process() calls; process() consumes
// the most recently committed vector for the full block. Allocation-free after
// construction.
//
// Uses a local TestParametricDSP to prove seam semantics without requiring
// a real runnable ParametricWaveNet inference path (that belongs to C2.2).
// ParametricWaveNetConfig::create() still throws not-implemented.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "NAM/dsp.h"
#include "allocation_tracking.h"

namespace test_parametric_dsp
{
namespace
{

// Minimal concrete DSP that implements IParametricControl for seam testing.
// process() output: output[f] = input[f] + current_params[0], making it
// observably dependent on the current parameter vector without real WaveNet
// inference.
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
  test_parametric_dsp::test_cp6a_nominal_params_seed_initial_state();
  test_parametric_dsp::test_cp6b_last_committed_params();
  test_parametric_dsp::test_cp6c_param_change_affects_output();
  test_parametric_dsp::test_cp6d_no_allocation_during_steady_state();
  test_parametric_dsp::test_cp6e_dim_mismatch_throws();
  test_parametric_dsp::test_cp6f_dynamic_cast_discovery();
}
