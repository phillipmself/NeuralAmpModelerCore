#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "../model_config.h"
#include "model.h"
#include "parametric_adapter.h"

namespace nam
{
namespace wavenet
{

/// \brief Configuration for a ParametricWaveNet model.
///
/// Stores the parsed inner WaveNetConfig and the ordered, self-describing parameter
/// specs (name/min/max/default per parameter). param_dim and the nominal/default vector
/// are derived from `params` at construction; min/max are UI metadata only.
struct ParametricWaveNetConfig : public ModelConfig
{
  WaveNetConfig inner_config;
  std::vector<ParamSpec> params;

  // Move-only: owns a move-only WaveNetConfig
  ParametricWaveNetConfig() = default;
  ParametricWaveNetConfig(ParametricWaveNetConfig&&) = default;
  ParametricWaveNetConfig& operator=(ParametricWaveNetConfig&&) = default;
  ParametricWaveNetConfig(const ParametricWaveNetConfig&) = delete;
  ParametricWaveNetConfig& operator=(const ParametricWaveNetConfig&) = delete;

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override;
};

/// \brief Concrete parametric WaveNet DSP.
///
/// Inherits the ordinary WaveNet inference path (generic Layer/LayerArray, never a2_fast — AD-C14)
/// and adds per-distinct-channel-size affine adapters exposed via IParametricControl.
///
/// Construction:
/// - The weight blob is split [inner_weights | adapter_tail]; inner_weights are forwarded to the
///   WaveNet base which validates them authoritatively.
/// - One ParametricAdapter per distinct channel size C is constructed from adapter_tail.
/// - Every Layer in every LayerArray is wired to its adapter via SetParametricAdapter().
/// - _params is seeded from nominal_params; _params_dirty = true.
///
/// Runtime:
/// - SetParams() updates _params (allocation-free) and sets _params_dirty.
/// - process() calls Recompute() on each adapter once if dirty, then delegates to WaveNet::process().
/// - Reset() does NOT reseed nominal params; the host owns param state after construction.
/// - Access is intentionally unsynchronized; the host must serialize
///   SetParams()/GetParams()/process()/Reset() per IParametricControl.
class ParametricWaveNet : public WaveNet, public IParametricControl
{
public:
  /// \param in_channels         WaveNet in_channels
  /// \param layer_array_params  Used by WaveNet base and to wire adapters (const ref; copied by base)
  /// \param head_scale          WaveNet head_scale
  /// \param with_head           WaveNet with_head
  /// \param head_params         Moved into WaveNet base (move-only)
  /// \param condition_dsp       Moved into WaveNet base (move-only)
  /// \param param_specs         Ordered per-parameter specs; param_dim and the nominal
  ///                            (default) vector are derived from this. Must be non-empty.
  /// \param inner_weights       Weight slice for the WaveNet base (exact count validated by base ctor)
  /// \param adapter_tail        Adapter weights in sorted distinct-C order
  /// \param sample_rate         Expected sample rate
  ParametricWaveNet(int in_channels,
                    const std::vector<LayerArrayParams>& layer_array_params,
                    float head_scale,
                    bool with_head,
                    std::optional<HeadParams> head_params,
                    std::unique_ptr<DSP> condition_dsp,
                    std::vector<ParamSpec> param_specs,
                    std::vector<float> inner_weights,
                    std::vector<float> adapter_tail,
                    double sample_rate);

  // IParametricControl
  /// Replace the current parameter vector.
  /// Allocation-free after construction; throws std::invalid_argument on dim mismatch.
  /// Intentionally unsynchronized: the host must ensure this does not overlap
  /// with process(), Reset(), prewarm(), or destruction.
  void SetParams(std::span<const float> params) override;
  std::span<const float> GetParams() const override;
  int ParamDim() const override;
  const std::vector<ParamSpec>& GetParamSpecs() const override;

  /// Recomputes adapters if params are dirty, then delegates to WaveNet::process().
  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override;

private:
#ifndef NDEBUG
  /// Debug checks to catch concurrent access to the param API for debug builds
  /// Not a thread safety mechanism!
  void _debug_enter_param_api_();
  void _debug_leave_param_api_();
#endif

  std::vector<ParametricAdapter> _adapters;         // one per distinct C, sorted-C order
  std::map<int, size_t> _channel_to_adapter_index;  // C -> index into _adapters
  std::vector<ParamSpec> _param_specs;               // per-parameter metadata, positional order
  std::vector<float> _params;                        // current param vector (pre-sized)
  int _param_dim;
  bool _params_dirty = true;
#ifndef NDEBUG
  std::atomic_flag _debug_param_api_active = ATOMIC_FLAG_INIT;
#endif
};

/// \brief Config parser for "ParametricWaveNet" — auto-registered at startup via ConfigParserHelper
std::unique_ptr<ModelConfig> create_parametric_config(const nlohmann::json& config, double sampleRate);

} // namespace wavenet
} // namespace nam
