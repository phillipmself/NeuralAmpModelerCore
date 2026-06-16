#pragma once

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

/// \brief Configuration for a ParametricWaveNet model (parser-only until C2.2c)
///
/// Stores the parsed inner WaveNetConfig and the three parametric metadata keys.
struct ParametricWaveNetConfig : public ModelConfig
{
  WaveNetConfig inner_config;
  std::vector<std::string> param_names;
  int param_dim = 0;
  std::vector<float> nominal_params;

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
class ParametricWaveNet : public WaveNet, public IParametricControl
{
public:
  /// \param in_channels         WaveNet in_channels
  /// \param layer_array_params  Used by WaveNet base and to wire adapters (const ref; copied by base)
  /// \param head_scale          WaveNet head_scale
  /// \param with_head           WaveNet with_head
  /// \param head_params         Moved into WaveNet base (move-only)
  /// \param condition_dsp       Moved into WaveNet base (move-only)
  /// \param param_dim           Dimensionality of the parameter vector
  /// \param nominal_params      Initial param vector; size must equal param_dim
  /// \param inner_weights       Weight slice for the WaveNet base (exact count validated by base ctor)
  /// \param adapter_tail        Adapter weights in sorted distinct-C order
  /// \param sample_rate         Expected sample rate
  ParametricWaveNet(int in_channels,
                    const std::vector<LayerArrayParams>& layer_array_params,
                    float head_scale,
                    bool with_head,
                    std::optional<HeadParams> head_params,
                    std::unique_ptr<DSP> condition_dsp,
                    int param_dim,
                    std::vector<float> nominal_params,
                    std::vector<float> inner_weights,
                    std::vector<float> adapter_tail,
                    double sample_rate);

  // IParametricControl
  /// Replace the current parameter vector. Allocation-free; throws std::invalid_argument on dim mismatch.
  void SetParams(std::span<const float> params) override;
  std::span<const float> GetParams() const override;
  int ParamDim() const override;

  /// Recomputes adapters if params are dirty, then delegates to WaveNet::process().
  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override;

private:
  std::vector<ParametricAdapter> _adapters;         // one per distinct C, sorted-C order
  std::map<int, size_t> _channel_to_adapter_index;  // C -> index into _adapters
  std::vector<float> _params;                        // current param vector (pre-sized)
  int _param_dim;
  bool _params_dirty = true;
};

/// \brief Config parser for "ParametricWaveNet" — auto-registered at startup via ConfigParserHelper
std::unique_ptr<ModelConfig> create_parametric_config(const nlohmann::json& config, double sampleRate);

} // namespace wavenet
} // namespace nam
