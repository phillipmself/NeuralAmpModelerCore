#include "parametric.h"

#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace nam
{
namespace wavenet
{

// =============================================================================
// ParametricWaveNet
// =============================================================================

ParametricWaveNet::ParametricWaveNet(int in_channels,
                                     const std::vector<LayerArrayParams>& layer_array_params,
                                     float head_scale,
                                     bool with_head,
                                     std::optional<HeadParams> head_params,
                                     std::unique_ptr<DSP> condition_dsp,
                                     int param_dim,
                                     std::vector<float> nominal_params,
                                     std::vector<float> inner_weights,
                                     std::vector<float> adapter_tail,
                                     double sample_rate)
: WaveNet(in_channels, layer_array_params, head_scale, with_head,
          std::move(head_params), std::move(inner_weights),
          std::move(condition_dsp), sample_rate)
, _params(std::move(nominal_params))
, _param_dim(param_dim)
, _params_dirty(true)
{
  if (_param_dim < 0)
    throw std::invalid_argument("ParametricWaveNet: param_dim must be non-negative");
  if (static_cast<int>(_params.size()) != _param_dim)
    throw std::invalid_argument(
      "ParametricWaveNet: nominal_params size " + std::to_string(_params.size())
      + " does not match param_dim " + std::to_string(_param_dim));

  // Build adapters from adapter_tail in sorted distinct-C order and map C → index.
  auto distinct_Cs = parametric_distinct_channel_sizes(layer_array_params);
  _adapters.reserve(distinct_Cs.size());
  {
    auto it = adapter_tail.begin();
    for (size_t i = 0; i < distinct_Cs.size(); ++i)
    {
      const int C = distinct_Cs[i];
      _adapters.emplace_back(C, _param_dim);
      _adapters.back().load_weights_(it);
      _channel_to_adapter_index[C] = i;
    }
    // adapter_tail is now fully consumed; the WaveNet base validated the inner slice.
  }

  // Wire every layer to its corresponding adapter (non-owning pointer; lifetime tied to _adapters).
  for (auto& arr : GetLayerArrays())
    for (auto& layer : arr.GetLayers())
    {
      const int C = static_cast<int>(layer.get_conv().get_out_channels());
      layer.SetParametricAdapter(&_adapters[_channel_to_adapter_index.at(C)]);
    }
}

void ParametricWaveNet::SetParams(std::span<const float> params)
{
#ifndef NDEBUG
  _debug_enter_param_api_();
  try
  {
#endif
    if (static_cast<int>(params.size()) != _param_dim)
      throw std::invalid_argument(
        "ParametricWaveNet::SetParams: expected " + std::to_string(_param_dim)
        + " params, got " + std::to_string(static_cast<int>(params.size())));
    std::copy(params.begin(), params.end(), _params.begin());
    _params_dirty = true;
#ifndef NDEBUG
  }
  catch (...)
  {
    _debug_leave_param_api_();
    throw;
  }
  _debug_leave_param_api_();
#endif
}

std::span<const float> ParametricWaveNet::GetParams() const
{
#ifndef NDEBUG
  const_cast<ParametricWaveNet*>(this)->_debug_enter_param_api_();
#endif
  const auto params = std::span<const float>(_params);
#ifndef NDEBUG
  const_cast<ParametricWaveNet*>(this)->_debug_leave_param_api_();
#endif
  return params;
}

int ParametricWaveNet::ParamDim() const
{
  return _param_dim;
}

void ParametricWaveNet::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
#ifndef NDEBUG
  _debug_enter_param_api_();
  try
  {
#endif
    if (_params_dirty)
    {
      for (auto& adapter : _adapters)
        adapter.Recompute(_params);
      _params_dirty = false;
    }
    WaveNet::process(input, output, num_frames);
#ifndef NDEBUG
  }
  catch (...)
  {
    _debug_leave_param_api_();
    throw;
  }
  _debug_leave_param_api_();
#endif
}

#ifndef NDEBUG
void ParametricWaveNet::_debug_enter_param_api_()
{
  assert(!_debug_param_api_active.test_and_set(std::memory_order_acquire));
}

void ParametricWaveNet::_debug_leave_param_api_()
{
  _debug_param_api_active.clear(std::memory_order_release);
}
#endif

// =============================================================================
// ParametricWaveNetConfig::create()  — now constructs a real ParametricWaveNet
// =============================================================================

std::unique_ptr<DSP> ParametricWaveNetConfig::create(std::vector<float> weights, double sampleRate)
{
  // Compute adapter weight count from inner config layer arrays (before moving inner_config).
  const auto distinct_Cs = parametric_distinct_channel_sizes(inner_config.layer_array_params);
  const int adapter_count = parametric_adapter_weight_count(distinct_Cs, param_dim);

  // CP3: weight blob must be large enough to contain the adapter tail.
  if (static_cast<int>(weights.size()) < adapter_count)
    throw std::runtime_error(
      "ParametricWaveNet: weight blob too short: need " + std::to_string(adapter_count)
      + " adapter-tail weights but only " + std::to_string(weights.size()) + " total weights provided");

  // Split [inner_weights | adapter_tail]; inner_weights are the leading slice.
  const size_t inner_end = weights.size() - static_cast<size_t>(adapter_count);
  std::vector<float> inner_weights(weights.begin(), weights.begin() + inner_end);
  std::vector<float> adapter_tail(weights.begin() + inner_end, weights.end());

  // Construct ParametricWaveNet. The WaveNet base ctor is the authoritative validator
  // for the inner slice (throws on leftover or underfull inner weights).
  return std::make_unique<ParametricWaveNet>(
    inner_config.in_channels,
    inner_config.layer_array_params, // passed by const ref; WaveNet copies it
    inner_config.head_scale,
    inner_config.with_head,
    std::move(inner_config.head_params),
    std::move(inner_config.condition_dsp),
    param_dim,
    nominal_params, // copied into ParametricWaveNet as the initial _params seed
    std::move(inner_weights),
    std::move(adapter_tail),
    sampleRate);
}

// =============================================================================
// Config parser
// =============================================================================

std::unique_ptr<nam::ModelConfig> create_parametric_config(const nlohmann::json& config, double sampleRate)
{
  auto pwc = std::make_unique<ParametricWaveNetConfig>();
  pwc->inner_config = parse_config_json(config, sampleRate);

  // Validate required parametric fields before accessing them.
  if (!config.contains("param_names"))
    throw std::runtime_error("ParametricWaveNet config missing required field 'param_names'");
  if (!config.contains("param_dim"))
    throw std::runtime_error("ParametricWaveNet config missing required field 'param_dim'");
  if (!config.contains("nominal_params"))
    throw std::runtime_error("ParametricWaveNet config missing required field 'nominal_params'");

  pwc->param_names = config.at("param_names").get<std::vector<std::string>>();
  pwc->param_dim = config.at("param_dim").get<int>();
  pwc->nominal_params = config.at("nominal_params").get<std::vector<float>>();

  // Validate dimension consistency.
  const int names_size = static_cast<int>(pwc->param_names.size());
  if (pwc->param_dim != names_size)
    throw std::runtime_error(
      "ParametricWaveNet config: param_dim=" + std::to_string(pwc->param_dim)
      + " does not match len(param_names)=" + std::to_string(names_size));

  const int nominal_size = static_cast<int>(pwc->nominal_params.size());
  if (pwc->param_dim != nominal_size)
    throw std::runtime_error(
      "ParametricWaveNet config: param_dim=" + std::to_string(pwc->param_dim)
      + " does not match len(nominal_params)=" + std::to_string(nominal_size));

  return pwc;
}

} // namespace wavenet
} // namespace nam

namespace
{
static nam::ConfigParserHelper _register_ParametricWaveNet("ParametricWaveNet", nam::wavenet::create_parametric_config);
}
