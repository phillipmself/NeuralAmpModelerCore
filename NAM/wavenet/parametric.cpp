#include "parametric.h"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace nam
{
namespace wavenet
{

// =============================================================================
// ParametricWaveNet
// =============================================================================

namespace
{
// The nominal (default) param vector is derived from the specs' defaults, positionally.
// This is exactly what the old flat `nominal_params` carried, so the forward pass is unchanged.
std::vector<float> nominal_from_specs(const std::vector<ParamSpec>& specs)
{
  std::vector<float> nominal;
  nominal.reserve(specs.size());
  for (const auto& s : specs)
    nominal.push_back(s.defaultValue);
  return nominal;
}
} // namespace

ParametricWaveNet::ParametricWaveNet(int in_channels,
                                     const std::vector<LayerArrayParams>& layer_array_params,
                                     float head_scale,
                                     bool with_head,
                                     std::optional<HeadParams> head_params,
                                     std::unique_ptr<DSP> condition_dsp,
                                     std::vector<ParamSpec> param_specs,
                                     std::vector<float> inner_weights,
                                     std::vector<float> adapter_tail,
                                     double sample_rate)
: WaveNet(in_channels, layer_array_params, head_scale, with_head,
          std::move(head_params), std::move(inner_weights),
          std::move(condition_dsp), sample_rate)
, _param_specs(std::move(param_specs))
, _params(nominal_from_specs(_param_specs))
, _param_dim(static_cast<int>(_param_specs.size()))
, _params_dirty(true)
{
  if (_param_specs.empty())
    throw std::invalid_argument("ParametricWaveNet: param_specs must contain at least one parameter");

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

const std::vector<ParamSpec>& ParametricWaveNet::GetParamSpecs() const
{
  return _param_specs;
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
  // param_dim is derived from the ordered spec list (positional parameter vector).
  const int param_dim = static_cast<int>(params.size());

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
    std::move(params), // ordered specs; param_dim + nominal vector derived inside the ctor
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

  // The self-describing `params` array is the only parametric metadata in the .nam now.
  // Each entry is {name, min, max, default}; order is significant (positional net input).
  if (!config.contains("params"))
    throw std::runtime_error(
      "ParametricWaveNet config missing 'params' array (name/min/max/default per parameter).");

  const auto& params_json = config.at("params");
  if (!params_json.is_array())
    throw std::runtime_error("ParametricWaveNet config: 'params' must be an array of objects.");

  pwc->params.reserve(params_json.size());
  for (size_t i = 0; i < params_json.size(); ++i)
  {
    const auto& entry = params_json[i];
    const std::string where = "ParametricWaveNet config: params[" + std::to_string(i) + "]";
    if (!entry.is_object() || !entry.contains("name") || !entry.contains("min")
        || !entry.contains("max") || !entry.contains("default"))
      throw std::runtime_error(where + " must be an object with name/min/max/default.");

    ParamSpec spec;
    spec.name = entry.at("name").get<std::string>();
    spec.min = entry.at("min").get<float>();
    spec.max = entry.at("max").get<float>();
    spec.defaultValue = entry.at("default").get<float>();

    const std::string named = "ParametricWaveNet config: param '" + spec.name + "'";
    if (!std::isfinite(spec.min) || !std::isfinite(spec.max) || !std::isfinite(spec.defaultValue))
      throw std::runtime_error(named + " has non-finite min/max/default.");
    if (!(spec.min <= spec.defaultValue && spec.defaultValue <= spec.max))
      throw std::runtime_error(
        named + " requires min <= default <= max, but got min=" + std::to_string(spec.min)
        + ", default=" + std::to_string(spec.defaultValue) + ", max=" + std::to_string(spec.max));

    pwc->params.push_back(std::move(spec));
  }

  if (pwc->params.empty())
    throw std::runtime_error("ParametricWaveNet config: 'params' array must contain at least one parameter.");

  return pwc;
}

} // namespace wavenet
} // namespace nam

namespace
{
static nam::ConfigParserHelper _register_ParametricWaveNet("ParametricWaveNet", nam::wavenet::create_parametric_config);
}
