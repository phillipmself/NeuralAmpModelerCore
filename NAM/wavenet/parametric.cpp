#include "parametric.h"

#include <stdexcept>

namespace nam
{
namespace wavenet
{

std::unique_ptr<DSP> ParametricWaveNetConfig::create(std::vector<float> /*weights*/, double /*sampleRate*/)
{
  throw std::runtime_error("ParametricWaveNet inference not yet implemented");
}

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
