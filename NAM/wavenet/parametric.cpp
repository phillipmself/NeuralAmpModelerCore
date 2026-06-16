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
  pwc->param_names = config.at("param_names").get<std::vector<std::string>>();
  pwc->param_dim = config.at("param_dim").get<int>();
  pwc->nominal_params = config.at("nominal_params").get<std::vector<float>>();
  return pwc;
}

} // namespace wavenet
} // namespace nam

namespace
{
static nam::ConfigParserHelper _register_ParametricWaveNet("ParametricWaveNet", nam::wavenet::create_parametric_config);
}
