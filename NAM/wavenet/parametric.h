#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../model_config.h"
#include "model.h"

namespace nam
{
namespace wavenet
{

/// \brief Configuration for a ParametricWaveNet model (parser-only, Phase 1)
///
/// Stores the parsed inner WaveNetConfig and the three parametric metadata keys.
/// create() throws not-implemented until Phase 2 DSP lands.
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

/// \brief Config parser for "ParametricWaveNet" — auto-registered at startup via ConfigParserHelper
std::unique_ptr<ModelConfig> create_parametric_config(const nlohmann::json& config, double sampleRate);

} // namespace wavenet
} // namespace nam
