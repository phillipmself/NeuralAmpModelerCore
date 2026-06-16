#pragma once

#include <vector>

#include "json.hpp"

namespace test_parametric_fixtures
{

// A2 nano shape constants — kept in one place so CP3/CP4 and CP5 stay aligned.
inline constexpr int kA2NumLayers = 23;
inline constexpr int kA2HeadKernelSize = 16;
inline constexpr float kA2LeakySlope = 0.01f;
inline constexpr int kA2KernelSizes[kA2NumLayers] = {
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6};
inline constexpr int kA2Dilations[kA2NumLayers] = {
  1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239};

// Adapter count for C=3, P=1: 2*C*P + 2*C = 2*3*1 + 2*3 = 12.
inline constexpr int kA2AdapterCount = 12;

inline nlohmann::json build_a2_inner_config()
{
  using nlohmann::json;

  const int channels = 3;

  json activation = json::array();
  json gating_mode = json::array();
  json secondary = json::array();
  json kernel_sizes_arr = json::array();
  json dilations_arr = json::array();
  for (int i = 0; i < kA2NumLayers; i++)
  {
    activation.push_back({{"type", "LeakyReLU"}, {"negative_slope", kA2LeakySlope}});
    gating_mode.push_back("none");
    secondary.push_back(nullptr);
    kernel_sizes_arr.push_back(kA2KernelSizes[i]);
    dilations_arr.push_back(kA2Dilations[i]);
  }

  json film_inactive = {{"active", false}, {"shift", true}, {"groups", 1}};

  json layer;
  layer["input_size"] = 1;
  layer["condition_size"] = 1;
  layer["channels"] = channels;
  layer["bottleneck"] = channels;
  layer["kernel_sizes"] = kernel_sizes_arr;
  layer["dilations"] = dilations_arr;
  layer["activation"] = activation;
  layer["gating_mode"] = gating_mode;
  layer["secondary_activation"] = secondary;
  layer["head"] = {{"out_channels", 1}, {"kernel_size", kA2HeadKernelSize}, {"bias", true}};
  layer["head1x1"] = {{"active", false}, {"out_channels", 1}, {"groups", 1}};
  layer["layer1x1"] = {{"active", true}, {"groups", 1}};
  layer["conv_pre_film"] = film_inactive;
  layer["conv_post_film"] = film_inactive;
  layer["input_mixin_pre_film"] = film_inactive;
  layer["input_mixin_post_film"] = film_inactive;
  layer["activation_pre_film"] = film_inactive;
  layer["activation_post_film"] = film_inactive;
  layer["layer1x1_post_film"] = film_inactive;
  layer["head1x1_post_film"] = film_inactive;
  layer["groups_input"] = 1;
  layer["groups_input_mixin"] = 1;

  json config;
  config["layers"] = json::array({layer});
  config["head_scale"] = 0.01f;
  return config;
}

inline int compute_a2_inner_weight_count()
{
  const int channels = 3;
  const int bn = channels; // bottleneck == channels, gating=none
  int total = channels;    // rechannel: Conv1x1(1, channels, false)
  for (int i = 0; i < kA2NumLayers; i++)
  {
    const int K = kA2KernelSizes[i];
    total += bn * channels * K + bn;   // conv1d weights + bias
    total += bn;                       // input mixin (no bias)
    total += channels * bn + channels; // layer1x1 + bias
  }
  total += channels * kA2HeadKernelSize + 1; // head rechannel Conv1D + bias
  total += 1;                                // head_scale
  return total;
}

inline nlohmann::json build_a2_parametric_fixture(const std::vector<float>& weights)
{
  using nlohmann::json;

  json config = build_a2_inner_config();
  config["param_names"] = json::array({"gain"});
  config["param_dim"] = 1;
  config["nominal_params"] = json::array({0.5});

  json j;
  j["version"] = "0.7.0";
  j["architecture"] = "ParametricWaveNet";
  j["config"] = config;
  j["weights"] = weights;
  j["sample_rate"] = 48000;
  return j;
}

} // namespace test_parametric_fixtures
