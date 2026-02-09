#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "parametric.h"
#include "util.h"

std::vector<nam::ParameterDescriptor> nam::parametric::ParseParameterDescriptors(
  const nlohmann::json& parametric_config)
{
  if (!parametric_config.is_object())
  {
    throw std::runtime_error("Expected `config.parametric` to be a JSON object.");
  }

  std::vector<std::string> names;
  names.reserve(parametric_config.size());
  for (const auto& item : parametric_config.items())
  {
    names.push_back(item.key());
  }
  std::sort(names.begin(), names.end());

  std::vector<ParameterDescriptor> descriptors;
  descriptors.reserve(names.size());
  for (const std::string& name : names)
  {
    const nlohmann::json& definition = parametric_config.at(name);
    if (!definition.is_object())
    {
      throw std::runtime_error("Parameter definition for `" + name + "` must be a JSON object.");
    }

    if (definition.find("type") == definition.end() || !definition["type"].is_string())
    {
      throw std::runtime_error("Parameter `" + name + "` is missing string field `type`.");
    }
    if (definition.find("default_value") == definition.end())
    {
      throw std::runtime_error("Parameter `" + name + "` is missing field `default_value`.");
    }

    const std::string type = util::lowercase(definition["type"].get<std::string>());
    ParameterDescriptor descriptor;
    descriptor.name = name;

    if (type == "boolean")
    {
      descriptor.type = ParameterType::Boolean;
      if (definition["default_value"].is_boolean())
      {
        descriptor.default_value = definition["default_value"].get<bool>() ? 1.0 : 0.0;
      }
      else if (definition["default_value"].is_number())
      {
        descriptor.default_value = definition["default_value"].get<double>() == 0.0 ? 0.0 : 1.0;
      }
      else
      {
        throw std::runtime_error("Boolean parameter `" + name + "` must have a numeric or boolean `default_value`.");
      }
    }
    else if (type == "continuous")
    {
      descriptor.type = ParameterType::Continuous;
      if (!definition["default_value"].is_number())
      {
        throw std::runtime_error("Continuous parameter `" + name + "` must have numeric `default_value`.");
      }
      descriptor.default_value = definition["default_value"].get<double>();

      const auto min_it = definition.find("minval");
      if (min_it != definition.end() && !min_it->is_null())
      {
        if (!min_it->is_number())
        {
          throw std::runtime_error("Continuous parameter `" + name + "` has non-numeric `minval`.");
        }
        descriptor.min_value = min_it->get<double>();
      }

      const auto max_it = definition.find("maxval");
      if (max_it != definition.end() && !max_it->is_null())
      {
        if (!max_it->is_number())
        {
          throw std::runtime_error("Continuous parameter `" + name + "` has non-numeric `maxval`.");
        }
        descriptor.max_value = max_it->get<double>();
      }

      if (descriptor.min_value.has_value() && descriptor.max_value.has_value()
          && *descriptor.min_value > *descriptor.max_value)
      {
        throw std::runtime_error("Continuous parameter `" + name + "` has `minval` > `maxval`.");
      }
      if (descriptor.min_value.has_value() && descriptor.default_value < *descriptor.min_value)
      {
        descriptor.default_value = *descriptor.min_value;
      }
      if (descriptor.max_value.has_value() && descriptor.default_value > *descriptor.max_value)
      {
        descriptor.default_value = *descriptor.max_value;
      }
    }
    else
    {
      throw std::runtime_error("Unrecognized parameter type `" + definition["type"].get<std::string>() + "` for `"
                               + name + "`.");
    }

    descriptors.push_back(descriptor);
  }

  return descriptors;
}
