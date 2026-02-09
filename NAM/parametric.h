#pragma once

#include <vector>

#include "dsp.h"
#include "json.hpp"

namespace nam
{
namespace parametric
{
/// \brief Parse parameter descriptors from a `.nam` parametric config object
///
/// Expected shape:
/// {
///   "ParamName": {
///     "type": "boolean" | "continuous",
///     "default_value": ...,
///     "minval": ... (continuous, optional),
///     "maxval": ... (continuous, optional)
///   },
///   ...
/// }
///
/// Keys are sorted lexicographically so the descriptor order matches trainer behavior.
std::vector<ParameterDescriptor> ParseParameterDescriptors(const nlohmann::json& parametric_config);
} // namespace parametric
} // namespace nam
