#pragma once

#include "vk1/common/common.hpp"

#include "vulkan/vulkan.h"
#include "vk_mem_alloc.h"
#include "vulkan/vk_enum_string_helper.h"

#include "fmt/core.h"

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/mat4x4.hpp"
#include "glm/vec4.hpp"

#define VK_CHECK(x)                                                  \
  do {                                                               \
    VkResult err = x;                                                \
    if (err) {                                                       \
      fmt::print("Detected Vulkan error: {}", string_VkResult(err)); \
      abort();                                                       \
    }                                                                \
  } while (0)
