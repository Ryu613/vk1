#pragma once

#include "vk1/common/vk_common.hpp"

namespace vk1 {
struct AllocatedImage {
  VkImage image;
  VkImageView imageView;
  VmaAllocation allocation;
  VkExtent3D imageExtent;
  VkFormat imageFormat;
};
}
