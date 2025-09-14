#pragma once

#include "vk1/common/vk_common.hpp"

namespace vk1::init {
inline VkCommandPoolCreateInfo command_pool_create_info(uint32_t queueFamilyIndex,
                                                        VkCommandPoolCreateFlags flags) {
  VkCommandPoolCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  info.pNext = nullptr;
  info.queueFamilyIndex = queueFamilyIndex;
  info.flags = flags;
  return info;
}

inline VkCommandBufferAllocateInfo command_buffer_allocate_info(VkCommandPool pool, uint32_t count = 1U) {
  VkCommandBufferAllocateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  info.pNext = nullptr;

  info.commandPool = pool;
  info.commandBufferCount = count;
  info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  return info;
}

inline VkFenceCreateInfo fence_create_info(VkFenceCreateFlags flags = 0) {
  VkFenceCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  info.pNext = nullptr;

  info.flags = flags;

  return info;
}

inline VkSemaphoreCreateInfo semaphore_create_info(VkSemaphoreCreateFlags flags = 0) {
  VkSemaphoreCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  info.pNext = nullptr;
  info.flags = flags;
  return info;
}

inline VkCommandBufferBeginInfo command_buffer_begin_info(VkCommandBufferUsageFlags flags = 0) {
  VkCommandBufferBeginInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  info.pNext = nullptr;

  info.pInheritanceInfo = nullptr;
  info.flags = flags;
  return info;
}

inline VkImageSubresourceRange image_subresource_range(VkImageAspectFlags aspectMask) {
  VkImageSubresourceRange subImage{};
  subImage.aspectMask = aspectMask;
  subImage.baseMipLevel = 0;
  subImage.levelCount = VK_REMAINING_MIP_LEVELS;
  subImage.baseArrayLayer = 0;
  subImage.layerCount = VK_REMAINING_ARRAY_LAYERS;

  return subImage;
}

inline VkSemaphoreSubmitInfo semaphore_submit_info(VkPipelineStageFlags2 stageMask, VkSemaphore semaphore) {
  VkSemaphoreSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
  submitInfo.pNext = nullptr;
  submitInfo.semaphore = semaphore;
  submitInfo.stageMask = stageMask;
  submitInfo.deviceIndex = 0;
  submitInfo.value = 1;

  return submitInfo;
}

inline VkCommandBufferSubmitInfo command_buffer_submit_info(VkCommandBuffer cmd) {
  VkCommandBufferSubmitInfo info{};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
  info.pNext = nullptr;
  info.commandBuffer = cmd;
  info.deviceMask = 0;

  return info;
}

inline VkSubmitInfo2 submit_info(VkCommandBufferSubmitInfo* cmd,
                                 VkSemaphoreSubmitInfo* signalSemaphoreInfo,
                                 VkSemaphoreSubmitInfo* waitSemaphoreInfo) {
  VkSubmitInfo2 info = {};
  info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
  info.pNext = nullptr;

  info.waitSemaphoreInfoCount = waitSemaphoreInfo == nullptr ? 0 : 1;
  info.pWaitSemaphoreInfos = waitSemaphoreInfo;

  info.signalSemaphoreInfoCount = signalSemaphoreInfo == nullptr ? 0 : 1;
  info.pSignalSemaphoreInfos = signalSemaphoreInfo;

  info.commandBufferInfoCount = 1;
  info.pCommandBufferInfos = cmd;

  return info;
}

inline VkImageCreateInfo image_create_info(VkFormat format, VkImageUsageFlags usageFlags, VkExtent3D extent) {
  VkImageCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.pNext = nullptr;

  info.imageType = VK_IMAGE_TYPE_2D;

  info.format = format;
  info.extent = extent;

  info.mipLevels = 1;
  info.arrayLayers = 1;

  // for MSAA. we will not be using it by default, so default it to 1 sample per pixel.
  info.samples = VK_SAMPLE_COUNT_1_BIT;

  // optimal tiling, which means the image is stored on the best gpu format
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = usageFlags;

  return info;
}

inline VkImageViewCreateInfo imageview_create_info(VkFormat format,
                                                   VkImage image,
                                                   VkImageAspectFlags aspectFlags) {
  // build a image-view for the depth image to use for rendering
  VkImageViewCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  info.pNext = nullptr;

  info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  info.image = image;
  info.format = format;
  info.subresourceRange.baseMipLevel = 0;
  info.subresourceRange.levelCount = 1;
  info.subresourceRange.baseArrayLayer = 0;
  info.subresourceRange.layerCount = 1;
  info.subresourceRange.aspectMask = aspectFlags;

  return info;
}

inline VkRenderingAttachmentInfo attachment_info(VkImageView view,
                                                 VkClearValue* clear,
                                                 VkImageLayout layout) {
  VkRenderingAttachmentInfo colorAttachment{
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .pNext = nullptr,
      .imageView = view,
      .imageLayout = layout,
      .loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
  };

  if (clear) {
    colorAttachment.clearValue = *clear;
  }

  return colorAttachment;
}

inline VkRenderingInfo rendering_info(VkExtent2D renderExtent,
                                      VkRenderingAttachmentInfo* colorAttachment,
                                      VkRenderingAttachmentInfo* depthAttachment) {
  VkRenderingInfo renderInfo{};
  renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
  renderInfo.pNext = nullptr;

  renderInfo.renderArea = VkRect2D{VkOffset2D{0, 0}, renderExtent};
  renderInfo.layerCount = 1;
  renderInfo.colorAttachmentCount = 1;
  renderInfo.pColorAttachments = colorAttachment;
  renderInfo.pDepthAttachment = depthAttachment;
  renderInfo.pStencilAttachment = nullptr;

  return renderInfo;
}

inline VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info(VkShaderStageFlagBits stage,
                                                                         VkShaderModule shaderModule,
                                                                         const char* entry = "main") {
  VkPipelineShaderStageCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  info.pNext = nullptr;

  // shader stage
  info.stage = stage;
  // module containing the code for this shader stage
  info.module = shaderModule;
  // the entry point of the shader
  info.pName = entry;
  return info;
}

inline VkPipelineLayoutCreateInfo pipeline_layout_create_info() {
  VkPipelineLayoutCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  info.pNext = nullptr;

  // empty defaults
  info.flags = 0;
  info.setLayoutCount = 0;
  info.pSetLayouts = nullptr;
  info.pushConstantRangeCount = 0;
  info.pPushConstantRanges = nullptr;
  return info;
}
}  // namespace vk1::init
