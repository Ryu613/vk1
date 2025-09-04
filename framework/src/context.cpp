#include "vk1/core/context.hpp"

#include <chrono>
#include <thread>
#include <iostream>

#include "VkBootstrap.h"
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include "vk1/common/vk_init.hpp"
#include "vk1/common/vk_util.hpp"

#include "vk1/core/pipeline.hpp"

namespace vk1 {
namespace {
constexpr bool bUseValidationLayers = true;
}  // namespace

Context& Context::get() {
  return *loadedContext;
}

void Context::init() {
  assert(loadedContext == nullptr);
  loadedContext = this;
  // init window
  SDL_Init(SDL_INIT_VIDEO);
  auto windowFlags = static_cast<SDL_WindowFlags>(SDL_WINDOW_VULKAN);
  window = SDL_CreateWindow("Vulkan Window",
                            SDL_WINDOWPOS_UNDEFINED,
                            SDL_WINDOWPOS_UNDEFINED,
                            static_cast<int>(windowExtent.width),
                            static_cast<int>(windowExtent.height),
                            windowFlags);
  // fixed: vcpkg sdl2 not correct, use sdl2[vulkan] instead!
  if (window == nullptr) {
    fmt::println("SDL create window failed: {}", SDL_GetError());
    return;
  }
  // init vulkan
  initVulkan();
  initSwapchain();
  initCommands();
  initSyncStructures();
  initDescriptors();
  initPipelines();

  initialized = true;
}

void Context::cleanup() {
  if (initialized) {
    vkDeviceWaitIdle(device);

    mainDeletionQueue.flush();

    for (auto& eachFrame : frames) {
      vkDestroyCommandPool(device, eachFrame.commandPool, nullptr);

      vkDestroyFence(device, eachFrame.renderFence, nullptr);
      vkDestroySemaphore(device, eachFrame.renderSemaphore, nullptr);
      vkDestroySemaphore(device, eachFrame.swapchainSemaphore, nullptr);

      eachFrame.deletionQueue.flush();
    }

    destroySwapchain();

    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyDevice(device, nullptr);

    vkb::destroy_debug_utils_messenger(instance, debugMsgr);

    vkDestroyInstance(instance, nullptr);

    SDL_DestroyWindow(window);
  }

  loadedContext = nullptr;
}

void Context::drawBackground(VkCommandBuffer cmd) {
  VkClearColorValue clearValue;
  //// blue color flash periodically
  //float flash = std::abs(std::sin(static_cast<float>(frameNumber) / 120.0f));
  //clearValue = {{0.0f, 0.0f, flash, 1.0f}};

  //VkImageSubresourceRange clearRange = init::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
  //vkCmdClearColorImage(cmd, drawImage.image, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &clearRange);

  // bind the gradient drawing compute pipeline
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gradientPipeline);

  // bind the descriptor set containing the draw image for the compute pipeline
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gradientPipelineLayout, 0, 1, &drawImageDescriptors, 0, nullptr);

  // execute the compute pipeline dispatch. We are using 16x16 workgroup size so we need to divide by it
  vkCmdDispatch(cmd, std::ceil(drawExtent.width / 16.0), std::ceil(drawExtent.height / 16.0), 1);
}

void Context::draw() {
  // wait gpu finished last frame rendering, timeout of 1 sec
  VK_CHECK(vkWaitForFences(device, 1, &getCurrentFrame().renderFence, true, 1000000000));

  getCurrentFrame().deletionQueue.flush();

  // request image from swapchain
  uint32_t swapchainImageIndex{};
  VK_CHECK(vkAcquireNextImageKHR(
      device, swapchain, 1000000000, getCurrentFrame().swapchainSemaphore, nullptr, &swapchainImageIndex));

  VK_CHECK(vkResetFences(device, 1, &getCurrentFrame().renderFence));

  // reset command buffer and record again
  VkCommandBuffer cmd = getCurrentFrame().mainCommandBuffer;
  VK_CHECK(vkResetCommandBuffer(cmd, 0));

  auto cmdBeginInfo = init::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

  drawExtent.width = drawImage.imageExtent.width;
  drawExtent.height = drawImage.imageExtent.height;

  VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

  util::transition_image(
      cmd, swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

  drawBackground(cmd);

  util::transition_image(cmd, drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  util::transition_image(cmd,
                         swapchainImages[swapchainImageIndex],
                         VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

  util::copy_image_to_image(
      cmd, drawImage.image, swapchainImages[swapchainImageIndex], drawExtent, swapchainExtent);

  util::transition_image(cmd,
                         swapchainImages[swapchainImageIndex],
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

  VK_CHECK(vkEndCommandBuffer(cmd));

  // prepare submission to device queue
  VkCommandBufferSubmitInfo cmdSubmitInfo = init::command_buffer_submit_info(cmd);
  VkSemaphoreSubmitInfo waitInfo = init::semaphore_submit_info(
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, getCurrentFrame().swapchainSemaphore);
  VkSemaphoreSubmitInfo signalInfo =
      init::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, getCurrentFrame().renderSemaphore);

  VkSubmitInfo2 submit = init::submit_info(&cmdSubmitInfo, &signalInfo, &waitInfo);

  VK_CHECK(vkQueueSubmit2(graphicsQueue, 1, &submit, getCurrentFrame().renderFence));

  // prepare present
  VkPresentInfoKHR presentInfo{
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .pNext = nullptr,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &getCurrentFrame().renderSemaphore,
      .swapchainCount = 1,
      .pSwapchains = &swapchain,
      .pImageIndices = &swapchainImageIndex,
  };

  VK_CHECK(vkQueuePresentKHR(graphicsQueue, &presentInfo));

  frameNumber++;
}

void Context::run() {
  SDL_Event sdlEvent;
  bool bQuit = false;

  while (!bQuit) {
    while (SDL_PollEvent(&sdlEvent) != 0) {
      if (sdlEvent.type == SDL_QUIT) {
        bQuit = true;
      }
      if (sdlEvent.type == SDL_WINDOWEVENT) {
        if (sdlEvent.window.event == SDL_WINDOWEVENT_MINIMIZED) {
          stopRendering = true;
        }
        if (sdlEvent.window.event == SDL_WINDOWEVENT_RESTORED) {
          stopRendering = false;
        }
      }
    }

    if (stopRendering) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }

    draw();
  }
}

void Context::initVulkan() {
  // init instance
  vkb::InstanceBuilder builder;

  auto instRet = builder.set_app_name("vulkan app")
                     .request_validation_layers(bUseValidationLayers)
                     .use_default_debug_messenger()
                     .require_api_version(1, 3, 0)
                     .build();
  vkb::Instance vkbInst = instRet.value();
  instance = vkbInst.instance;
  debugMsgr = vkbInst.debug_messenger;
  // init device
  SDL_Vulkan_CreateSurface(window, instance, &surface);

  // vulkan 1.3 features
  VkPhysicalDeviceVulkan13Features features13{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
  };
  features13.dynamicRendering = VK_TRUE;
  features13.synchronization2 = VK_TRUE;

  // vulkan 1.2 features
  VkPhysicalDeviceVulkan12Features features12{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
  };
  features12.bufferDeviceAddress = VK_TRUE;
  features12.descriptorIndexing = VK_TRUE;

  // gpu select, need to support above features
  vkb::PhysicalDeviceSelector selector{vkbInst};
  auto physicalDevice = selector.set_minimum_version(1, 3)
                            .set_required_features_13(features13)
                            .set_required_features_12(features12)
                            .set_surface(surface)
                            .select()
                            .value();
  // create device
  vkb::DeviceBuilder deviceBuilder{physicalDevice};
  vkb::Device vkbDevice = deviceBuilder.build().value();

  device = vkbDevice.device;
  gpu = physicalDevice.physical_device;

  // get graphics queue
  graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
  graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

  // init vma
  VmaAllocatorCreateInfo allocatorInfo{
      .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice = gpu,
      .device = device,
      .instance = instance,
  };

  vmaCreateAllocator(&allocatorInfo, &allocator);

  mainDeletionQueue.pushFunction([&]() { vmaDestroyAllocator(allocator); });
}

void Context::createSwapchain(VkExtent2D extent) {
  vkb::SwapchainBuilder swapchainBuilder{gpu, device, surface};
  swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;
  vkb::Swapchain vkbSwapchain = swapchainBuilder
                                    .set_desired_format(VkSurfaceFormatKHR{
                                        .format = swapchainImageFormat,
                                        .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
                                    })
                                    .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                                    .set_desired_extent(extent.width, extent.height)
                                    .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                                    .build()
                                    .value();
  swapchainExtent = vkbSwapchain.extent;
  swapchain = vkbSwapchain.swapchain;
  swapchainImages = vkbSwapchain.get_images().value();
  swapchainImageViews = vkbSwapchain.get_image_views().value();
}

void Context::destroySwapchain() {
  vkDestroySwapchainKHR(device, swapchain, nullptr);
  for (auto& eachView : swapchainImageViews) {
    vkDestroyImageView(device, eachView, nullptr);
  }
}

void Context::initSwapchain() {
  createSwapchain(windowExtent);

  VkExtent3D drawImageExtent{
      .width = windowExtent.width,
      .height = windowExtent.height,
      .depth = 1,
  };

  drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
  drawImage.imageExtent = drawImageExtent;

  VkImageUsageFlags drawImageUsages{};
  drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
  drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  VkImageCreateInfo rimgInfo =
      init::image_create_info(drawImage.imageFormat, drawImageUsages, drawImageExtent);

  VmaAllocationCreateInfo rimgAllocInfo{
      .usage = VMA_MEMORY_USAGE_GPU_ONLY,
      .requiredFlags = static_cast<VkMemoryPropertyFlags>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
  };

  vmaCreateImage(allocator, &rimgInfo, &rimgAllocInfo, &drawImage.image, &drawImage.allocation, nullptr);

  // build a image view for draw image for rendering
  VkImageViewCreateInfo rviewInfo =
      init::imageview_create_info(drawImage.imageFormat, drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);

  VK_CHECK(vkCreateImageView(device, &rviewInfo, nullptr, &drawImage.imageView));

  mainDeletionQueue.pushFunction([=]() {
    vkDestroyImageView(device, drawImage.imageView, nullptr);
    vmaDestroyImage(allocator, drawImage.image, drawImage.allocation);
  });
}

void Context::initCommands() {
  VkCommandPoolCreateInfo commandPoolInfo =
      init::command_pool_create_info(graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  // create command pool & buffer for each overlapped frame
  for (auto& eachFrame : frames) {
    VK_CHECK(vkCreateCommandPool(device, &commandPoolInfo, nullptr, &eachFrame.commandPool));

    // allocate default command buffer for rendering
    VkCommandBufferAllocateInfo cmdAllocInfo = init::command_buffer_allocate_info(eachFrame.commandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &eachFrame.mainCommandBuffer));
  }
}

void Context::initSyncStructures() {
  auto fenceCreateInfo = init::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
  auto semaphoreCreateInfo = init::semaphore_create_info();
  // create sync object for every frames
  for (auto& eachFrame : frames) {
    VK_CHECK(vkCreateFence(device, &fenceCreateInfo, nullptr, &eachFrame.renderFence));

    VK_CHECK(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &eachFrame.swapchainSemaphore));
    VK_CHECK(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &eachFrame.renderSemaphore));
  }
}

void Context::initDescriptors() {
  // create a descriptor pool that will hold 10 sets with 1 image each
  std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};

  globalDescriptorAllocator.initPool(device, 10, sizes);

  {
    DescriptorLayoutBuilder builder;
    builder.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    drawImageDescriptorLayout = builder.build(device, VK_SHADER_STAGE_COMPUTE_BIT);
  }

  drawImageDescriptors = globalDescriptorAllocator.allocate(device, drawImageDescriptorLayout);

  VkDescriptorImageInfo imgInfo{
      .imageView = drawImage.imageView,
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
  };

  VkWriteDescriptorSet drawImageWrite = {};
  drawImageWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  drawImageWrite.pNext = nullptr;

  drawImageWrite.dstBinding = 0;
  drawImageWrite.dstSet = drawImageDescriptors;
  drawImageWrite.descriptorCount = 1;
  drawImageWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  drawImageWrite.pImageInfo = &imgInfo;

  vkUpdateDescriptorSets(device, 1, &drawImageWrite, 0, nullptr);

  mainDeletionQueue.pushFunction([&]() {
    globalDescriptorAllocator.destroyPool(device);

    vkDestroyDescriptorSetLayout(device, drawImageDescriptorLayout, nullptr);
  });
}

void Context::initPipelines() {
  initBackgroundPipelines();
}

void Context::initBackgroundPipelines() {
  // create pipeline layout
  VkPipelineLayoutCreateInfo computeLayout{};
  computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  computeLayout.pNext = nullptr;
  computeLayout.pSetLayouts = &drawImageDescriptorLayout;
  computeLayout.setLayoutCount = 1;

  VK_CHECK(vkCreatePipelineLayout(device, &computeLayout, nullptr, &gradientPipelineLayout));

  // create pipeline
  VkShaderModule computeDrawShader{};
  if (!util::load_shader_module("shaders/gradient.comp.spv", device, &computeDrawShader)) {
    fmt::print("Error when building the compute shader \n");
  }

  VkPipelineShaderStageCreateInfo stageinfo{};
  stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stageinfo.pNext = nullptr;
  stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stageinfo.module = computeDrawShader;
  stageinfo.pName = "main";

  VkComputePipelineCreateInfo computePipelineCreateInfo{};
  computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  computePipelineCreateInfo.pNext = nullptr;
  computePipelineCreateInfo.layout = gradientPipelineLayout;
  computePipelineCreateInfo.stage = stageinfo;

  VK_CHECK(vkCreateComputePipelines(
      device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &gradientPipeline));

  vkDestroyShaderModule(device, computeDrawShader, nullptr);

  mainDeletionQueue.pushFunction([&]() {
    vkDestroyPipelineLayout(device, gradientPipelineLayout, nullptr);
    vkDestroyPipeline(device, gradientPipeline, nullptr);
  });
}

}  // namespace vk1
