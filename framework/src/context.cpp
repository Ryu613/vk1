#include "vk1/core/context.hpp"

#include <chrono>
#include <thread>
#include <iostream>

#include "VkBootstrap.h"
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

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

  initImgui();

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
  // VkClearColorValue clearValue;
  //// blue color flash periodically
  // float flash = std::abs(std::sin(static_cast<float>(frameNumber) / 120.0f));
  // clearValue = {{0.0f, 0.0f, flash, 1.0f}};

  // VkImageSubresourceRange clearRange = init::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
  // vkCmdClearColorImage(cmd, drawImage.image, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &clearRange);

  ComputeEffect& effect = bgEffects[currentBgEffect];

  // bind the gradient drawing compute pipeline
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);

  // bind the descriptor set containing the draw image for the compute pipeline
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gradientPipelineLayout, 0, 1, &drawImageDescriptors, 0, nullptr);

  // ComputePushConstants pc{
  //     .data1 = glm::vec4(1, 0, 0, 1),
  //     .data2 = glm::vec4(0, 0, 1, 1),
  // };

  vkCmdPushConstants(cmd,
                     gradientPipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT,
                     0,
                     sizeof(ComputePushConstants),
                     &effect.data);

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

  util::transition_image(cmd, drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

  drawBackground(cmd);

  util::transition_image(
      cmd, drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

  drawGeometry(cmd);

  util::transition_image(cmd, drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

  util::transition_image(cmd,
                         swapchainImages[swapchainImageIndex],
                         VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

  util::copy_image_to_image(
      cmd, drawImage.image, swapchainImages[swapchainImageIndex], drawExtent, swapchainExtent);

  // set swapchain image layout to Attachment Optimal so we can draw it
  util::transition_image(cmd,
                         swapchainImages[swapchainImageIndex],
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

  drawImgui(cmd, swapchainImageViews[swapchainImageIndex]);

  util::transition_image(cmd,
                         swapchainImages[swapchainImageIndex],
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
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

void Context::drawGeometry(VkCommandBuffer cmd) {
  VkRenderingAttachmentInfo colorAttachment =
      init::attachment_info(drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  VkRenderingInfo renderInfo = init::rendering_info(drawExtent, &colorAttachment, nullptr);
  vkCmdBeginRendering(cmd, &renderInfo);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, trianglePipeline);

  VkViewport viewport{
      .x = 0,
      .y = 0,
      .width = static_cast<float>(drawExtent.width),
      .height = static_cast<float>(drawExtent.height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };

  vkCmdSetViewport(cmd, 0, 1, &viewport);

  VkRect2D scissor{
      .offset = {0, 0},
      .extent = {drawExtent.width, drawExtent.height},
  };

  vkCmdSetScissor(cmd, 0, 1, &scissor);

  vkCmdDraw(cmd, 3, 1, 0, 0);

  vkCmdEndRendering(cmd);
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

      ImGui_ImplSDL2_ProcessEvent(&sdlEvent);
    }

    if (stopRendering) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    if (ImGui::Begin("background")) {
      ComputeEffect& selected = bgEffects[currentBgEffect];

      ImGui::Text("selected effect: ", selected.name);

      ImGui::SliderInt("Effect Index", &currentBgEffect, 0, bgEffects.size() - 1);

      ImGui::InputFloat4("data1", (float*)&selected.data.data1);
      ImGui::InputFloat4("data2", (float*)&selected.data.data2);
      ImGui::InputFloat4("data3", (float*)&selected.data.data3);
      ImGui::InputFloat4("data4", (float*)&selected.data.data4);
      ImGui::End();
    }

    ImGui::Render();

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

  initTrianglePipeline();
}

void Context::initBackgroundPipelines() {
  // create pipeline layout
  VkPipelineLayoutCreateInfo computeLayout{};
  computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  computeLayout.pNext = nullptr;
  computeLayout.pSetLayouts = &drawImageDescriptorLayout;
  computeLayout.setLayoutCount = 1;

  VkPushConstantRange pushConstant{};
  pushConstant.offset = 0;
  pushConstant.size = sizeof(ComputePushConstants);
  pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  computeLayout.pPushConstantRanges = &pushConstant;
  computeLayout.pushConstantRangeCount = 1;

  VK_CHECK(vkCreatePipelineLayout(device, &computeLayout, nullptr, &gradientPipelineLayout));

  // create pipeline
  VkShaderModule gradientShader{};
  if (!util::load_shader_module("shaders/gradient_color.comp.spv", device, &gradientShader)) {
    fmt::print("Error when building the compute shader \n");
  }

  VkShaderModule skyShader{};
  if (!util::load_shader_module("shaders/sky.comp.spv", device, &skyShader)) {
    fmt::print("Error when building the compute shader \n");
  }

  VkPipelineShaderStageCreateInfo stageinfo{};
  stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stageinfo.pNext = nullptr;
  stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stageinfo.module = gradientShader;
  stageinfo.pName = "main";

  VkComputePipelineCreateInfo computePipelineCreateInfo{};
  computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  computePipelineCreateInfo.pNext = nullptr;
  computePipelineCreateInfo.layout = gradientPipelineLayout;
  computePipelineCreateInfo.stage = stageinfo;

  ComputeEffect gradient{};
  gradient.layout = gradientPipelineLayout;
  gradient.name = "gradient";
  gradient.data = {};

  gradient.data.data1 = glm::vec4(1, 0, 0, 1);
  gradient.data.data2 = glm::vec4(0, 0, 1, 1);

  VK_CHECK(vkCreateComputePipelines(
      device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &gradient.pipeline));

  computePipelineCreateInfo.stage.module = skyShader;

  ComputeEffect sky{};
  gradient.layout = gradientPipelineLayout;
  gradient.name = "sky";
  gradient.data = {};

  gradient.data.data1 = glm::vec4(0.1, 0.2, 0.4, 0.97);

  VK_CHECK(vkCreateComputePipelines(
      device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &sky.pipeline));

  bgEffects.push_back(gradient);
  bgEffects.push_back(sky);

  vkDestroyShaderModule(device, gradientShader, nullptr);
  vkDestroyShaderModule(device, skyShader, nullptr);

  mainDeletionQueue.pushFunction([=]() {
    vkDestroyPipelineLayout(device, gradientPipelineLayout, nullptr);
    vkDestroyPipeline(device, sky.pipeline, nullptr);
    vkDestroyPipeline(device, gradient.pipeline, nullptr);
  });
}

void Context::initImgui() {
  // 1: create descriptor pool for IMGUI
  //  the size of the pool is very oversize, but it's copied from imgui demo
  //  itself.
  VkDescriptorPoolSize pool_sizes[] = {{VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
                                       {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
                                       {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
                                       {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
                                       {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
                                       {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
                                       {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
                                       {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
                                       {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
                                       {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
                                       {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};

  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets = 1000;
  pool_info.poolSizeCount = static_cast<uint32_t>(std::size(pool_sizes));
  pool_info.pPoolSizes = pool_sizes;

  VkDescriptorPool imguiPool{};
  VK_CHECK(vkCreateDescriptorPool(device, &pool_info, nullptr, &imguiPool));

  // 2: initialize imgui library

  // this initializes the core structures of imgui
  ImGui::CreateContext();

  // this initializes imgui for SDL
  ImGui_ImplSDL2_InitForVulkan(window);

  // this initializes imgui for Vulkan
  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = instance;
  init_info.PhysicalDevice = gpu;
  init_info.Device = device;
  init_info.Queue = graphicsQueue;
  init_info.DescriptorPool = imguiPool;
  init_info.MinImageCount = 3;
  init_info.ImageCount = 3;
  init_info.UseDynamicRendering = true;

  // dynamic rendering parameters for imgui to use
  init_info.PipelineRenderingCreateInfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
  init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapchainImageFormat;

  init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

  ImGui_ImplVulkan_Init(&init_info);

  ImGui_ImplVulkan_CreateFontsTexture();

  // add the destroy the imgui created structures
  mainDeletionQueue.pushFunction([=]() {
    ImGui_ImplVulkan_Shutdown();
    vkDestroyDescriptorPool(device, imguiPool, nullptr);
  });
}

void Context::drawImgui(VkCommandBuffer cmd, VkImageView targetImageView) {
  VkRenderingAttachmentInfo colorAttachment =
      init::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  VkRenderingInfo renderInfo = init::rendering_info(swapchainExtent, &colorAttachment, nullptr);

  vkCmdBeginRendering(cmd, &renderInfo);

  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

  vkCmdEndRendering(cmd);
}

void Context::initTrianglePipeline() {
  VkShaderModule triangleFragShader{};
  if (!util::load_shader_module("shaders/colored_triangle.frag.spv", device, &triangleFragShader)) {
    fmt::print("Error when building the triangle fragment shader \n");
  }
  VkShaderModule triangleVertShader{};
  if (!util::load_shader_module("shaders/colored_triangle.vert.spv", device, &triangleVertShader)) {
    fmt::print("Error when building the triangle vertex shader \n");
  }
  VkPipelineLayoutCreateInfo pipelineLayoutInfo = init::pipeline_layout_create_info();
  VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &trianglePipelineLayout));

  Pipeline::Builder builder;
  builder.pipelineLayout = trianglePipelineLayout;
  builder.shaders(triangleVertShader, triangleFragShader);
  builder.inputTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  builder.polygonMode(VK_POLYGON_MODE_FILL);
  builder.cullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
  builder.multisamplingNone();
  builder.disableBlending();
  builder.disableDepthTest();
  builder.colorAttachmentFormat(drawImage.imageFormat);
  builder.depthFormat(VK_FORMAT_UNDEFINED);

  trianglePipeline = builder.build(device);

  vkDestroyShaderModule(device, triangleFragShader, nullptr);
  vkDestroyShaderModule(device, triangleVertShader, nullptr);

  mainDeletionQueue.pushFunction([&]() {
    vkDestroyPipelineLayout(device, trianglePipelineLayout, nullptr);
    vkDestroyPipeline(device, trianglePipeline, nullptr);
  });
}

void Context::immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function) {}

AllocatedBuffer Context::createBuffer(size_t allocSize,
                                      VkBufferUsageFlags usage,
                                      VmaMemoryUsage memoryUsage) const {
  VkBufferCreateInfo bufferInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
  };
  bufferInfo.size = allocSize;
  bufferInfo.usage = usage;

  VmaAllocationCreateInfo vmaallocInfo{
      .usage = memoryUsage,
      .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
  };
  AllocatedBuffer newBuffer{};

  VK_CHECK(vmaCreateBuffer(
      allocator, &bufferInfo, &vmaallocInfo, &newBuffer.buffer, &newBuffer.allocation, &newBuffer.info));

  return newBuffer;
}

void Context::destroyBuffer(const AllocatedBuffer& buffer) const {
  vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation);
}

GPUMeshBuffers Context::uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices) {
  const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
  const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

  GPUMeshBuffers newSurface{};

  newSurface.vertexBuffer =
      createBuffer(vertexBufferSize,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                   VMA_MEMORY_USAGE_CPU_ONLY);
  VkBufferDeviceAddressInfo deviceAddressInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = newSurface.vertexBuffer.buffer,
  };
  newSurface.vertexBufferAddress = vkGetBufferDeviceAddress(device, &deviceAddressInfo);
  newSurface.indexBuffer = createBuffer(indexBufferSize,
                                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        VMA_MEMORY_USAGE_GPU_ONLY);

  AllocatedBuffer staging = createBuffer(
      vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
  void* data = staging.allocation->GetMappedData();

  memcpy(data, vertices.data(), vertexBufferSize);
  memcpy((char*)data + vertexBufferSize, indices.data(), indexBufferSize);

  immediateSubmit([&](VkCommandBuffer cmd) {
    VkBufferCopy vertexCopy{0};
    vertexCopy.dstOffset = 0;
    vertexCopy.srcOffset = 0;
    vertexCopy.size = vertexBufferSize;

    vkCmdCopyBuffer(cmd, staging.buffer, newSurface.vertexBuffer.buffer, 1, &vertexCopy);

    VkBufferCopy indexCopy{0};
    indexCopy.dstOffset = 0;
    indexCopy.srcOffset = vertexBufferSize;
    indexCopy.size = indexBufferSize;

    vkCmdCopyBuffer(cmd, staging.buffer, newSurface.indexBuffer.buffer, 1, &indexCopy);
  });

  destroyBuffer(staging);

  return newSurface;
}
}  // namespace vk1
