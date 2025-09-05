#pragma once

#include "vk1/common/vk_common.hpp"

#include <array>
#include <deque>
#include <functional>

#include "SDL.h"
#include "SDL_vulkan.h"

#include "vk1/common/vk_type.hpp"
#include "vk1/core/descriptor.hpp"

namespace vk1 {

constexpr unsigned int FRAME_OVERLAP = 2;

struct DeletionQueue {
  std::deque<std::function<void()>> deletors;

  void pushFunction(std::function<void()>&& function) {
    deletors.push_back(function);
  }

  void flush() {
    for (auto it = deletors.rbegin(); it != deletors.rend(); ++it) {
      (*it)();
    }
    deletors.clear();
  }
};

struct FrameData {
  VkCommandPool commandPool{VK_NULL_HANDLE};
  VkCommandBuffer mainCommandBuffer{VK_NULL_HANDLE};

  VkSemaphore swapchainSemaphore{VK_NULL_HANDLE};
  VkSemaphore renderSemaphore{VK_NULL_HANDLE};
  VkFence renderFence{VK_NULL_HANDLE};

  DeletionQueue deletionQueue;
};

struct ComputePushConstants {
  glm::vec4 data1;
  glm::vec4 data2;
  glm::vec4 data3;
  glm::vec4 data4;
};

struct ComputeEffect {
  const char* name;
  VkPipeline pipeline;
  VkPipelineLayout layout;
  ComputePushConstants data;
};

class Context {
 public:
  bool initialized{false};
  int frameNumber{0};
  bool stopRendering{false};

  VkExtent2D windowExtent{1920, 1080};
  SDL_Window* window{nullptr};

  VkInstance instance{VK_NULL_HANDLE};
  VkDebugUtilsMessengerEXT debugMsgr{VK_NULL_HANDLE};
  VkPhysicalDevice gpu{VK_NULL_HANDLE};
  VkDevice device{VK_NULL_HANDLE};
  VkSurfaceKHR surface{VK_NULL_HANDLE};

  VkSwapchainKHR swapchain{VK_NULL_HANDLE};
  VkFormat swapchainImageFormat{};

  std::vector<VkImage> swapchainImages;
  std::vector<VkImageView> swapchainImageViews;
  VkExtent2D swapchainExtent{};

  std::array<FrameData, FRAME_OVERLAP> frames;

  DeletionQueue mainDeletionQueue;

  VmaAllocator allocator;

  AllocatedImage drawImage;
  VkExtent2D drawExtent;

  DescriptorAllocator globalDescriptorAllocator;

  VkDescriptorSet drawImageDescriptors;
  VkDescriptorSetLayout drawImageDescriptorLayout;

  VkPipeline gradientPipeline;
  VkPipelineLayout gradientPipelineLayout;

  inline FrameData& getCurrentFrame() {
    return frames[frameNumber % FRAME_OVERLAP];
  }

  VkQueue graphicsQueue{VK_NULL_HANDLE};
  uint32_t graphicsQueueFamily{};

  std::vector<ComputeEffect> bgEffects;
  int currentBgEffect{0};

  static Context& get();

  // initializes everything in the engine
  void init();

  // shuts down the engine
  void cleanup();

  // draw loop
  void draw();

  // run main loop
  void run();

 private:
  inline static Context* loadedContext = nullptr;

  void initVulkan();
  void initSwapchain();
  void initCommands();
  void initSyncStructures();

  void createSwapchain(VkExtent2D extent);
  void destroySwapchain();

  void drawBackground(VkCommandBuffer cmd);

  void initDescriptors();

  void initPipelines();
  void initBackgroundPipelines();

  void initImgui();
  void drawImgui(VkCommandBuffer cmd, VkImageView targetImageView);
};
}  // namespace vk1
