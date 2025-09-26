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

struct AllocatedBuffer {
  VkBuffer buffer;
  VmaAllocation allocation;
  VmaAllocationInfo info;
};

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec4 color;
  float uv_x;
  float uv_y;
};

// holds the resources needed for a mesh
struct GPUMeshBuffers {
  AllocatedBuffer indexBuffer;
  AllocatedBuffer vertexBuffer;
  VkDeviceAddress vertexBufferAddress;
};

// push constants for our mesh object draws
struct GPUDrawPushConstants {
  glm::mat4 worldMatrix;
  VkDeviceAddress vertexBuffer;
};

struct GPUSceneData {
  glm::mat4 view;
  glm::mat4 proj;
  glm::mat4 viewproj;
  glm::vec4 ambientColor;
  glm::vec4 sunlightDirection;  // w for sun power
  glm::vec4 sunlightColor;
};

enum class MaterialPass : uint8_t {
  MainColor,
  Transparent,
  Other,
};

struct MaterialPipeline {
  VkPipeline pipeline;
  VkPipelineLayout layout;
};

struct MaterialInstance {
  MaterialPipeline* pipeline;
  VkDescriptorSet materialSet;
  MaterialPass passType;
};

struct DrawContext;

class IRenderable {
  virtual void draw(const glm::mat4& topMatrix, DrawContext& ctx) = 0;
};

struct Node : public IRenderable {
  std::weak_ptr<Node> parent;
  std::vector<std::shared_ptr<Node>> children;

  glm::mat4 localTransform;
  glm::mat4 worldTransform;

  void refreshTransform(const glm::mat4& parentMatrix) {
    worldTransform = parentMatrix * localTransform;
    for (auto c : children) {
      c->refreshTransform(worldTransform);
    }
  }

  virtual void draw(const glm::mat4& topMatrix, DrawContext& ctx) {
    for (auto& c : children) {
      c->draw(topMatrix, ctx);
    }
  }
};
}  // namespace vk1
