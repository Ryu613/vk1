#include "vk1/core/descriptor.hpp"

namespace vk1 {
void DescriptorLayoutBuilder::addBinding(uint32_t binding, VkDescriptorType type) {
  VkDescriptorSetLayoutBinding newbind{
      .binding = binding,
      .descriptorType = type,
      .descriptorCount = 1,
  };

  bindings.push_back(newbind);
}

void DescriptorLayoutBuilder::clear() {
  bindings.clear();
}

VkDescriptorSetLayout DescriptorLayoutBuilder::build(VkDevice device,
                                                     VkShaderStageFlags shaderStages,
                                                     void* pNext,
                                                     VkDescriptorSetLayoutCreateFlags flags) {
  for (auto& binding : bindings) {
    binding.stageFlags |= shaderStages;
  }

  VkDescriptorSetLayoutCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = pNext,
      .flags = flags,
      .bindingCount = static_cast<uint32_t>(bindings.size()),
      .pBindings = bindings.data(),
  };

  VkDescriptorSetLayout set{};
  VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &set));

  return set;
}

void DescriptorAllocator::initPool(VkDevice device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios) {
  std::vector<VkDescriptorPoolSize> poolSizes;
  for (PoolSizeRatio ratio : poolRatios) {
    poolSizes.push_back(VkDescriptorPoolSize{
        .type = ratio.type,
        .descriptorCount = static_cast<uint32_t>(ratio.ratio) * maxSets,
    });
  }

  VkDescriptorPoolCreateInfo pool_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool_info.flags = 0;
  pool_info.maxSets = maxSets;
  pool_info.poolSizeCount = (uint32_t)poolSizes.size();
  pool_info.pPoolSizes = poolSizes.data();

  vkCreateDescriptorPool(device, &pool_info, nullptr, &pool);
}

void DescriptorAllocator::clearDescriptors(VkDevice device) {
  vkResetDescriptorPool(device, pool, 0);
}

void DescriptorAllocator::destroyPool(VkDevice device) {
  vkDestroyDescriptorPool(device, pool, nullptr);
}

VkDescriptorSet DescriptorAllocator::allocate(VkDevice device, VkDescriptorSetLayout layout) {
  VkDescriptorSetAllocateInfo allocInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
  };
  allocInfo.pNext = nullptr;
  allocInfo.descriptorPool = pool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &layout;

  VkDescriptorSet ds;
  VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &ds));

  return ds;
}
}  // namespace vk1
