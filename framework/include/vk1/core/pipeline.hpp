#pragma once

#include "vk1/common/vk_common.hpp"

namespace vk1 {
class Pipeline {
 public:
  struct Builder {
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
    VkPipelineInputAssemblyStateCreateInfo inputAssembly;
    VkPipelineRasterizationStateCreateInfo rasterizer;
    VkPipelineColorBlendAttachmentState colorBlendAttachment;
    VkPipelineMultisampleStateCreateInfo multisampling;
    VkPipelineLayout pipelineLayout;
    VkPipelineDepthStencilStateCreateInfo depthStencil;
    VkPipelineRenderingCreateInfo renderInfo;
    VkFormat colorAttFormat;

    Builder() {
      clear();
    }

    void clear();

    VkPipeline build(VkDevice device);

    void shaders(VkShaderModule vertexShader, VkShaderModule fragmentShader);

    void inputTopology(VkPrimitiveTopology topology);

    void polygonMode(VkPolygonMode mode);

    void cullMode(VkCullModeFlags cullMode, VkFrontFace frontFace);

    void multisamplingNone();

    void disableBlending();

    void colorAttachmentFormat(VkFormat format);

    void depthFormat(VkFormat format);

    void disableDepthTest();

    void enableDepthTest(bool depthWriteEnable, VkCompareOp op);
  };
};
}  // namespace vk1
