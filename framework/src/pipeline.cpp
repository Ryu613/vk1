#include "vk1/core/pipeline.hpp"

#include <array>

#include "vk1/common/vk_init.hpp"

namespace vk1 {
void Pipeline::Builder::clear() {
  inputAssembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
  };
  rasterizer = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
  };
  colorBlendAttachment = {};
  multisampling = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
  };
  pipelineLayout = {};
  depthStencil = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
  };
  renderInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
  };
  shaderStages.clear();
}

VkPipeline Pipeline::Builder::build(VkDevice device) {
  VkPipelineViewportStateCreateInfo viewportState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
  };
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;

  VkPipelineColorBlendStateCreateInfo colorBlending{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
  };
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_COPY;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;

  VkPipelineVertexInputStateCreateInfo vertexInputInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
  };

  VkGraphicsPipelineCreateInfo pipelineInfo = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
  };
  pipelineInfo.pNext = &renderInfo;
  pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
  pipelineInfo.pStages = shaderStages.data();
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDepthStencilState = &depthStencil;
  pipelineInfo.layout = pipelineLayout;

  std::array<VkDynamicState, 2> state{
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
  };
  VkPipelineDynamicStateCreateInfo dynamicInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
  };
  dynamicInfo.pDynamicStates = &state[0];
  dynamicInfo.dynamicStateCount = 2;

  pipelineInfo.pDynamicState = &dynamicInfo;

  VkPipeline newPipeline{VK_NULL_HANDLE};
  if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) !=
      VK_SUCCESS) {
    fmt::println("failed to create pipeline");
  }
  return newPipeline;
}

void Pipeline::Builder::shaders(VkShaderModule vertexShader, VkShaderModule fragmentShader) {
  shaderStages.clear();

  shaderStages.push_back(init::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, vertexShader));
  shaderStages.push_back(
      init::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader));
}

void Pipeline::Builder::inputTopology(VkPrimitiveTopology topology) {
  inputAssembly.topology = topology;
  inputAssembly.primitiveRestartEnable = VK_FALSE;
}

void Pipeline::Builder::polygonMode(VkPolygonMode mode) {
  rasterizer.polygonMode = mode;
  rasterizer.lineWidth = 1.0f;
}

void Pipeline::Builder::cullMode(VkCullModeFlags cullMode, VkFrontFace frontFace) {
  rasterizer.cullMode = cullMode;
  rasterizer.frontFace = frontFace;
}

void Pipeline::Builder::multisamplingNone() {
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  multisampling.minSampleShading = 1.0f;
  multisampling.pSampleMask = nullptr;
  multisampling.alphaToCoverageEnable = VK_FALSE;
  multisampling.alphaToOneEnable = VK_FALSE;
}

void Pipeline::Builder::disableBlending() {
  colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_FALSE;
}

void Pipeline::Builder::colorAttachmentFormat(VkFormat format) {
  colorAttFormat = format;
  renderInfo.colorAttachmentCount = 1;
  renderInfo.pColorAttachmentFormats = &colorAttFormat;
}

void Pipeline::Builder::depthFormat(VkFormat format) {
  renderInfo.depthAttachmentFormat = format;
}

void Pipeline::Builder::disableDepthTest() {
  depthStencil.depthTestEnable = VK_FALSE;
  depthStencil.depthWriteEnable = VK_FALSE;
  depthStencil.depthCompareOp = VK_COMPARE_OP_NEVER;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.stencilTestEnable = VK_FALSE;
  depthStencil.front = {};
  depthStencil.back = {};
  depthStencil.minDepthBounds = 0.0f;
  depthStencil.maxDepthBounds = 1.0f;
}
}  // namespace vk1
