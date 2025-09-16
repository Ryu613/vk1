#pragma once

#include "vk1/common/vk_type.hpp"

#include <unordered_map>
#include <filesystem>

namespace vk1 {
struct GeoSurface {
  uint32_t startIndex;
  uint32_t count;
};

struct MeshAsset {
  std::string name;
  std::vector<GeoSurface> surfaces;
  GPUMeshBuffers meshBuffers;
};

class Context;

std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(Context* ctx,
                                                                      std::filesystem::path filePath);
}  // namespace vk1
