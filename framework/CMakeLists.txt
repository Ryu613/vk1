include(${PROJECT_ROOT_DIR}/cmake/deps.cmake)

file(GLOB_RECURSE vk1_src CONFIGURE_DEPENDS include/*.hpp src/*.cpp src/*.hpp)

add_library(vk1 STATIC ${vk1_src})

target_link_libraries(vk1 
	PUBLIC 
		fmt::fmt-header-only
		glm::glm-header-only
		Vulkan::Vulkan
		vk-bootstrap::vk-bootstrap
		GPUOpen::VulkanMemoryAllocator
		Vulkan::UtilityHeaders
		$<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
		imgui::imgui
		fastgltf::fastgltf
)

target_include_directories(vk1 PUBLIC include ${Stb_INCLUDE_DIR})

add_executable(test_vk1 test/main.cpp)

add_dependencies(test_vk1 Shaders)

target_link_libraries(test_vk1 PRIVATE vk1)

set_property(TARGET test_vk1 PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:test_vk1>)

add_custom_command(TARGET test_vk1 POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:test_vk1> $<TARGET_FILE_DIR:test_vk1>
  COMMAND_EXPAND_LISTS
  )
