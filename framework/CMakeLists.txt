include(${PROJECT_ROOT_DIR}/cmake/deps.cmake)

file(GLOB_RECURSE vk1_src CONFIGURE_DEPENDS include/*.hpp src/*.cpp src/*.hpp)

add_library(vk1 STATIC ${vk1_src})

target_include_directories(vk1 PUBLIC include)

add_executable(test_vk1 test/main.cpp)

target_link_libraries(test_vk1 PRIVATE vk1)
