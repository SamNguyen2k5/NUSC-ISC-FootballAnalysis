cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(NUSC-ISC-FootballAnalysis CXX)

# 1. Automatically detect Apple Silicon or Intel Homebrew paths
if(NOT CMAKE_PREFIX_PATH)
    # Check Apple Silicon path first, fallback to Intel path
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/usr/local")
endif()

# 2. Locate the PyTorch package installed by Homebrew
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)

# 3. Ensure we use the proper C++ standard required by LibTorch
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 4. Define your executable blueprint
# 5. Link the LibTorch libraries and include directories
file(GLOB_RECURSE SHARED_SOURCES "src/*/*.cpp")
add_executable(main src/main.cpp ${SHARED_SOURCES})

target_link_libraries(main PRIVATE 
    ${TORCH_LIBRARIES} 
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json)

target_include_directories(main PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS})

target_precompile_headers(main PRIVATE
    <iostream>
    <fstream>
    <filesystem>
    <torch/torch.h>
    <nlohmann/json.hpp>
    <argparse/argparse.hpp>
)

# 6. Optional: If using CUDA/MPS features, copy compile flags
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif()
