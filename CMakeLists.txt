cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED PATHS "C:/Users/OpenARK/Desktop/openark_dependencies/opencv-3.4")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
target_link_libraries(example-app "${OpenCV_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
