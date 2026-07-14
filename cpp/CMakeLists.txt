cmake_minimum_required(VERSION 3.16)
project(lvm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 CONFIG REQUIRED)
find_package(dlib CONFIG REQUIRED)

add_executable(lvm_app apps/main.cpp)
target_link_libraries(lvm_app PRIVATE Eigen3::Eigen dlib::dlib)


