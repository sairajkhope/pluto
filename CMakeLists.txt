cmake_minimum_required(VERSION 3.15)
project(toddlerbot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (Debug or Release)" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# ==================== Include Directories ====================
include_directories(${CMAKE_SOURCE_DIR}/toddlerbot/actuation/include)
include_directories(${CMAKE_SOURCE_DIR}/toddlerbot/sim/include)

# ==================== Python Binding Module ====================
find_package(pybind11 REQUIRED)

# (Dynamixel C++ modules removed in favor of pure Python implementation)