# project
cmake_minimum_required (VERSION 2.6)
project (libelas)

set(CMAKE_CXX_STANDARD 14)

# We use OpenCV for handling various image formats.
find_package(OpenCV REQUIRED)
# We use zlib for compressing parts of the kintinuous logs.
find_package(ZLIB REQUIRED)

# directories
set (LIBELAS_SRC_DIR src)

# include directory
include_directories("${LIBELAS_SRC_DIR}")
INCLUDE_DIRECTORIES(${ZLIB_INCLUDE_DIR})

# use sse3 instruction set
# Also enable c++14 support for clean filesystem interactions without requiring
# boost or other dependencies.
SET(CMAKE_CXX_FLAGS "-msse3 -O3 -std=c++14 -lstdc++fs")

# sources
FILE(GLOB LIBELAS_SRC_FILES "src/*.cpp")
# Ensure we exclude the demo/utility entry points.
list(REMOVE_ITEM LIBELAS_SRC_FILES "${CMAKE_SOURCE_DIR}/src/main.cpp")
list(REMOVE_ITEM LIBELAS_SRC_FILES "${CMAKE_SOURCE_DIR}/src/kitti2klg.cpp")


# make release version
set(CMAKE_BUILD_TYPE Release)

# build demo program
SET(LIBELAS_DEMO_SRC_FILES ${LIBELAS_SRC_FILES})
list(APPEND LIBELAS_DEMO_SRC_FILES "src/main.cpp")
add_executable(elas ${LIBELAS_DEMO_SRC_FILES})

SET(KITTI2KLG_SRC_FILES ${LIBELAS_SRC_FILES})
list(APPEND KITTI2KLG_SRC_FILES "src/kitti2klg.cpp")
add_executable(kitti2klg ${KITTI2KLG_SRC_FILES})

# Ensure we also link OpenCV, zlib, as well as the modern C++ filesystem library.
target_link_libraries(kitti2klg libstdc++fs.a ${ZLIB_LIBRARY} ${OpenCV_LIBS})

