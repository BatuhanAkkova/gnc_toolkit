cmake_minimum_required(VERSION 3.14)
project(GNCToolkit)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# --- Fetch Eigen3 automatically ---
FetchContent_Declare(
  Eigen3
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
)
FetchContent_MakeAvailable(Eigen3)

# Header paths
include_directories(include)

# KF Library
add_library(kf src/kf.cpp)
target_link_libraries(kf PUBLIC Eigen3::Eigen)

# EKF Library
add_library(ekf src/ekf.cpp)
target_link_libraries(ekf PUBLIC Eigen3::Eigen)

# KF Test
add_executable(test_kf tests/test_kf.cpp)
target_link_libraries(test_kf kf Eigen3::Eigen)

# EKF Test
add_executable(test_ekf tests/test_ekf.cpp)
target_link_libraries(test_ekf ekf Eigen3::Eigen)
