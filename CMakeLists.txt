cmake_minimum_required(VERSION 3.10)

project(yolov8)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

set(PROJECT_SOURCES
    main.cpp main.h
    yolo/YoloDet.cpp yolo/YoloDet.h
    yolo/YoloPose.cpp yolo/YoloPose.h
    tools/ImageTools.cpp tools/ImageTools.h
    tools/timeCheck.cpp tools/timeCheck.h
    lidar/lens_transform.cpp lidar/lens_transform.h
    lidar/NSL3130AA.cpp lidar/NSL3130AA.h
    lidar/videoSource.cpp lidar/videoSource.h
    lidar/NSLFrame.h
)

add_executable(yolov8 ${PROJECT_SOURCES})

target_include_directories(yolov8 PRIVATE ./ ./yolo ./tools ./lidar)
target_link_libraries(yolov8 PRIVATE ${OpenCV_LIBS} Threads::Threads)