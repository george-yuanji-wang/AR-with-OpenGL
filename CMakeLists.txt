cmake_minimum_required(VERSION 3.5)
project(ar_aruco_app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

add_library(aruco_core
    src/aruco_tracker.cpp
    src/obj_loader.cpp
)

target_include_directories(aruco_core PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(aruco_core
    ${OpenCV_LIBS}
    ${OPENGL_LIBRARIES}
    glfw
)

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} aruco_core)

add_executable(detect_test src/detect_test.cpp)
target_link_libraries(detect_test aruco_core)
