cmake_minimum_required(VERSION 3.10)
SET(CMAKE_BUILD_TYPE "Debug")
PROJECT(RK_AI_VIDEO)
#设置目标处理器架构
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_SYSTEM_NAME Linux)
# 指定交叉编译器 arm-linux-gcc 和 arm-linux-g++
# set(CMAKE_C_COMPILER  /home/alientek/orangepi-build/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-c++)
# set(CMAKE_CXX_COMPILER/home/alientek/orangepi-build/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++)

# set(CMAKE_C_FLAGS "-pipe -O2 -Wall -W")
# set(CMAKE_CXX_FLAGS "-pipe -O2 -std=gnu++11 -Wall -W -D_REENTRANT -fPIC")

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/RK_AI_VIDEO/${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")


message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

# track
include_directories(/usr/include/eigen3)
include_directories(${PROJECT_SOURCE_DIR}/include)
message(${PROJECT_SOURCE_DIR}/include)

#opencv
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/opencv-4.9/install/lib/cmake/opencv4/)
# set(OpenCV_DIR /home/orangepi/ubuntu_desktop/rknn/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV/)
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
find_package(OpenCV REQUIRED CONFIG)

# message(${OpenCV_DIR})
# 多核
# find_package(OpenMP)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# set(SYS_INCLUDE /usr/local/include/)
# set(SYS_LIB /usr/local/lib/)


set(DRM_PATH ${CMAKE_SOURCE_DIR}/../3rdparty/libdrm-2.4.120/install/)

include_directories(${DRM_PATH}/include/libdrm)

include_directories(${OpenCV_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

#x264路径
set(X264_PATH ${CMAKE_SOURCE_DIR}/../3rdparty/x264/install)
include_directories(${X264_PATH}/include)

add_library(x264 SHARED IMPORTED )
#指定所添加依赖库的导入路径
set_target_properties(x264 PROPERTIES IMPORTED_LOCATION ${X264_PATH}/lib/libx264.so )


#FFMPEG路径
set(FFMPEG_RK_PATH ${CMAKE_SOURCE_DIR}/../3rdparty/ffmpeg/)
include_directories(${FFMPEG_RK_PATH}/include)
set(FFMPEG_RK_LIB ${FFMPEG_RK_PATH}/lib)

#对于find_package找不到的外部依赖库，可以用add_library添加
# SHARED表示添加的是动态库
# IMPORTED表示是引入已经存在的动态库
add_library( avcodec SHARED IMPORTED )
#指定所添加依赖库的导入路径

set_target_properties( avcodec PROPERTIES IMPORTED_LOCATION ${FFMPEG_RK_LIB}/libavcodec.so )
 
add_library( avfilter SHARED IMPORTED )
set_target_properties( avfilter PROPERTIES IMPORTED_LOCATION ${FFMPEG_RK_LIB}/libavfilter.so )
 
add_library( swresample SHARED IMPORTED )
set_target_properties( swresample PROPERTIES IMPORTED_LOCATION ${FFMPEG_RK_LIB}/libswresample.so )
 
add_library( swscale SHARED IMPORTED )
set_target_properties( swscale PROPERTIES IMPORTED_LOCATION ${FFMPEG_RK_LIB}/libswscale.so )
 
add_library( avformat SHARED IMPORTED )
set_target_properties( avformat PROPERTIES IMPORTED_LOCATION ${FFMPEG_RK_LIB}/libavformat.so )
 
add_library( avutil SHARED IMPORTED )
set_target_properties( avutil PROPERTIES IMPORTED_LOCATION ${FFMPEG_RK_LIB}/libavutil.so )



# mpp
set(MPP_PATH ${CMAKE_SOURCE_DIR}/../3rdparty/rkmpp/install)

set(MPP_LIBS ${MPP_PATH}/lib/librockchip_mpp.so)
# 规定.h头文件路径
include_directories(${MPP_PATH}/include)

# rga
set(RGA_PATH ${CMAKE_SOURCE_DIR}/../3rdparty/librga-rockchip/build/meson_aarch64/install)
set(RGA_LIB ${RGA_PATH}/lib/librga.so)
include_directories(${RGA_PATH}/include/rga)
# rknn_api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../rknn/runtime/RK356X/${CMAKE_SYSTEM_NAME}/librknn_api)
set(RKNN_RT_LIB ${RKNN_API_PATH}/aarch64/librknnrt.so)
include_directories(${RKNN_API_PATH}/include)


include_directories(${CMAKE_SOURCE_DIR}/include)

# 生成可执行文件
# update[2023/7/26] add mpp encoder and decode
add_executable(yolov5_rtsp_with_ffmpeg
  src/yolov5_rtsp_with_ffmpeg.cpp
  src/rknn/rknn_func.cc
  src/rknn/preprocess.cc
  src/rknn/postprocess.cc

  src/bytetrack/BYTETracker.cpp
  src/bytetrack/kalmanFilter.cpp
  src/bytetrack/lapjv.cpp
  src/bytetrack/STrack.cpp
  src/bytetrack/utils.cpp
)

# add_executable(rtsp_send_opencv_mpp ${PROJECT_SOURCE_DIR}/src/rtsp_with_mpp_using_opencv.cpp)

# add_executable(rtsp_send_opencv_mpp_rgb ${PROJECT_SOURCE_DIR}/src/rtsp_with_mpp_using_opencv_rgb.cpp)

set(CMAKE_INSTALL_RPATH "lib")

# 链接操作
target_link_libraries(yolov5_rtsp_with_ffmpeg
    ${RKNN_RT_LIB}
    ${OpenCV_LIBS}
    ${FFMPEG_LIBS}
    avformat avfilter avcodec avutil swresample swscale
    ${RGA_LIB}
    ${MPP_LIBS}
    x264
  )


# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rk_ffmpeg)
install(TARGETS yolov5_rtsp_with_ffmpeg DESTINATION ./)

install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(DIRECTORY model DESTINATION ./)
install(FILES config.conf DESTINATION ./)

if(MPP_LIBS)
  install(TARGETS yolov5_rtsp_with_ffmpeg DESTINATION ./)
  install(PROGRAMS ${MPP_LIBS} DESTINATION lib)
endif()
if(FFMPEG_LIBS)
  install(TARGETS yolov5_rtsp_with_ffmpeg DESTINATION ./)
  install(PROGRAMS ${FFMPEG_LIBS} DESTINATION lib)
endif()
# if(OpenCV_LIBS)
#   install(TARGETS yolov5_rtsp_with_ffmpeg DESTINATION ./)
#   install(PROGRAMS ${OpenCV_LIBS} DESTINATION lib)
# endif()

# if(ZLMEDIAKIT_LIBS)
#   install(PROGRAMS ${ZLMEDIAKIT_LIBS} DESTINATION lib)
# endif()
