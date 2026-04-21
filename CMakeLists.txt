SET(PROJECT_NAME DSO)

CMAKE_MINIMUM_REQUIRED(VERSION 3.22 FATAL_ERROR)
PROJECT(${PROJECT_NAME} LANGUAGES CXX CUDA C)
#set(CMAKE_VERBOSE_MAKEFILE ON)

option(GSO_ENABLE_GUI "Build Pangolin/ImGui viewer support" OFF)
option(GSO_ENABLE_RERUN "Fetch the rerun SDK" OFF)
set(GSO_CUDA_ARCHITECTURES "80" CACHE STRING "CUDA architectures to build for")

set(BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
#set(BUILD_TYPE RelWithDebInfo)
set(CAFFE2_USE_CUDNN on)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# required libraries
# set(PYTHON_EXECUTABLE "/usr/bin/python3")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")
find_package(SuiteParse REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Boost COMPONENTS system thread REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)
find_package(glm REQUIRED)
find_package(OpenCV REQUIRED)

# optional libraries
find_package(LibZip QUIET)
if (GSO_ENABLE_GUI)
    find_package(Pangolin 0.2 QUIET)
endif()

if (GSO_ENABLE_RERUN)
    include(FetchContent)
    FetchContent_Declare(rerun_sdk URL
        https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
    FetchContent_MakeAvailable(rerun_sdk)
endif()

# flags
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -O2 -g0 -DNDEBUG -march=native"
#   "${SSE_FLAGS} -O3 -g -std=c++0x -fno-omit-frame-pointer"
)

if (MSVC)
     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif (MSVC)

# Sources files
set(dso_SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystem.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptimize.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptPoint.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemDebugStuff.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemMarginalize.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/Residuals.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseTracker.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseInitializer.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/ImmaturePoint.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/HessianBlocks.cpp
  ${PROJECT_SOURCE_DIR}/src/FullSystem/PixelSelector2.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctional.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedTopHessian.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedSCHessian.cpp
  ${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctionalStructs.cpp
  ${PROJECT_SOURCE_DIR}/src/util/settings.cpp
  ${PROJECT_SOURCE_DIR}/src/util/Undistort.cpp
  ${PROJECT_SOURCE_DIR}/src/util/globalCalib.cpp
)


include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
  ${PROJECT_SOURCE_DIR}/thirdparty/sse2neon
  ${PROJECT_SOURCE_DIR}/thirdparty/opencv_cuda_shim
  ${PROJECT_SOURCE_DIR}/thirdparty/pcl
  ${EIGEN3_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  /home/jovyan/work/.gso-build-env/include/eigen3
  /opt/conda/include
  /opt/conda/targets/x86_64-linux/include
  ${PROJECT_SOURCE_DIR}
) 


# decide if we have pangolin
if (GSO_ENABLE_GUI AND Pangolin_FOUND)
	message("--- found PANGOLIN, compiling dso_pangolin library.")
	include_directories( ${Pangolin_INCLUDE_DIRS} ) 
	set(dso_pangolin_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/KeyFrameDisplay.cpp
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/PangolinDSOViewer.cpp)
	set(HAS_PANGOLIN 1)
elseif (GSO_ENABLE_GUI)
	message("--- could not find PANGOLIN, not compiling dso_pangolin library.")
	message("    this means there will be no 3D display / GUI available for dso_dataset.")
	set(dso_pangolin_SOURCE_FILES )
	set(HAS_PANGOLIN 0)
else ()
	message("--- GUI support disabled, skipping Pangolin viewer sources.")
	set(dso_pangolin_SOURCE_FILES )
	set(HAS_PANGOLIN 0)
endif ()

# decide if we have openCV
if (OpenCV_FOUND)
	message("--- found OpenCV, compiling dso_opencv library.")
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	if (GSO_ENABLE_GUI)
		set(dso_opencv_SOURCE_FILES 
		  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
		  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageRW_OpenCV.cpp)
	else ()
		set(dso_opencv_SOURCE_FILES 
		  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageDisplay_dummy.cpp
		  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageRW_OpenCV.cpp)
	endif ()
	set_source_files_properties(
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageRW_OpenCV.cpp
	  PROPERTIES COMPILE_FLAGS "-U_GLIBCXX_USE_CXX11_ABI -D_GLIBCXX_USE_CXX11_ABI=1")
	set(HAS_OPENCV 1)
else ()
	message("--- could not find OpenCV, not compiling dso_opencv library.")
	message("    this means there will be no image display, and image read / load functionality.")
	set(dso_opencv_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageDisplay_dummy.cpp
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageRW_dummy.cpp)
	set(HAS_OPENCV 0)
endif ()

# decide if we have ziplib.
if (LIBZIP_LIBRARY)
	message("--- found ziplib (${LIBZIP_VERSION}), compiling with zip capability.")
	add_definitions(-DHAS_ZIPLIB=1)
	include_directories( ${LIBZIP_INCLUDE_DIR_ZIP} ${LIBZIP_INCLUDE_DIR_ZIPCONF} ) 
else()
	message("--- not found ziplib (${LIBZIP_LIBRARY}), compiling without zip capability.")
	set(LIBZIP_LIBRARY "")
endif()

## 2DGS
# simple_knn
add_library(simple_knn SHARED
    thirdparty/simple-knn/simple_knn.cu
    thirdparty/simple-knn/simple_knn.h
    thirdparty/simple-knn/spatial.cu
    thirdparty/simple-knn/spatial.h)
# target_compile_features(simple_knn PUBLIC cxx_std_17)
target_link_libraries(simple_knn "${TORCH_LIBRARIES}")

# rasterizer
add_library(cuda_rasterizer SHARED
    include/operate_points.h
    src/GS/operate_points.cu
    include/rasterize_points.h
    src/GS/rasterize_points.cu
    include/stereo_vision.h
    src/GS/stereo_vision.cu
    cuda_rasterizer/auxiliary.h
    cuda_rasterizer/backward.cu
    cuda_rasterizer/backward.h
    cuda_rasterizer/config.h
    cuda_rasterizer/forward.cu
    cuda_rasterizer/forward.h
    cuda_rasterizer/operate_points.h
    cuda_rasterizer/rasterizer.h
    cuda_rasterizer/rasterizer_impl.cu
    cuda_rasterizer/rasterizer_impl.h
    cuda_rasterizer/stereo_vision.h)
	set_target_properties(cuda_rasterizer PROPERTIES CUDA_ARCHITECTURES "${GSO_CUDA_ARCHITECTURES}")
# target_compile_features(cuda_rasterizer PUBLIC cxx_std_17)
target_include_directories(cuda_rasterizer PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuda_rasterizer
    glm::glm
    ${TORCH_LIBRARIES}
    # Eigen3::Eigen
    )

if (GSO_ENABLE_GUI)
    find_package(OpenGL REQUIRED)
    ##################################################################################
    ##  Build the ImGui library to ${PROJECT_SOURCE_DIR}/lib
    ##################################################################################

    add_library(imgui SHARED
        viewer/imgui/imconfig.h
        viewer/imgui/imgui_demo.cpp
        viewer/imgui/imgui_draw.cpp
        viewer/imgui/imgui_impl_glfw.cpp
        viewer/imgui/imgui_impl_glfw.h
        viewer/imgui/imgui_impl_opengl3_loader.h
        viewer/imgui/imgui_impl_opengl3.cpp
        viewer/imgui/imgui_impl_opengl3.h
        viewer/imgui/imgui_internal.h
        viewer/imgui/imgui_tables.cpp
        viewer/imgui/imgui_widgets.cpp
        viewer/imgui/imgui.cpp
        viewer/imgui/imgui.h
        viewer/imgui/imstb_rectpack.h
        viewer/imgui/imstb_textedit.h
        viewer/imgui/imstb_truetype.h)
    target_link_libraries(imgui glfw OpenGL::GL)
endif()

##################################################################################
##  Build the gaussian mapper library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(gaussian_mapper SHARED
    thirdparty/opencv_cuda_shim/opencv_abi_shim.cpp
    thirdparty/tinyply/tinyply.h
    thirdparty/tinyply/tinyply.cpp
    include/gaussian_keyframe.h
    include/gaussian_model.h
    include/gaussian_parameters.h
    include/gaussian_rasterizer.h
    include/gaussian_renderer.h
    include/gaussian_scene.h
    include/gaussian_trainer.h
    include/gaussian_mapper.h
    include/general_utils.h
    include/graphics_utils.h
    include/loss_utils.h
    include/sh_utils.h
    include/tensor_utils.h
    include/camera.h
    include/point_cloud.h
    include/point2d.h
    include/point3d.h
    include/types.h
    src/GS/gaussian_keyframe.cpp
    src/GS/gaussian_model.cpp
    src/GS/gaussian_parameters.cpp
    src/GS/gaussian_rasterizer.cpp
    src/GS/gaussian_renderer.cpp
    src/GS/gaussian_scene.cpp
    src/GS/gaussian_trainer.cpp
    src/GS/gaussian_mapper.cpp)
set_source_files_properties(
    thirdparty/opencv_cuda_shim/opencv_abi_shim.cpp
    PROPERTIES COMPILE_FLAGS "-U_GLIBCXX_USE_CXX11_ABI -D_GLIBCXX_USE_CXX11_ABI=1")
target_include_directories(gaussian_mapper PRIVATE
    /opt/conda/include)
target_compile_options(gaussian_mapper PRIVATE
    -I/opt/conda/include)
target_link_libraries(gaussian_mapper
    # ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    ${OpenCV_LIBRARIES}
    ${TORCH_LIBRARIES}
    # Eigen3::Eigen
    simple_knn
    cuda_rasterizer
    png)

if (GSO_ENABLE_GUI)
    ##################################################################################
    ##  Build the viewer library to ${PROJECT_SOURCE_DIR}/lib
    ##################################################################################

    add_library(gaussian_viewer SHARED
        viewer/drawer_utils.h
        viewer/imgui_viewer.cpp
        viewer/imgui_viewer.h
        # viewer/map_drawer.cpp
        # viewer/map_drawer.h
        )
    target_link_libraries(gaussian_viewer
        gaussian_mapper
        imgui
        ${OpenCV_LIBRARIES}
        ${TORCH_LIBRARIES}
        glm::glm
        glfw
        OpenGL::GL)
endif()

# compile main library.
include_directories( ${CSPARSE_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR}) 
add_library(dso ${dso_SOURCE_FILES} ${dso_opencv_SOURCE_FILES} ${dso_pangolin_SOURCE_FILES})
target_link_libraries(dso ${TORCH_LIBRARIES} Threads::Threads)

#set_property( TARGET dso APPEND_STRING PROPERTY COMPILE_FLAGS -Wall )
#set_property(TARGET dso PROPERTY CXX_STANDARD 17)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin") # OSX
    set(BOOST_THREAD_LIBRARY boost_thread-mt)
else()
    set(BOOST_THREAD_LIBRARY boost_thread)
endif()

# build main executable
if (OpenCV_FOUND)
	message("--- compiling dso_dataset.")
	add_executable(dso_dataset ${PROJECT_SOURCE_DIR}/src/main_dso_pangolin.cpp )
    if (GSO_ENABLE_GUI AND Pangolin_FOUND)
        target_compile_definitions(dso_dataset PRIVATE GSO_ENABLE_GUI=1)
        target_link_libraries(dso_dataset PRIVATE dso boost_system cxsparse ${BOOST_THREAD_LIBRARY} ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OpenCV_LIBS}
                                ${TORCH_LIBRARIES} Threads::Threads gaussian_mapper gaussian_viewer)
    else()
        target_compile_definitions(dso_dataset PRIVATE GSO_ENABLE_GUI=0)
        target_link_libraries(dso_dataset PRIVATE dso boost_system cxsparse ${BOOST_THREAD_LIBRARY} ${LIBZIP_LIBRARY} ${OpenCV_LIBS}
                                ${TORCH_LIBRARIES} Threads::Threads gaussian_mapper)
    endif()
    if (GSO_ENABLE_RERUN)
        target_compile_definitions(dso_dataset PRIVATE GSO_ENABLE_RERUN=1)
        target_link_libraries(dso_dataset PRIVATE rerun_sdk)
    else()
        target_compile_definitions(dso_dataset PRIVATE GSO_ENABLE_RERUN=0)
    endif()

else()
	message("--- not building dso_dataset, since OpenCV was not found.")
endif()
