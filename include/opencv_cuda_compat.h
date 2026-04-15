#pragma once

#if defined(USE_ROCM)
#define GSO_HAS_OPENCV_CUDA 0
#elif defined(__has_include)
#if __has_include(<opencv2/cudaimgproc.hpp>) && __has_include(<opencv2/cudastereo.hpp>) && __has_include(<opencv2/cudawarping.hpp>)
#define GSO_HAS_OPENCV_CUDA 1
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#else
#define GSO_HAS_OPENCV_CUDA 0
#endif
#else
#define GSO_HAS_OPENCV_CUDA 0
#endif
