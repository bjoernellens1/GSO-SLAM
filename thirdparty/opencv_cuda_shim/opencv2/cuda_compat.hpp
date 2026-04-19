#pragma once

#include <opencv2/opencv.hpp>
#include <utility>

namespace cv {
namespace cuda {

template <typename... Args>
inline void resize(const GpuMat& src, GpuMat& dst, Args&&... args)
{
    cv::Mat out;
    cv::Mat src_host;
    src.download(src_host);
    cv::resize(src_host, out, std::forward<Args>(args)...);
    dst.upload(out);
}

template <typename... Args>
inline void cvtColor(const GpuMat& src, GpuMat& dst, Args&&... args)
{
    cv::Mat out;
    cv::Mat src_host;
    src.download(src_host);
    cv::cvtColor(src_host, out, std::forward<Args>(args)...);
    dst.upload(out);
}

template <typename... Args>
inline void reprojectImageTo3D(const GpuMat& src, GpuMat& dst, Args&&... args)
{
    cv::Mat out;
    cv::Mat src_host;
    src.download(src_host);
    cv::reprojectImageTo3D(src_host, out, std::forward<Args>(args)...);
    dst.upload(out);
}

class StereoSGM {
public:
    StereoSGM() = default;

    template <typename... Args>
    static cv::Ptr<StereoSGM> create(Args&&...)
    {
        return cv::Ptr<StereoSGM>(new StereoSGM());
    }

    void compute(const GpuMat& left, const GpuMat& right, GpuMat& disp) const
    {
        (void)right;
        cv::Mat out(left.rows, left.cols, CV_32FC1, cv::Scalar(0.0f));
        disp.upload(out);
    }

    int getMinDisparity() const { return min_disp_; }
    int getNumDisparities() const { return num_disp_; }

private:
    int min_disp_ = 0;
    int num_disp_ = 64;
};

}  // namespace cuda
}  // namespace cv
