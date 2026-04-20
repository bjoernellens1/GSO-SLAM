#include <opencv2/core.hpp>

namespace cv {
void error_forward(
    int status,
    const std::string& err,
    const char* func,
    const char* file,
    int line) __asm__("_ZN2cv5errorEiRKSsPKcS3_i");

void error_forward(
    int status,
    const std::string& err,
    const char* func,
    const char* file,
    int line)
{
    cv::error(status, err, func, file, line);
}
} // namespace cv
