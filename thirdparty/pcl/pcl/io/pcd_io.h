#pragma once

#include <string>

namespace pcl {
namespace io {

template <typename PointT>
inline int savePCDFile(const std::string&, const PointT&) {
    return 0;
}

template <typename PointT>
inline int savePCDFileASCII(const std::string&, const PointT&) {
    return 0;
}

template <typename PointT>
inline int savePCDFileBinary(const std::string&, const PointT&) {
    return 0;
}

}  // namespace io
}  // namespace pcl
