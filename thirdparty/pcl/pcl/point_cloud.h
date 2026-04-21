#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace pcl {

template <typename PointT>
struct PointCloud {
    using PointType = PointT;
    using Ptr = std::shared_ptr<PointCloud<PointT>>;
    using ConstPtr = std::shared_ptr<const PointCloud<PointT>>;

    std::vector<PointT> points;
    std::size_t width = 0;
    std::size_t height = 0;
    bool is_dense = true;

    inline std::size_t size() const { return points.size(); }
    inline bool empty() const { return points.empty(); }
    inline void clear() {
        points.clear();
        width = 0;
        height = 0;
        is_dense = true;
    }
};

}  // namespace pcl
