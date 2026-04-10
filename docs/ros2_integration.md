# ROS2 Integration

This guide explains how to wrap GSO-SLAM as a ROS2 (Humble / Iron) node that subscribes to standard camera topics and publishes pose, map, and rendered image outputs.

**Effort estimate:** 2–3 weeks  
**ROS2 version tested:** Humble (Ubuntu 22.04)

---

## Architecture

```
sensor_msgs/Image (camera/color/image_raw)
        │
        ▼
┌─────────────────────────┐
│  gso_slam_node.cpp      │
│  (ROS2 Node)            │
│                         │
│  ┌──────────────────┐   │    geometry_msgs/PoseStamped
│  │  dso::FullSystem │───┼──► /gso_slam/pose
│  └──────────────────┘   │
│                         │    sensor_msgs/PointCloud2
│  ┌──────────────────┐   │──► /gso_slam/gaussian_cloud
│  │  GaussianMapper  │   │
│  └──────────────────┘   │    sensor_msgs/Image
│                         │──► /gso_slam/rendered_rgb
└─────────────────────────┘
```

---

## Package Structure

```
gso_slam_ros2/
├── CMakeLists.txt
├── package.xml
├── src/
│   └── gso_slam_node.cpp       # main ROS2 node
├── include/
│   └── gso_slam_ros2/
│       ├── image_converter.hpp # sensor_msgs/Image → dso::ImageAndExposure
│       └── pose_converter.hpp  # dso::FrameShell   → geometry_msgs/PoseStamped
├── launch/
│   └── gso_slam.launch.py
└── config/
    ├── realsense_d435.yaml     # example sensor config (maps to GSO-SLAM YAML)
    └── office0.yaml            # Replica office0 config
```

---

## CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.22)
project(gso_slam_ros2 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)

# GSO-SLAM must be built and installed first, or use ExternalProject
find_package(gso_slam REQUIRED)  # provides gaussian_mapper, dso targets

add_executable(gso_slam_node src/gso_slam_node.cpp)
target_link_libraries(gso_slam_node
    gso_slam::gaussian_mapper
    gso_slam::dso
    ${PCL_LIBRARIES}
)
ament_target_dependencies(gso_slam_node
    rclcpp sensor_msgs geometry_msgs nav_msgs
    cv_bridge image_transport pcl_conversions
)

install(TARGETS gso_slam_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
```

---

## Main Node Implementation

```cpp
// src/gso_slam_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include "include/gaussian_mapper.h"
#include "src/FullSystem/FullSystem.h"
#include "src/util/Undistort.h"

class GsoSlamNode : public rclcpp::Node {
public:
    GsoSlamNode() : Node("gso_slam") {
        // Declare parameters (maps to GSO-SLAM YAML config keys)
        declare_parameter("config_path", "");
        declare_parameter("result_dir",  "/tmp/gso_slam_result");
        declare_parameter("calib_path",  "");
        declare_parameter("use_viewer",  false);

        const auto config  = get_parameter("config_path").as_string();
        const auto result  = get_parameter("result_dir").as_string();
        const auto calib   = get_parameter("calib_path").as_string();
        const bool viewer  = get_parameter("use_viewer").as_bool();

        // Initialise DSO
        // (load calibration, create FullSystem — same as main_dso_pangolin.cpp)
        undistort_ = dso::Undistort::getUndistorterForFile(calib, "", "");
        dso::setGlobalCalib(
            undistort_->getSize()[0], undistort_->getSize()[1],
            undistort_->getK().cast<float>());
        full_system_ = std::make_shared<dso::FullSystem>();

        // Initialise GaussianMapper
        mapper_ = std::make_shared<GaussianMapper>(
            full_system_, config, result, /*seed=*/0, torch::kCUDA);

        // ROS2 subscribers and publishers
        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "camera/color/image_raw", 10,
            std::bind(&GsoSlamNode::imageCallback, this, std::placeholders::_1));

        pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
            "gso_slam/pose", 10);
        cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            "gso_slam/gaussian_cloud", 1);
        rendered_pub_ = create_publisher<sensor_msgs::msg::Image>(
            "gso_slam/rendered_rgb", 1);

        // Start mapper thread
        mapper_thread_ = std::thread([this]() { mapper_->run(); });

        // Periodic map publisher (1 Hz)
        map_timer_ = create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&GsoSlamNode::publishMap, this));

        RCLCPP_INFO(get_logger(), "GSO-SLAM node started");
    }

    ~GsoSlamNode() {
        mapper_->signalStop();
        if (mapper_thread_.joinable()) mapper_thread_.join();
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        // Convert ROS image to DSO MinimalImage
        auto cv_img = cv_bridge::toCvShare(msg, "mono8");
        dso::MinimalImageB min_img(
            cv_img->image.cols, cv_img->image.rows,
            (unsigned char*)cv_img->image.data);

        auto img_and_exp = undistort_->undistort<unsigned char>(
            &min_img, /*exposure=*/1.0, /*timestamp=*/msg->header.stamp.sec);

        full_system_->addActiveFrame(img_and_exp, frame_id_++);

        // Publish latest rendered view at reduced frequency
        if (frame_id_ % 5 == 0) publishRenderedView();
    }

    void publishRenderedView() {
        // Get latest DSO pose
        auto shell = full_system_->getLatestFrameShell();
        if (!shell) return;
        Sophus::SE3f Tcw = shell->camToWorld.inverse().cast<float>();

        cv::Mat rendered = mapper_->renderFromPose(
            Tcw,
            undistort_->getSize()[0],
            undistort_->getSize()[1]);

        if (rendered.empty()) return;
        auto ros_img = cv_bridge::CvImage(
            std_msgs::msg::Header(), "bgr8", rendered).toImageMsg();
        rendered_pub_->publish(*ros_img);
    }

    void publishMap() {
        auto xyz = mapper_->gaussians_->getXYZ().cpu().contiguous();
        if (xyz.size(0) == 0) return;

        // Convert to PointCloud2
        pcl::PointCloud<pcl::PointXYZ> cloud;
        cloud.resize(xyz.size(0));
        auto* data = xyz.data_ptr<float>();
        for (int i = 0; i < xyz.size(0); ++i) {
            cloud[i].x = data[i * 3 + 0];
            cloud[i].y = data[i * 3 + 1];
            cloud[i].z = data[i * 3 + 2];
        }
        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(cloud, msg);
        msg.header.stamp    = now();
        msg.header.frame_id = "map";
        cloud_pub_->publish(msg);
    }

    // DSO + Mapper
    std::unique_ptr<dso::Undistort>      undistort_;
    std::shared_ptr<dso::FullSystem>     full_system_;
    std::shared_ptr<GaussianMapper>      mapper_;
    std::thread                          mapper_thread_;

    // ROS2
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr   cloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr         rendered_pub_;
    rclcpp::TimerBase::SharedPtr                                  map_timer_;

    int frame_id_ = 0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GsoSlamNode>());
    rclcpp::shutdown();
    return 0;
}
```

---

## Pose Publishing via Output3DWrapper

For more complete pose publishing (including all keyframes), implement the `dso::IOWrap::Output3DWrapper` interface:

```cpp
class ROS2PosePublisher : public dso::IOWrap::Output3DWrapper {
public:
    ROS2PosePublisher(rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub)
        : pub_(pub) {}

    void publishCamPose(dso::FrameShell* frame,
                        dso::CalibHessian* HCalib) override {
        Sophus::SE3f pose = frame->camToWorld.cast<float>();
        geometry_msgs::msg::PoseStamped msg;
        msg.header.stamp    = rclcpp::Clock().now();
        msg.header.frame_id = "map";
        // Convert Sophus::SE3f to geometry_msgs::Pose
        auto t = pose.translation();
        auto q = pose.unit_quaternion();
        msg.pose.position.x    = t.x(); msg.pose.position.y = t.y(); msg.pose.position.z = t.z();
        msg.pose.orientation.w = q.w(); msg.pose.orientation.x = q.x();
        msg.pose.orientation.y = q.y(); msg.pose.orientation.z = q.z();
        pub_->publish(msg);
    }

private:
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_;
};
```

Register it before starting DSO:

```cpp
full_system_->outputWrapper.push_back(
    new ROS2PosePublisher(pose_pub_));
```

---

## Launch File

```python
# launch/gso_slam.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package    = 'gso_slam_ros2',
            executable = 'gso_slam_node',
            name       = 'gso_slam',
            parameters = [{
                'config_path': '/path/to/cfg/gaussian_mapper/Monocular/Replica/office0.yaml',
                'calib_path':  '/path/to/dataset/camera.txt',
                'result_dir':  '/tmp/gso_result',
                'use_viewer':  False,
            }],
            remappings = [
                ('/camera/color/image_raw', '/realsense/color/image_raw'),
            ],
        )
    ])
```

---

## Threading Model

| Thread | Runs | Notes |
|---|---|---|
| ROS2 executor (main) | `rclcpp::spin()`, image callbacks | Must not block |
| Mapper thread | `GaussianMapper::run()` | Continuous training loop |
| ImGuiViewer (optional) | OpenGL rendering | Must be on main thread; incompatible with ROS2 spin on the same thread — use separate process or disable viewer |

**Recommended for production:** Disable the ImGui viewer (`use_viewer=false`) and use RViz2 to visualise the published `PointCloud2` and pose topics instead.

---

## Caveats

| Issue | Solution |
|---|---|
| DSO global state (`setting_*` vars) | Only one `dso::FullSystem` per process; use a standalone node |
| ROS2 time vs. DSO timestamps | Map `msg->header.stamp` to DSO's float timestamp; use `rclcpp::Time::seconds()` |
| Thread safety of `GaussianMapper` | The mapper uses internal mutexes; all public methods are safe to call from the image callback thread |
| GPU memory pressure | Reduce `GaussianViewer.image_scale` and `densify_until_iter` in the YAML config for real-time operation |
