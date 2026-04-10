/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Photo-SLAM.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <memory>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>

#include "thirdparty/Sophus/sophus/se3.hpp"

#include "types.h"
#include "camera.h"
#include "point2d.h"
#include "general_utils.h"
#include "graphics_utils.h"
#include "tensor_utils.h"

/**
 * @brief A single training viewpoint in the Gaussian Splatting map.
 *
 * Stores the camera pose (SE3), intrinsics, RGB and depth image tensors,
 * and the pre-computed projection matrices required by the CUDA rasterizer.
 * Keyframes are created by @c GaussianMapper when DSO selects a new keyframe
 * and are stored in @c GaussianScene.
 */
class GaussianKeyframe
{
public:
    GaussianKeyframe() {}

    /**
     * @brief Construct a GaussianKeyframe with a frame ID and creation iteration.
     * @param fid           Unique frame identifier (matches DSO's frame ID).
     * @param creation_iter Training iteration at which this keyframe was created.
     */
    GaussianKeyframe(std::size_t fid, int creation_iter = 0)
        : fid_(fid), creation_iter_(creation_iter) {}

    /**
     * @brief Set the camera pose from a quaternion and translation.
     * @param qw,qx,qy,qz  Rotation quaternion components (w first).
     * @param tx,ty,tz      Translation vector components.
     */
    void setPose(
        const double qw,
        const double qx,
        const double qy,
        const double qz,
        const double tx,
        const double ty,
        const double tz);
    
    /**
     * @brief Set the camera pose from an Eigen quaternion and translation vector.
     * @param q  Unit quaternion representing rotation.
     * @param t  Translation vector (camera origin in world frame).
     */
    void setPose(
        const Eigen::Quaterniond& q,
        const Eigen::Vector3d& t);

    /// @brief Returns the camera-to-world pose as a double-precision SE3.
    Sophus::SE3d getPose();
    /// @brief Returns the camera-to-world pose as a single-precision SE3.
    Sophus::SE3f getPosef();

    /**
     * @brief Copy camera intrinsics from a @c Camera object into this keyframe.
     * @param camera  The camera model (focal lengths, principal point, image size).
     */
    void setCameraParams(const Camera& camera);

    /**
     * @brief Store the list of observed 2D keypoints in this frame.
     * @param points2D  Vector of (u, v) 2D image coordinates.
     */
    void setPoints2D(const std::vector<Eigen::Vector2d>& points2D);

    /**
     * @brief Associate a 3D map point ID with a specific 2D observation.
     * @param point2D_idx  Index into the @c points2D_ vector.
     * @param point3D_id   Global ID of the corresponding 3D point.
     */
    void setPoint3DIdxForPoint2D(
        const point2D_idx_t point2D_idx,
        const point3D_id_t point3D_id);

    /**
     * @brief Pre-compute world-view, projection, and full-projection transform tensors.
     *
     * Must be called after @c setPose() and @c setCameraParams() before passing
     * this keyframe to the rasterizer.
     */
    void computeTransformTensors();

    /**
     * @brief Compute the 4×4 world-to-view matrix used by the OpenGL / CUDA pipeline.
     * @param trans  Optional translation offset applied to the camera origin.
     * @param scale  Uniform scale applied to the scene.
     * @return 4×4 float matrix mapping world coordinates to camera space.
     */
    Eigen::Matrix4f getWorld2View2(
        const Eigen::Vector3f& trans = {0.0f, 0.0f, 0.0f},
        float scale = 1.0f);

    /**
     * @brief Compute the perspective projection matrix for this keyframe.
     * @param znear        Near clipping plane distance.
     * @param zfar         Far clipping plane distance.
     * @param fovX         Horizontal field of view in radians.
     * @param fovY         Vertical field of view in radians.
     * @param device_type  Torch device on which to allocate the result tensor.
     * @return 4×4 float tensor containing the projection matrix.
     */
    torch::Tensor getProjectionMatrix(
        float znear,
        float zfar,
        float fovX,
        float fovY,
        torch::DeviceType device_type = torch::kCUDA);

    /**
     * @brief Returns the current Gaussian pyramid training level for this keyframe.
     * @return Integer level index (0 = full resolution, higher = coarser).
     */
    int getCurrentGausPyramidLevel();

public:
    std::size_t fid_;
    int creation_iter_;
    int remaining_times_of_use_ = 0;

    bool set_camera_ = false;

    camera_id_t camera_id_;
    int camera_model_id_ = 0;

    std::string img_filename_;
    cv::Mat img_undist_, img_auxiliary_undist_;
    torch::Tensor original_image_; ///< image
    torch::Tensor original_depth_; ///< depth image
    torch::Tensor sparse_depth_; ///< depth image
    int image_width_;              ///< image
    int image_height_;             ///< image

    int num_gaus_pyramid_sub_levels_;
    std::vector<int> gaus_pyramid_times_of_use_;
    std::vector<std::size_t> gaus_pyramid_width_;            ///< gaus_pyramid image
    std::vector<std::size_t> gaus_pyramid_height_;           ///< gaus_pyramid image
    std::vector<torch::Tensor> gaus_pyramid_original_image_; ///< gaus_pyramid image
    // Tensor gt_alpha_mask_;

    std::vector<float> intr_; ///< intrinsics

    float FoVx_; ///< intrinsics
    float FoVy_; ///< intrinsics

    bool set_pose_ = false;
    bool set_projection_matrix_ = false;

    Eigen::Quaterniond R_quaternion_;  ///< extrinsics
    Eigen::Vector3d t_;                ///< extrinsics
    Sophus::SE3d Tcw_;                 ///< extrinsics

    torch::Tensor R_tensor_; ///< extrinsics
    torch::Tensor t_tensor_; ///< extrinsics

    float zfar_ = 100.0f;
    float znear_ = 0.01f;

    Eigen::Vector3f trans_ = {0.0f, 0.0f, 0.0f};
    float scale_ = 1.0f;

    torch::Tensor world_view_transform_;    ///< transform tensors
    torch::Tensor projection_matrix_;       ///< transform tensors
    torch::Tensor full_proj_transform_;     ///< transform tensors
    torch::Tensor camera_center_;           ///< transform tensors

    std::vector<Point2D> points2D_;
    std::vector<float> kps_pixel_;
    std::vector<float> kps_point_local_;

    bool done_inactive_geo_densify_ = false;
};
