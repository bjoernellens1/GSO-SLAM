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

#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <tuple>
#include <filesystem>

#include "types.h"
#include "camera.h"
#include "point3d.h"
#include "point2d.h"
#include "gaussian_parameters.h"
#include "gaussian_model.h"
#include "gaussian_keyframe.h"

/**
 * @brief Manages the keyframe database, cameras, and cached 3D point cloud.
 *
 * @c GaussianScene is the data store used by @c GaussianMapper.  It maintains:
 * - The set of @c GaussianKeyframe objects indexed by frame ID.
 * - The set of @c Camera intrinsic models indexed by camera ID.
 * - A cache of @c Point3D objects used for initial Gaussian seed generation.
 * - The scene normalisation radius (@c cameras_extent_) used for densification thresholds.
 */
class GaussianScene
{
public:
    /**
     * @brief Construct a GaussianScene, optionally loading a pre-trained model from disk.
     *
     * @param args              Model parameters (source path, SH degree, etc.).
     * @param load_iteration    If > 0, load the model saved at this iteration count.
     * @param shuffle           If true, shuffle the keyframe order for training.
     * @param resolution_scales Per-scale resolution multipliers for multi-scale training.
     */
    GaussianScene(
        GaussianModelParams& args,
        int load_iteration = 0,
        bool shuffle = true,
        std::vector<float> resolution_scales = {1.0f});

public:
    /// @brief Register a camera intrinsic model in the scene.
    void addCamera(Camera& camera);
    /// @brief Retrieve a camera model by its ID.
    Camera& getCamera(camera_id_t cameraId);

    /**
     * @brief Add a new keyframe to the scene and optionally update the shuffle order.
     * @param new_kf   Shared pointer to the keyframe to add.
     * @param shuffled Output flag set to @c true if the keyframe shuffle was regenerated.
     */
    void addKeyframe(std::shared_ptr<GaussianKeyframe> new_kf, bool* shuffled);
    /// @brief Retrieve a keyframe by its frame ID.
    std::shared_ptr<GaussianKeyframe> getKeyframe(std::size_t fid);
    /// @brief Returns a mutable reference to the keyframe map (fid → keyframe).
    std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>& keyframes();
    /// @brief Returns a copy of the full keyframe map.
    std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> getAllKeyframes();

    /**
     * @brief Cache a 3D map point (used during initial Gaussian seeding).
     * @param point3D_id  Global identifier for the point.
     * @param point3d     The @c Point3D data to cache.
     */
    void cachePoint3D(point3D_id_t point3D_id, Point3D& point3d);
    /// @brief Retrieve a cached 3D point by its ID.
    Point3D& getPoint3D(point3D_id_t point3DId);
    /// @brief Clear all cached 3D points (frees memory after seeding).
    void clearCachedPoint3D();

    /**
     * @brief Apply a scaled rigid transformation to all keyframe poses and 3D points.
     * @param s  Uniform scale factor.
     * @param T  SE3 rigid transformation applied after scaling.
     */
    void applyScaledTransformation(
        const float s = 1.0,
        const Sophus::SE3f T = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));

    /**
     * @brief Compute the NeRF++ normalisation: scene centre and radius.
     * @return Tuple of (scene centre as Eigen::Vector3f, normalisation radius as float).
     */
    std::tuple<Eigen::Vector3f, float> getNerfppNorm();

    /**
     * @brief Split keyframes into training and test sets.
     * @param test_ratio  Fraction of keyframes to allocate to the test set (e.g. 0.1).
     * @return Pair of (train map, test map).
     */
    std::tuple<std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>,
               std::map<std::size_t, std::shared_ptr<GaussianKeyframe>>>
        splitTrainAndTestKeyframes(const float test_ratio);

public:
    float cameras_extent_; ///< scene_info.nerf_normalization["radius"]

    int loaded_iter_;

    std::map<camera_id_t, Camera> cameras_;
    std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> keyframes_;
    std::map<point3D_id_t, Point3D> cached_point_cloud_;

protected:
    std::mutex mutex_kfs_;
};
