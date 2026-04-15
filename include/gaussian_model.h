/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * This file is Derivative Works of Gaussian Splatting,
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023,
 * as part of Photo-SLAM.
 */

#pragma once

#include <memory>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>

#include <torch/torch.h>
#ifdef USE_ROCM
#include <c10/hip/HIPCachingAllocator.h>
#define GSO_GPU_CACHE_EMPTY() c10::hip::HIPCachingAllocator::emptyCache()
#else
#include <c10/cuda/CUDACachingAllocator.h>
#define GSO_GPU_CACHE_EMPTY() c10::cuda::CUDACachingAllocator::emptyCache()
#endif

#include "thirdparty/Sophus/sophus/se3.hpp"

#include "thirdparty/simple-knn/spatial.h"
#include "thirdparty/tinyply/tinyply.h"
#include "types.h"
#include "point3d.h"
#include "operate_points.h"
#include "general_utils.h"
#include "sh_utils.h"
#include "tensor_utils.h"
#include "gaussian_parameters.h"

#define GAUSSIAN_MODEL_TENSORS_TO_VEC                        \
    this->Tensor_vec_xyz_ = {this->xyz_};                    \
    this->Tensor_vec_feature_dc_ = {this->features_dc_};     \
    this->Tensor_vec_feature_rest_ = {this->features_rest_}; \
    this->Tensor_vec_opacity_ = {this->opacity_};            \
    this->Tensor_vec_scaling_ = {this->scaling_};            \
    this->Tensor_vec_rotation_ = {this->rotation_};

#define GAUSSIAN_MODEL_INIT_TENSORS(device_type)                                             \
    this->xyz_ = torch::empty(0, torch::TensorOptions().device(device_type));                \
    this->features_dc_ = torch::empty(0, torch::TensorOptions().device(device_type));        \
    this->features_rest_ = torch::empty(0, torch::TensorOptions().device(device_type));      \
    this->scaling_ = torch::empty(0, torch::TensorOptions().device(device_type));            \
    this->rotation_ = torch::empty(0, torch::TensorOptions().device(device_type));           \
    this->opacity_ = torch::empty(0, torch::TensorOptions().device(device_type));            \
    this->max_radii2D_ = torch::empty(0, torch::TensorOptions().device(device_type));        \
    this->xyz_gradient_accum_ = torch::empty(0, torch::TensorOptions().device(device_type)); \
    this->denom_ = torch::empty(0, torch::TensorOptions().device(device_type));              \
    GAUSSIAN_MODEL_TENSORS_TO_VEC

/**
 * @brief Stores and manages all per-Gaussian learnable parameters.
 *
 * Each 3D Gaussian is characterised by:
 *  - A 3D centre (@c xyz_)
 *  - Spherical harmonic colour coefficients (@c features_dc_, @c features_rest_)
 *  - A log-scale vector (@c scaling_) activated via @c exp
 *  - A unit quaternion (@c rotation_) activated via normalisation
 *  - A logit-opacity scalar (@c opacity_) activated via @c sigmoid
 *
 * Parameters are stored as @c torch::Tensor objects on the GPU and are
 * optimised by a @c torch::optim::Adam instance with per-parameter learning
 * rates and optional exponential decay for positions.
 *
 * Supports densification (clone / split), pruning, opacity reset,
 * PLY serialisation, and rigid scene transformation.
 */
class GaussianModel
{
public:
    /// @brief Construct from a maximum SH degree (allocates empty tensors).
    GaussianModel(const int sh_degree);
    /// @brief Construct from model parameters (reads SH degree and device from params).
    GaussianModel(const GaussianModelParams& model_params);

    /// @brief Returns activated scaling values (@f$ \exp(\text{scaling\_}) @f$).
    torch::Tensor getScalingActivation();
    /// @brief Returns normalised rotation quaternions.
    torch::Tensor getRotationActivation();
    /// @brief Returns raw Gaussian centre positions.
    torch::Tensor getXYZ();
    /// @brief Concatenates DC and rest SH features into a single @c [N, K, 3] tensor.
    torch::Tensor getFeatures();
    /// @brief Returns activated opacity values (@f$ \sigma(\text{opacity\_}) @f$).
    torch::Tensor getOpacityActivation();
    /**
     * @brief Computes the 3D covariance matrices for all Gaussians.
     * @param scaling_modifier  Uniform scale multiplier applied to all Gaussians.
     * @return Tensor of shape @c [N, 6] (upper-triangular 3×3 symmetric matrix).
     */
    torch::Tensor getCovarianceActivation(int scaling_modifier = 1);

    /// @brief Increment the active SH degree by one (up to @c max_sh_degree_).
    void oneUpShDegree();
    /// @brief Force-set the active SH degree.
    void setShDegree(const int sh);

    /**
     * @brief Initialise Gaussians from a point cloud (positions + colours).
     * @param points         Flat @c [N*3] array of XYZ positions.
     * @param colors         Flat @c [N*3] array of RGB colours in [0,1].
     * @param spatial_lr_scale  Scene extent used to set initial learning rate scale.
     */
    void createFromPcd(
        std::vector<float> points, 
        std::vector<float> colors,
        const float spatial_lr_scale);

    /**
     * @brief Initialise Gaussians from a point cloud with explicit scale and rotation.
     * @param points         Flat @c [N*3] array of XYZ positions.
     * @param colors         Flat @c [N*3] array of RGB colours.
     * @param scales_vec     Flat @c [N*3] array of log-scale values.
     * @param rots_vec       Flat @c [N*4] array of quaternion values (w,x,y,z).
     * @param spatial_lr_scale  Scene extent for LR scaling.
     */
    void createFromPcd(
        std::vector<float> points, 
        std::vector<float> colors,
        std::vector<float> scales_vec,
        std::vector<float> rots_vec,
        const float spatial_lr_scale);

    /**
     * @brief Initialise Gaussians from a COLMAP-style point cloud map.
     * @param pcd              Map from point3D_id to @c Point3D data.
     * @param spatial_lr_scale  Scene extent for LR scaling.
     */
    void createFromPcd(
        std::map<point3D_id_t, Point3D> pcd,
        const float spatial_lr_scale);

    /**
     * @brief Append new Gaussians to the model from a flat point array.
     * @param points    Flat @c [M*3] float array of new positions.
     * @param colors    Flat @c [M*3] float array of new RGB colours.
     * @param iteration Current training iteration (used for exist_since_iter_).
     */
    void increasePcd(std::vector<float> points, std::vector<float> colors, const int iteration);

    /// @overload Appends Gaussians with explicit scales and colours.
    void increasePcd(std::vector<float> points,     std::vector<float> colors, 
                     std::vector<float> scales_vec, std::vector<float> rots_vec, const int iteration);

    /// @overload Appends Gaussians from pre-built torch tensors.
    void increasePcd(torch::Tensor& new_point_cloud, torch::Tensor& new_colors, const int iteration);

    /**
     * @brief Apply a scaled rigid transformation to all Gaussian centres and scales.
     * @param s  Uniform scale factor.
     * @param T  SE3 transformation applied after scaling.
     */
    void applyScaledTransformation(
        const float s = 1.0,
        const Sophus::SE3f T = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));

    /// @brief Post-process new positions and scales after a transformation (used internally).
    void scaledTransformationPostfix(
        torch::Tensor& new_xyz,
        torch::Tensor& new_scaling);

    /**
     * @brief Apply a corrective transformation to Gaussians visible in a specific keyframe.
     *
     * Called on loop closure to warp Gaussian primitives whose centres project into a given
     * keyframe's frustum, correcting their positions to match the updated pose.
     *
     * @param point_not_transformed_flags  Boolean mask of Gaussians not yet transformed.
     * @param diff_pose                    Relative pose correction tensor.
     * @param kf_world_view_transform      World-to-view matrix of the keyframe.
     * @param kf_full_proj_transform       Full projection matrix of the keyframe.
     * @param kf_creation_iter             Training iteration at which the keyframe was created.
     * @param stable_num_iter_existence    Minimum iterations a Gaussian must exist to be eligible.
     * @param num_transformed              Output count of Gaussians actually transformed.
     * @param scale                        Uniform scale of the transformation.
     */
    void scaledTransformVisiblePointsOfKeyframe(
        torch::Tensor& point_not_transformed_flags,
        torch::Tensor& diff_pose,
        torch::Tensor& kf_world_view_transform,
        torch::Tensor& kf_full_proj_transform,
        const int kf_creation_iter,
        const int stable_num_iter_existence,
        int& num_transformed,
        const float scale = 1.0f);

    /**
     * @brief Initialise the Adam optimiser with per-parameter learning rates.
     * @param training_args  Optimisation parameters (learning rates, etc.).
     */
    void trainingSetup(const GaussianOptimizationParams& training_args);

    /**
     * @brief Compute and apply the exponential learning rate schedule for positions.
     * @param step  Current training iteration.
     * @return Current position learning rate.
     */
    float updateLearningRate(int step);

    /// @brief Override the position learning rate directly.
    void setPositionLearningRate(float position_lr);
    /// @brief Override the feature (SH) learning rate directly.
    void setFeatureLearningRate(float feature_lr);
    /// @brief Override the opacity learning rate directly.
    void setOpacityLearningRate(float opacity_lr);
    /// @brief Override the scaling learning rate directly.
    void setScalingLearningRate(float scaling_lr);
    /// @brief Override the rotation learning rate directly.
    void setRotationLearningRate(float rot_lr);

    /**
     * @brief Reset all opacities to a low value (used periodically to remove floaters).
     *
     * Replaces current opacity values with @c inverse_sigmoid(0.01).
     */
    void resetOpacity();

    /**
     * @brief Replace a parameter tensor in the Adam optimiser state.
     * @param t           New tensor value.
     * @param tensor_idx  Index of the parameter group in the optimiser.
     * @return A reference tensor connected to the optimiser state.
     */
    torch::Tensor replaceTensorToOptimizer(torch::Tensor& t, int tensor_idx);

    /**
     * @brief Remove Gaussians that satisfy the given boolean mask.
     * @param mask  Boolean tensor of shape @c [N]; @c true entries are pruned.
     */
    void prunePoints(torch::Tensor& mask);

    /**
     * @brief Concatenate newly created Gaussians into the model and optimiser state.
     *
     * Called after a densification step (clone or split) to append the new primitives.
     *
     * @param new_xyz            New positions tensor.
     * @param new_features_dc    New DC SH features.
     * @param new_features_rest  New higher-order SH features.
     * @param new_opacities      New logit-opacity values.
     * @param new_scaling        New log-scale values.
     * @param new_rotation       New quaternion values.
     * @param new_exist_since_iter  Iteration stamps for the new Gaussians.
     */
    void densificationPostfix(
        torch::Tensor& new_xyz,
        torch::Tensor& new_features_dc,
        torch::Tensor& new_features_rest,
        torch::Tensor& new_opacities,
        torch::Tensor& new_scaling,
        torch::Tensor& new_rotation,
        torch::Tensor& new_exist_since_iter);

    /**
     * @brief Densify by splitting Gaussians whose 2D gradient exceeds the threshold.
     * @param grads           Accumulated 2D position gradients, shape @c [N, 2].
     * @param grad_threshold  Minimum gradient magnitude to trigger a split.
     * @param scene_extent    Scene normalisation radius.
     * @param N               Number of child Gaussians to generate per split.
     */
    void densifyAndSplit(
        torch::Tensor& grads,
        float grad_threshold,
        float scene_extent,
        int N = 2);

    /**
     * @brief Densify by cloning under-reconstructed Gaussians.
     * @param grads           Accumulated 2D position gradients.
     * @param grad_threshold  Minimum gradient magnitude to trigger a clone.
     * @param scene_extent    Scene normalisation radius.
     */
    void densifyAndClone(
        torch::Tensor& grads,
        float grad_threshold,
        float scene_extent);

    /**
     * @brief Combined densify-and-prune step called periodically during training.
     * @param max_grad       Gradient threshold for densification.
     * @param min_opacity    Opacity threshold below which Gaussians are pruned.
     * @param extent         Scene normalisation radius.
     * @param max_screen_size  Maximum 2D radius in pixels; larger Gaussians are split.
     */
    void densifyAndPrune(
        float max_grad,
        float min_opacity,
        float extent,
        int max_screen_size);

    /**
     * @brief Accumulate 2D gradient statistics for densification decisions.
     * @param viewspace_point_tensor  2D means tensor with computed gradients.
     * @param update_filter           Boolean mask of Gaussians to include in statistics.
     */
    void addDensificationStats(
        torch::Tensor& viewspace_point_tensor,
        torch::Tensor& update_filter);

    /**
     * @brief Load Gaussian parameters from a PLY file.
     * @param ply_path  Path to the input @c .ply file.
     */
    void loadPly(std::filesystem::path ply_path);

    /**
     * @brief Save all Gaussian parameters to a PLY file.
     * @param result_path  Output path for the @c .ply file.
     */
    void savePly(std::filesystem::path result_path);

    /**
     * @brief Save only the sparse seed point cloud (before densification) to PLY.
     * @param result_path  Output path for the sparse @c .ply file.
     */
    void saveSparsePointsPly(std::filesystem::path result_path);

    /// @brief Returns the current percent-dense threshold.
    float percentDense();
    /// @brief Sets the percent-dense densification threshold.
    void setPercentDense(const float percent_dense);

protected:
    float exponLrFunc(int step);

public:
    torch::DeviceType device_type_;

    int active_sh_degree_;
    int max_sh_degree_;

    torch::Tensor xyz_;
    torch::Tensor features_dc_;
    torch::Tensor features_rest_;
    torch::Tensor scaling_;
    torch::Tensor rotation_;
    torch::Tensor opacity_;
    torch::Tensor max_radii2D_;
    torch::Tensor xyz_gradient_accum_;
    torch::Tensor denom_;
    torch::Tensor exist_since_iter_;

    std::vector<torch::Tensor> Tensor_vec_xyz_,
                               Tensor_vec_feature_dc_,
                               Tensor_vec_feature_rest_,
                               Tensor_vec_opacity_,
                               Tensor_vec_scaling_ ,
                               Tensor_vec_rotation_;

    std::shared_ptr<torch::optim::Adam> optimizer_;
    float percent_dense_;
    float spatial_lr_scale_;

    torch::Tensor sparse_points_xyz_;
    torch::Tensor sparse_points_color_;

protected:
    float lr_init_;
    float lr_final_;
    int lr_delay_steps_;
    float lr_delay_mult_;
    int max_steps_;

    std::mutex mutex_settings_;
};
