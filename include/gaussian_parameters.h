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

#include <string>
#include <filesystem>

/**
 * @brief Model configuration parameters for the Gaussian Splatting representation.
 *
 * Controls dataset paths, spherical-harmonic degree, image resolution,
 * background colour, and the compute device used for tensor storage.
 */
class GaussianModelParams
{
public:
    /**
     * @brief Construct a GaussianModelParams with optional defaults.
     *
     * @param source_path  Path to the input dataset or COLMAP sparse reconstruction.
     * @param model_path   Directory where the trained model (PLY) is saved.
     * @param exec_path    Path to the executable (informational only).
     * @param sh_degree    Maximum degree of spherical harmonics used (0–3).
     * @param images       Sub-directory name containing input images (default: "images").
     * @param resolution   Down-scaling factor for input images; -1 means native resolution.
     * @param white_background  If true, use a white background for rendering.
     * @param data_device  Device string for tensor storage, e.g. @c "cuda" or @c "cpu".
     * @param eval         If true, run in evaluation mode (no training).
     */
    GaussianModelParams(
        std::filesystem::path source_path = "",
        std::filesystem::path model_path = "",
        std::filesystem::path exec_path = "",
        int sh_degree = 3,
        std::string images = "images",
        float resolution = -1.0f,
        bool white_background = false,
        std::string data_device = "cuda",
        bool eval = false);

public:
    int sh_degree_;
    std::filesystem::path source_path_;
    std::filesystem::path model_path_;
    std::string images_;
    float resolution_;
    bool white_background_;
    std::string data_device_;
    bool eval_;
};

/**
 * @brief Pipeline configuration controlling pre-computation of intermediate quantities.
 *
 * These flags determine whether spherical-harmonic colours and 3-D covariance matrices
 * are pre-computed on the CPU before the CUDA forward pass, or computed inline on the GPU.
 */
class GaussianPipelineParams
{
public:
    /**
     * @brief Construct a GaussianPipelineParams.
     *
     * @param convert_SHs   If true, evaluate SH coefficients to RGB on the CPU before
     *                      passing them to the CUDA rasterizer.
     * @param compute_cov3D If true, compute 3-D covariance matrices on the CPU.
     * @param depth_ratio   Blending ratio between median and expected depth; 1.0 = median only.
     */
    GaussianPipelineParams(
        bool convert_SHs = false,
        bool compute_cov3D = false,
        float depth_ratio = 1.0f);

public:
    bool convert_SHs_;
    bool compute_cov3D_;
    float depth_ratio_;
};

/**
 * @brief Optimisation hyper-parameters for training the Gaussian Splatting model.
 *
 * Controls learning rates for each Gaussian parameter group, densification schedule,
 * and the SSIM/L1 loss blend coefficient.
 */
class GaussianOptimizationParams
{
public:
    /**
     * @brief Construct a GaussianOptimizationParams with optional defaults.
     *
     * @param iterations              Total number of training iterations.
     * @param position_lr_init        Initial learning rate for Gaussian positions.
     * @param position_lr_final       Final learning rate for positions (after exponential decay).
     * @param position_lr_delay_mult  Delay multiplier for the position learning rate schedule.
     * @param position_lr_max_steps   Number of steps over which position LR decays.
     * @param feature_lr              Learning rate for spherical-harmonic feature coefficients.
     * @param opacity_lr              Learning rate for logit-opacity values.
     * @param scaling_lr              Learning rate for log-scale values.
     * @param rotation_lr             Learning rate for quaternion rotation values.
     * @param percent_dense           Fraction of scene extent used as the densification threshold.
     * @param lambda_dssim            Weight for the SSIM loss term (1 - lambda_dssim weights L1).
     * @param densification_interval  Number of iterations between densification steps.
     * @param opacity_reset_interval  Number of iterations between opacity resets (0 = never).
     * @param densify_from_iter       Iteration at which densification begins.
     * @param densify_until_iter      Iteration at which densification ends.
     * @param densify_grad_threshold  2D position gradient threshold for cloning/splitting.
     */
    GaussianOptimizationParams(
        int iterations = 30'000,
        float position_lr_init = 0.00016f,
        float position_lr_final = 0.0000016f,
        float position_lr_delay_mult = 0.01f,
        int position_lr_max_steps = 30'000,
        float feature_lr = 0.0025f,
        float opacity_lr = 0.05f,
        float scaling_lr = 0.005f,
        float rotation_lr = 0.001f,
        float percent_dense = 0.01f,
        float lambda_dssim = 0.2f,
        int densification_interval = 100,
        int opacity_reset_interval = 3000,
        int densify_from_iter = 500,
        int densify_until_iter = 15'000,
        float densify_grad_threshold = 0.0002f);

public:
    int iterations_;
    float position_lr_init_;
    float position_lr_final_;
    float position_lr_delay_mult_;
    int position_lr_max_steps_;
    float feature_lr_;
    float opacity_lr_;
    float scaling_lr_;
    float rotation_lr_;
    float percent_dense_;
    float lambda_dssim_;
    int densification_interval_;
    int opacity_reset_interval_;
    int densify_from_iter_;
    int densify_until_iter_;
    float densify_grad_threshold_;
};
