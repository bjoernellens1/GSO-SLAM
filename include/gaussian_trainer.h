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

#include <torch/torch.h>

#include <iomanip>
#include <random>
#include <chrono>
#include <memory>
#include <thread>
#include <mutex>
#include <vector>
#include <unordered_map>

#include <opencv2/opencv.hpp>

// #include "ORB-SLAM3/include/System.h"
#include "src/FullSystem/FullSystem.h"

#include "loss_utils.h"
#include "gaussian_parameters.h"
#include "gaussian_model.h"
#include "gaussian_scene.h"
#include "gaussian_renderer.h"


/**
 * @brief Stateless trainer providing one iteration of the Gaussian Splatting training loop.
 *
 * The training loop:
 * 1. Sample a random keyframe from the scene.
 * 2. Render the scene from that keyframe's viewpoint.
 * 3. Compute the combined L1 + SSIM photometric loss.
 * 4. Back-propagate gradients and step the Adam optimiser.
 * 5. Periodically densify, prune, and reset opacity.
 */
class GaussianTrainer
{
public:
    GaussianTrainer();

    /**
     * @brief Execute one training epoch (all iterations) from scratch.
     *
     * This is used for offline / COLMAP-based training only.  For online SLAM,
     * @c GaussianMapper::trainForOneIteration() is used instead.
     *
     * @param scene                Scene containing training keyframes.
     * @param gaussians            Gaussian model to optimise.
     * @param dataset              Model parameters.
     * @param opt                  Optimisation parameters.
     * @param pipe                 Pipeline parameters.
     * @param device_type          Torch device.
     * @param testing_iterations   Iterations at which to run test rendering.
     * @param saving_iterations    Iterations at which to save a PLY checkpoint.
     * @param checkpoint_iterations  Iterations at which to save a training checkpoint.
     */
    static void trainingOnce(
        std::shared_ptr<GaussianScene> scene,
        std::shared_ptr<GaussianModel> gaussians,
        GaussianModelParams& dataset,
        GaussianOptimizationParams& opt,
        GaussianPipelineParams& pipe,
        torch::DeviceType device_type = torch::kCUDA,
        std::vector<int> testing_iterations = {},
        std::vector<int> saving_iterations = {},
        std::vector<int> checkpoint_iterations = {}/*, checkpoint*/);

    /**
     * @brief Print a training progress report at the specified iteration.
     *
     * @param iteration         Current training iteration.
     * @param num_iterations    Total number of iterations.
     * @param Ll1               L1 loss value.
     * @param loss              Combined loss value.
     * @param ema_loss_for_log  Exponential moving average of the loss (for display).
     * @param l1_loss           L1 loss function callable.
     * @param elapsed_time      Wall-clock time for this iteration in milliseconds.
     * @param gaussians         The Gaussian model (for Gaussian count reporting).
     * @param scene             The scene (for test rendering).
     * @param pipe              Pipeline parameters.
     * @param background        Background colour tensor.
     */
    static void trainingReport(
        int iteration,
        int num_iterations,
        torch::Tensor& Ll1,
        torch::Tensor& loss,
        float ema_loss_for_log,
        std::function<torch::Tensor(torch::Tensor&, torch::Tensor&)> l1_loss,
        int64_t elapsed_time,
        GaussianModel& gaussians,
        GaussianScene& scene,
        GaussianPipelineParams& pipe,
        torch::Tensor& background);

};
