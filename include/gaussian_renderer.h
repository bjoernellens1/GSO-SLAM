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

#include <tuple>

#include <torch/torch.h>

#include "sh_utils.h"

#include "gaussian_parameters.h"
#include "gaussian_keyframe.h"
#include "gaussian_model.h"
#include "gaussian_rasterizer.h"

/**
 * @brief Static renderer that produces colour, depth, and auxiliary outputs for a single viewpoint.
 *
 * Assembles camera intrinsics and projection matrices from a @c GaussianKeyframe,
 * then calls the @c GaussianRasterizer to produce:
 *  - Rendered RGB image
 *  - Rendered depth map
 *  - Alpha / transmittance tensors
 *  - Viewspace 2D means (used for densification gradient accumulation)
 *  - Visibility radius per Gaussian
 */
class GaussianRenderer
{
public:
    /**
     * @brief Render a single view of the Gaussian scene.
     *
     * @param viewpoint_camera  The keyframe defining camera pose and intrinsics.
     * @param image_height      Output image height in pixels.
     * @param image_width       Output image width in pixels.
     * @param gaussians         The Gaussian model to render.
     * @param pipe              Pipeline parameters (pre-computation flags).
     * @param bg_color          Background colour tensor (shape @c [3]).
     * @param override_color    Per-Gaussian colour override tensor (empty = use SH colours).
     * @param scaling_modifier  Uniform scale multiplier for all Gaussians.
     * @param has_override_color  Whether @p override_color should be used.
     *
     * @return 10-tuple of tensors:
     *         (rendered_image, depth, alpha, rendered_dist,
     *          viewspace_points, visibility_filter, radii,
     *          transmittance, rendered_depth2, rendered_opacity)
     */
    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> render(
        std::shared_ptr<GaussianKeyframe> viewpoint_camera,
        int image_height,
        int image_width,
        std::shared_ptr<GaussianModel> gaussians,
        GaussianPipelineParams& pipe,
        torch::Tensor& bg_color,
        torch::Tensor& override_color,
        float scaling_modifier = 1.0f,
        bool has_override_color = false);
};
