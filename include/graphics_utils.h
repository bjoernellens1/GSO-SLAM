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

#include <Eigen/Geometry>

namespace graphics_utils
{

/**
 * @brief Rounds an integer up or down to the nearest multiple of 16.
 *
 * Used to align tile dimensions in the CUDA rasterizer (tiles are 16×16).
 *
 * @param integer Input integer.
 * @return Nearest multiple of 16 (rounds down for remainders < 8, up otherwise).
 */
inline int roundToIntegerMultipleOf16(int integer)
{
    int remainder = integer % 16;

    if (remainder == 0) {
        return integer;
    }
    else if (remainder < 8) {
        return integer - remainder;
    }
    else {
        return integer - remainder + 16;
    }

    return integer;
}

/**
 * @brief Converts a field-of-view angle to a focal length in pixels.
 *
 * @param fov   Field of view in radians.
 * @param pixels  Image dimension in pixels along the same axis.
 * @return Focal length in pixels.
 */
inline float fov2focal(float fov, int pixels)
{
    return pixels / (2.0f * std::tan(fov / 2.0f));
}

/**
 * @brief Converts a focal length in pixels to a field-of-view angle.
 *
 * @param focal   Focal length in pixels.
 * @param pixels  Image dimension in pixels along the same axis.
 * @return Field of view in radians.
 */
inline float focal2fov(float focal, int pixels)
{
    return 2.0f * std::atan(pixels / (2.0f * focal));
}

}
