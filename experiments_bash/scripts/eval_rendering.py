#!/usr/bin/python

# Modified by Raul Mur-Artal
# Automatically compute the optimal scale factor for monocular VO/SLAM.

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Requirements: 
# sudo apt-get install python-argparse

"""
This script computes the absolute trajectory error from the ground truth
trajectory and the estimated trajectory.
"""

from math import exp
import sys
import argparse
import os
# import rerun as rr
import cv2
import numpy as np
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def mse(img1, img2):
    """Compute per-image mean squared error.

    Args:
        img1 (torch.Tensor): Predicted image tensor of shape ``[C, H, W]``.
        img2 (torch.Tensor): Ground-truth image tensor of shape ``[C, H, W]``.

    Returns:
        torch.Tensor: Scalar tensor containing the mean squared error.
    """
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    """Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

    Uses the formula ``20 * log10(1 / sqrt(MSE))`` where pixel values are in [0, 1].

    Args:
        img1 (torch.Tensor): Predicted image tensor of shape ``[C, H, W]``.
        img2 (torch.Tensor): Ground-truth image tensor of shape ``[C, H, W]``.

    Returns:
        torch.Tensor: Scalar tensor with the PSNR value in dB.
    """
    mse = ((img1 - img2) ** 2).contiguous().view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gaussian(window_size, sigma):
    """Create a 1D normalised Gaussian kernel.

    Args:
        window_size (int): Number of elements in the kernel.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        torch.Tensor: 1D Gaussian kernel of shape ``[window_size]``, summing to 1.
    """
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """Build a 2D separable Gaussian kernel for SSIM computation.

    Args:
        window_size (int): Spatial size of the kernel.
        channel (int): Number of image channels.

    Returns:
        torch.autograd.Variable: Kernel of shape ``[channel, 1, window_size, window_size]``.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    """Compute Structural Similarity Index (SSIM) between two images.

    Args:
        img1 (torch.Tensor): First image, shape ``[C, H, W]``, values in [0, 1].
        img2 (torch.Tensor): Second image, shape ``[C, H, W]``.
        window_size (int): Gaussian kernel window size (default 11).
        size_average (bool): If ``True``, return the mean SSIM over all pixels.

    Returns:
        torch.Tensor: Scalar SSIM value in [0, 1] (higher is better).
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """Internal SSIM computation given a pre-built kernel.

    Args:
        img1 (torch.Tensor): First image ``[C, H, W]``.
        img2 (torch.Tensor): Second image ``[C, H, W]``.
        window: Pre-built Gaussian kernel from :func:`create_window`.
        window_size (int): Spatial kernel size.
        channel (int): Number of image channels.
        size_average (bool): If ``True``, return the mean over all pixels.

    Returns:
        torch.Tensor: SSIM value or per-pixel SSIM map.
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def align(model, data):
    """Align two trajectories using the closed-form method of Horn (1987).

    Computes the optimal scale, rotation, and translation to align ``model`` to ``data``
    in the least-squares sense.

    Args:
        model (np.ndarray): First trajectory as a ``(3, N)`` array of XYZ positions.
        data (np.ndarray): Second (reference) trajectory as a ``(3, N)`` array.

    Returns:
        tuple: A 4-tuple ``(rot, trans, trans_error, scale)`` where:
            - ``rot`` is a ``(3, 3)`` rotation matrix aligning model to data.
            - ``trans`` is a ``(3, 1)`` translation vector.
            - ``trans_error`` is a ``(N,)`` array of per-point translation errors after alignment.
            - ``scale`` is the optimal uniform scale factor.
    """
    """Align two trajectories using the method of Horn (closed-form).
    
    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)
    
    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    
    """
    np.set_printoptions(precision=3,suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3, -1))
    data_zerocentered = data - data.mean(1).reshape((3, -1))
    
    W = np.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = np.linalg.svd(W.transpose())
    S = np.matrix(np.identity( 3 ))
    if(np.linalg.det(U) * np.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh

    rotmodel = rot*model_zerocentered
    dots = 0.0
    norms = 0.0

    for column in range(data_zerocentered.shape[1]):
        dots += np.dot(data_zerocentered[:,column].transpose(),rotmodel[:,column])
        normi = np.linalg.norm(model_zerocentered[:,column])
        norms += normi*normi

    s = float(dots/norms)    

    # print("scale: %f " % s  )
    
    trans = data.mean(1).reshape((3,-1)) - s*rot * model.mean(1).reshape((3,-1))
    
    model_aligned = s*rot * model + trans
    alignment_error = model_aligned - data
    
    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error),0)).A[0]
        
    return rot,trans,trans_error, s

def plot_traj(ax,stamps,traj,style,color,label):
    """
    Plot a trajectory using matplotlib. 
    
    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend
    
    """
    stamps.sort()
    interval = np.median([s-t for s,t in zip(stamps[1:],stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x)>0:
            ax.plot(x,y,style,color=color,label=label)
            label=""
            x=[]
            y=[]
        last= stamps[i]
    if len(x)>0:
        ax.plot(x,y,style,color=color,label=label)
            

if __name__=="__main__":
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory. 
    ''')
    parser.add_argument('gt_file', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('est_path', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('gt_path', help='est depth path')
    args = parser.parse_args()
    
    eval_depth = False
    
    # load gt pose
    gt_xyz_whole = []
    with open(args.gt_file, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        gt_xyz_whole.append([c2w[0,3], c2w[1,3], c2w[2,3]])
    gt_xyz_whole = np.array(gt_xyz_whole)  # [n,3]

    # load estimated traj
    est_idxs = []
    est_xyz = []
    est_pose_path = os.path.join(args.est_path, "pose")
    est_pose_file_list = sorted(os.listdir(est_pose_path))
    for est_pose_file in est_pose_file_list:
        est_c2w = np.loadtxt(os.path.join(est_pose_path, est_pose_file))
        est_xyz.append([est_c2w[0,3], est_c2w[1,3], est_c2w[2,3]])
        est_idxs.append((int)(est_pose_file.split('.')[0]))
    est_xyz = np.array(est_xyz)
    
    # associate
    gt_xyz = gt_xyz_whole[est_idxs]

    gt_xyz = gt_xyz.T
    est_xyz = est_xyz.T
    
    rot,trans,trans_error,scale = align(est_xyz,gt_xyz)
    
    est_xyz_aligned = scale * rot * est_xyz + trans
    
    print(f"ATE RMSE: {trans_error.mean()*100}")
    
    # plot
    # rr.init("Evaluation result", spawn=True)
    
    # ## GT
    # rr.log(
    #     "Ground_Truth_Traj",
    #     rr.Points3D(gt_xyz_whole)
    # )
    
    # ## Est
    # rr.log(
    #     "Estimate_Traj",
    #     rr.Points3D(est_xyz_aligned.T)
    # )
    
    gt_imgs_path = os.path.join(args.gt_path, "results")
    
    for item in os.listdir(args.est_path):
        item_path = os.path.join(args.est_path, item)
        
        if os.path.isdir(item_path) and item.endswith('_shutdown'):
            gs_result_path = item_path
    
    # eval_depth
    if eval_depth:
        ## estimated depths
        est_depth_path = os.path.join(gs_result_path, "depth")
        
        est_depth_files = sorted(os.listdir(est_depth_path))
        
        ## gt depths
        gt_depth_files = []
        for file in os.listdir(gt_imgs_path):
            if file.endswith('.png'):
                gt_depth_files.append(file)

        gt_depth_files = sorted(gt_depth_files)

        ## calculate depth L1 error
        depth_l1_erorrs = []
        for est_depth_file in est_depth_files:
            est_depth = cv2.imread(os.path.join(est_depth_path, est_depth_file), cv2.IMREAD_UNCHANGED)
            # est_depth = est_depth[:,:,0]
            est_depth = est_depth / 6553.5

            est_depth_scaled = est_depth * scale
            est_depth_mask = (est_depth > 0.)

            # est_depth_mask = est_depth > 0.
            # est_depth_scaled = np.clip(est_depth_scaled, 0., 10.)

            est_depth_file_ = est_depth_file.split(".")[0]
            est_depth_idx = int(est_depth_file_.split("_")[1])
            
            gt_depth_file = gt_depth_files[est_depth_idx]
                        
            gt_depth = cv2.imread(os.path.join(gt_imgs_path, gt_depth_file), cv2.IMREAD_UNCHANGED)
            gt_depth_mask = gt_depth > 0
            gt_depth = gt_depth / 6553.5
        
            # mask = np.bitwise_and(est_depth_mask, gt_depth_mask)
            mask = est_depth_mask & gt_depth_mask
            
            diff_depth_l1_ = np.abs(est_depth_scaled.astype(np.float32) - gt_depth.astype(np.float32))
            diff_depth_l1 = diff_depth_l1_ * mask
            l1_error = diff_depth_l1.sum() / mask.sum()
            depth_l1_erorrs.append(l1_error)
            # print(l1_error)

            # vis
            # diff_depth_l1_vis = diff_depth_l1_
            # # diff_depth_l1_vis = cv2.normalize(diff_depth_l1_vis, None, 0, 255, cv2.NORM_MINMAX)
            # diff_depth_l1_vis = diff_depth_l1_vis / 6.0 * 255
            # diff_depth_l1_vis = np.uint8(diff_depth_l1_vis)
            # diff_depth_l1_vis_colormap = cv2.applyColorMap(diff_depth_l1_vis, cv2.COLORMAP_JET)
            # cv2.imshow("depth_error", diff_depth_l1_vis)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            
            # est_depth_vis = cv2.normalize(est_depth_scaled, None, 0, 255, cv2.NORM_MINMAX)
            # gt_depth_vis = cv2.normalize(gt_depth, None, 0, 255, cv2.NORM_MINMAX)
            # est_depth_vis = est_depth_scaled / 10.0 * 255.
            # gt_depth_vis = gt_depth / 10.0 * 255.
            # est_depth_vis = np.uint8(est_depth_vis)
            # gt_depth_vis = np.uint8(gt_depth_vis)
            
            # est_colormap = cv2.applyColorMap(est_depth_vis, cv2.COLORMAP_JET)
            # gt_colormap = cv2.applyColorMap(gt_depth_vis, cv2.COLORMAP_JET)
            
            # cv2.imshow("est", est_depth_vis)
            # cv2.imshow('gt', gt_depth_vis)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
        print(f"Depth L1: {np.mean(depth_l1_erorrs) * 100.}")

    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    # eval rgb
    ## estimated rgbs
    est_rgb_path = os.path.join(gs_result_path, "image")
    est_rgb_files = sorted(os.listdir(est_rgb_path))
    
    gt_rgb_path = os.path.join(gs_result_path, "image_gt")
    gt_rgb_files = os.listdir(gt_rgb_path)
    gt_rgb_files = sorted(gt_rgb_files)
    
    psnr_array, ssim_array, lpips_array = [], [], []
    
    for est_rgb_file in est_rgb_files:
        est_img = cv2.imread(os.path.join(est_rgb_path, est_rgb_file))
        est_img = torch.from_numpy(est_img)
        est_img = est_img.to("cuda").float()
        est_img = est_img / 255.0
        est_img = est_img.permute(2, 0, 1)
        est_img = torch.clamp(est_img, 0.0, 1.0)
        
        idx = est_rgb_file.split('.')[0].split('_')[1]
        
        for file in gt_rgb_files:
            if file.split('.')[0].split('_')[1] == idx:
                gt_rgb_file = file
        
        gt_img = cv2.imread(os.path.join(gt_rgb_path, gt_rgb_file))
        gt_img = torch.from_numpy(gt_img)
        gt_img = gt_img.to("cuda").float()
        gt_img = gt_img / 255.0
        gt_img = gt_img.permute(2, 0, 1)
        
        mask = gt_img > 0
        
        # print(est_img.shape, gt_img.shape)
        
        psnr_score = psnr((est_img[mask]).unsqueeze(0), (gt_img[mask]).unsqueeze(0))
        ssim_score = ssim((est_img).unsqueeze(0), (gt_img).unsqueeze(0))
        lpips_score = cal_lpips((est_img).unsqueeze(0), (gt_img).unsqueeze(0))
        
        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())
        
        # print(gt_rgb_file, est_rgb_file)
        # print(psnr_score, ssim_score, lpips_score)
    print(f"PSNR: {np.mean(psnr_array)} / SSIM: {np.mean(ssim_array)} / LPIPS: {np.mean(lpips_array)}")
        
        
        