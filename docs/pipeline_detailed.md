# Pipeline Overview

This repository runs a two-stage pipeline:

1. `FullSystem` performs direct visual odometry and decides when a frame becomes a keyframe.
2. `GaussianMapper` consumes those keyframes and builds / refines the Gaussian scene.

The code is intentionally asymmetric. DSO is the tracker. The Gaussian mapper is the reconstruction back-end.

## 1. Input Data

The runtime entry point is `src/main_dso_pangolin.cpp`.

The main inputs are:

- image sequence path from `files=...`
- calibration path from `calib=...`
- optional gamma and vignette calibration
- TUM association file from `dataassociation=...` when the dataset is RGB-D
- scene config from `cfg_yaml=...`

Dataset loading lives in `src/util/DatasetReader.h`.

Behavior by dataset family:

- TUM RGB-D: the loader accepts either the raw `rgb.txt` list used by the existing scripts or a 4-column rgb/depth association file. Only the 4-column form populates depth paths.
- Replica / ScanNet: the image directory is scanned and depth images are split out from the same folder layout.
- Monocular TUM: the loader falls back to reading the RGB directory only.

The code now treats RGB-D support explicitly instead of guessing it from the config path.

## 2. Conversion Path

The loaded image is converted into DSO's internal image representation:

- `ImageFolderReader::getImage()` loads the raw image.
- The undistorter applies calibration and photometric preprocessing.
- DSO consumes the result as `ImageAndExposure`.

For TUM RGB-D runs, the association file also provides a depth path. That depth is available to the reader, but it is not automatically used by the tracker.

## 3. What DSO Actually Tracks

DSO tracks camera motion from the RGB stream.

The core tracking step is in `FullSystem::trackNewCoarse()`:

- it uses the current frame and the previous keyframe / reference frame
- it runs coarse-to-fine direct photometric alignment
- it tries several motion hypotheses if the first guess is weak
- it estimates pose and affine brightness parameters

Important point:

- the tracker does not use Gaussian map pose estimates as input
- the tracker does not use raw Kinect depth for pose tracking in the current pipeline

The pose estimate is formed entirely from RGB alignment plus the internal DSO geometry state.

## 4. Keyframe Decision Logic

After tracking, `FullSystem::deliverTrackedFrame()` and `FullSystem::mappingLoop()` decide whether a frame becomes a keyframe.

The decision depends on:

- whether the system is initialized
- whether the queue is falling behind real time
- whether a new keyframe is needed for mapping
- whether the current frame is still useful as a non-keyframe

When a keyframe is chosen, `makeKeyFrame()` is called.

## 5. What DSO Produces

DSO maintains:

- keyframe poses
- affine brightness parameters
- a sparse inverse-depth map
- immature points that may later become active points

Point creation is done in two phases:

1. `makeNewTraces()` creates `ImmaturePoint` candidates from selected pixels.
2. `activatePointsMT()` / `optimizeImmaturePoint()` converts good candidates into `PointHessian` points once their depth is sufficiently constrained.

Point cleanup happens continuously:

- immature points are deleted when they are out of bounds, outliers, or never become usable
- active points are dropped or marginalized when they become out-of-bounds, lose residual support, or the active window moves on

This is DSO's sparse structure. It is not the Gaussian map.

## 6. Gaussian Reconstruction

`GaussianMapper` is the reconstruction backend.

### 6.1 Seeding the map

The first map is created from DSO keyframes in `GaussianMapper::combineMappingOperations()`:

- it consumes newly frozen DSO keyframes
- it copies the DSO world-space points, colors, scales, and rotations
- it creates the initial `GaussianModel` with `createFromPcd()`

That is the bootstrap. The Gaussian scene starts from DSO's sparse geometry.

### 6.2 Training loop

Once initialized, the mapper enters a training loop:

- sample a keyframe
- render the current Gaussian model from that pose
- compare rendered image against the stored keyframe image
- optimize Gaussian parameters

The render-and-optimize path is in `GaussianTrainer` and `GaussianRenderer`.

### 6.3 Densification

The mapper adds detail with `GaussianModel::densifyAndPrune()`:

- points with strong gradients can be split or cloned
- low-opacity points can be removed
- overlarge or undersupported points are pruned

The main control parameters are:

- `Optimization.densify_from_iter`
- `Optimization.densify_until_iter`
- `Optimization.densification_interval`
- `Optimization.densify_grad_threshold`
- `Optimization.densify_min_opacity`

### 6.4 Extra point injection

There is also an inactive-geometry densification path in `GaussianMapper::increasePcdByKeyframeInactiveGeoDensify()`.

It is sensor-mode dependent:

- `MONOCULAR`: use local keypoint neighborhoods and sparse DSO geometry
- `STEREO`: use stereo disparity
- `RGBD`: use the auxiliary depth image and reproject depth into 3D

In this branch, new points are accumulated in a cache and periodically flushed into the Gaussian model. That batching reduces update overhead.

## 7. Feedback Between GS and DSO

This code does have back-feeding, but it is not pose feedback.

The direction is:

- DSO keyframes and sparse points feed the Gaussian mapper
- the Gaussian mapper can render depths back onto DSO keyframes
- DSO then updates some point inverse depths from those rendered depths

The feedback path is:

- `FullSystem::makeKeyFrame()` sets `callKFUpdateFromGS = true`
- `GaussianMapper::run()` notices that request and calls `updateKeyFramesFromGS()`
- `updateKeyFramesFromGS()` renders depth from the current Gaussian scene and updates matching DSO point inverse depths

What it does not do:

- it does not replace DSO's camera pose estimate with a Gaussian pose
- it does not feed a Gaussian-derived pose back into `trackNewCoarse()`

So the tracker still operates on RGB frames and DSO's own motion estimate. The Gaussian scene only helps refine sparse depth after a keyframe has been accepted.

## 8. Performance Controls

The main runtime controls for staying near real time are:

- headless build: `GSO_ENABLE_GUI=OFF`
- one-thread compile in this notebook environment: `cmake --build . -j1`
- lock the renderer during model updates with `mutex_render_`
- batch keyframe and depth-cache updates instead of inserting points one by one
- stop densification after the configured iteration limit
- use sparse DSO geometry as the initial seed instead of starting from a dense reconstruction
- use GPU image resizing and rendering paths when CUDA is available

There is also a second level of performance control in the DSO front-end:

- if tracking falls behind, the mapper can skip non-essential work
- keyframe creation is throttled by the current tracking and mapping state
- inactive points are removed aggressively when they are no longer useful

## 9. Practical Summary

If you only need the operational answer:

- DSO tracks poses from RGB input.
- The Gaussian mapper reconstructs a scene from DSO keyframes and sparse depth.
- The map can feed depth corrections back to DSO keyframes.
- The map does not currently replace the tracker's pose estimate.
- Raw TUM depth is only used if the run is explicitly configured for RGB-D.
