# Bug 3: RGB-D Frontend Scale Drift — Evaluation & Findings

## Problem Statement

The DSO frontend tracks **monocular-only** even when running in RGB-D mode. Depth images are loaded and stored on frames (`fh->kf_depth`) but never used during pose estimation. This causes DSO's poses to drift in an arbitrary monocular scale, and the mapper's post-hoc depth alignment cannot fully compensate.

## Evidence

### Rendered Geometry
All three context pose views (start, mid, end) show **exploded geometry**: mostly black with scattered blurry blobs and streaks. No coherent scene structure is visible.

### Scale Drift Log
```
Keyframe 634 smoothed RGB-D depth scale from raw 0.259776 to sequence 0.442492
```
- The alignment scale of ~0.44 means DSO's monocular scale is ~2.3x the metric RGB-D scale
- This drifts throughout the sequence (from ~0.65 early to ~0.33 late in earlier runs)
- The mapper's `stabilizeRgbdAlignmentScale()` tries to smooth this but cannot fully compensate because the poses themselves are wrong scale

### Root Cause Location
`src/FullSystem/FullSystem.cpp:933` — `addActiveFrame()` always calls `trackNewCoarse(fh)` for pose estimation. This function (line 275-464) performs purely photometric alignment. The depth image `fh->kf_depth` (stored at line 886) is never consulted during tracking.

## Fixed Bugs (Already Applied)

### Bug 1: `SLAM.depth_scale` Never Read from Config
- **File:** `src/GS/gaussian_mapper.cpp`
- **Fix:** Added `depth_scale = getOptionalValue<float>(settings_file, "SLAM.depth_scale", 1000.0f);` at the end of `readConfigFromFile()`
- **Status:** ✅ Applied

### Bug 2: TUM Wrapper Missing `which_dataset=tum_rgbd`
- **File:** `experiments_bash/tum.sh`
- **Fix:** Added `which_dataset=tum_rgbd` to the dso_dataset command
- **Status:** ✅ Applied

### Config Updates
- All 6 kitchen configs: `SLAM.depth_scale: 1000`, `SLAM.sensor_type: rgbd`
- All 4 freiburg configs: `SLAM.depth_scale: 5000`, `SLAM.sensor_type: rgbd`
- **Status:** ✅ Applied

## Bug 3: Three Proposed Approaches

### Option A: Post-Tracking Scale Correction (Recommended)

**Concept:** Correct pose scale after photometric tracking. DSO's photometric optimization is completely unchanged — rotation and relative geometry stay correct. Only the translation magnitude (scale) is corrected.

**Location:** `src/FullSystem/FullSystem.cpp`, in `addActiveFrame()` after `trackNewCoarse()` returns (around line 939).

**Algorithm:**
1. After `trackNewCoarse(fh)` succeeds, if `mode==1` (RGB-D) and `fh->kf_depth` is available:
   - For each active point in the frame (`fh->pointHessians`), get its DSO depth (`1/ph->idepth_scaled`)
   - Look up the metric depth from `fh->kf_depth` at the same pixel `(ph->u, ph->v)`
   - Compute per-point scale ratios: `metric_depth / DSO_depth`
   - Take the median ratio as the frame's scale estimate
   - Smooth this scale over time using exponential moving average
   - Apply the smoothed scale factor to `fh->shell->camToWorld.translation()`

**Files to modify:**
1. `src/util/settings.h` — add `extern float setting_rgbdScaleSmoothing;`
2. `src/util/settings.cpp` — add `float setting_rgbdScaleSmoothing = 0.1f;`
3. `src/FullSystem/FullSystem.h` — add `float rgbdSmoothedScale_;` and `float rgbdDepthScale_;` members
4. `src/FullSystem/FullSystem.cpp` — add scale correction logic after `trackNewCoarse()`
5. `src/main_dso_pangolin.cpp` — pass `depth_scale` from config to `FullSystem`

**Trade-offs:**
- ✅ Minimal changes to DSO's core tracking — photometric optimization is untouched
- ✅ Works incrementally, correcting scale drift frame-by-frame
- ✅ Low risk of breaking existing functionality
- ⚠️ Scale correction is applied after tracking, so the initial pose hypothesis is still monocular (but this is fine since the correction is applied before the pose is used)
- ⚠️ Requires the depth image to be available at tracking time (it is, via `fh->kf_depth`)

### Option B: Depth-Aided Tracking (ICP + Photometric)

**Concept:** Add geometric (depth-based) residuals alongside photometric residuals in the coarse tracker. This would make tracking truly RGB-D aware.

**Location:** `src/FullSystem/CoarseTracker.cpp`, in `calcRes()` and `calcGSSSE()`.

**Algorithm:**
1. In `makeCoarseDepthL0()`, also build a dense depth map from `lastRef->kf_depth` (converted to metric using `depth_scale`)
2. In `calcRes()`, for each reference point with a valid depth measurement, add a geometric residual term: `projected_depth - measured_depth`
3. In `calcGSSSE()`, compute the Jacobian of the geometric residual with respect to SE3 pose parameters
4. Weight the geometric residual appropriately relative to the photometric residual

**Trade-offs:**
- ✅ Most accurate — tracking would be truly RGB-D aware from the start
- ✅ Would eliminate scale drift entirely at the source
- ❌ Much more invasive — requires rewriting core tracking functions
- ❌ High risk of breaking existing functionality
- ❌ Requires careful tuning of geometric vs photometric residual weights
- ❌ Would need to handle depth sensor noise, missing depth pixels, etc.

### Option C: Skip Bug 3

**Concept:** Keep only the two config/script fixes (Bugs 1 and 2). The pipeline runs but geometry remains blurry.

**Trade-offs:**
- ✅ Zero risk — no code changes to the tracking pipeline
- ✅ Config fixes are still valuable (depth_scale is correctly read, depth files are loaded)
- ❌ Geometry remains exploded/blurry — renders are unusable
- ❌ The fundamental RGB-D integration problem is not addressed

## Test Results Summary

| Dataset | Config | Status | Frames | Keyframes | Scale | Vertices | Geometry Quality |
|---------|--------|--------|--------|-----------|-------|----------|-----------------|
| Kitchen ICL 10Hz | depth_scale=1000 | ✅ Runs | 681 | 133 | ~0.44 | 15,597 | Exploded |
| Kitchen ICL 10Hz (200 frames) | depth_scale=1000 | ✅ Runs | 199 | 40 | ~0.51 | 17,518 | Exploded |
| Freiburg 1 Desk | depth_scale=5000 | ❌ Crashes | - | - | - | - | N/A |

**Note:** Freiburg 1 Desk crashes with a segfault in `GaussianMapper::run`. This is a pre-existing issue on the branch (not caused by Bug 1/2 fixes) — the original code runs fine with freiburg1_desk. The crash is in the branch's existing `gaussian_mapper.cpp` changes (~680 lines of diff), likely related to how `kfSparseDepth` or `kfDepth` is handled for the freiburg dataset.

## Key Log Observations

```
[Gaussian Mapper]Keyframe 634 smoothed RGB-D depth scale from raw 0.259776 to sequence 0.442492 using 133 keyframes.
```

The `raw` scale (0.26) is the per-frame median ratio of DSO depth to metric RGB-D depth. The `sequence` scale (0.44) is the smoothed value across 133 keyframes. The fact that these differ significantly and change over time confirms that scale drift is ongoing throughout the sequence.

## Recommended Path Forward

1. **Implement Option A** (post-tracking scale correction) — it's the safest and most effective fix
2. **Debug the Freiburg crash** separately — it's a pre-existing branch issue unrelated to the RGB-D scale problem
3. **Re-evaluate renders** after Option A is implemented to verify geometry quality improves
