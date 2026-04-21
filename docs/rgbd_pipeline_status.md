# RGB-D Pipeline Fix Status

## Overview

Three bugs were identified in the GSO-SLAM RGB-D pipeline that caused exploded/blurry geometry when running TUM RGB-D and Kitchen ICL datasets. Bugs 1 and 2 are fully fixed. Bug 3 has a partial implementation (post-tracking scale correction) that compiles but has not been validated on a full sequence due to a pre-existing GaussianMapper crash on this branch.

---

## Bug 1: `SLAM.depth_scale` Never Read from Config

**Status:** ✅ Fixed and applied

**Problem:** `depth_scale` was hardcoded to `1000.0f` in `gaussian_mapper.h:284`. The `readConfigFromFile()` function never read `SLAM.depth_scale` from config YAML files.

**Fix:** Added `depth_scale = getOptionalValue<float>(settings_file, "SLAM.depth_scale", 1000.0f);` at the end of `readConfigFromFile()` in `src/GS/gaussian_mapper.cpp`.

---

## Bug 2: TUM Wrapper Missing `which_dataset=tum_rgbd`

**Status:** ✅ Fixed and applied

**Problem:** `experiments_bash/tum.sh` passed `dataassociation=rgb.txt` but never set `which_dataset=tum_rgbd`, so `ImageFolderReader` never called `LoadImages()` to populate depth file lists.

**Fix:** Added `which_dataset=tum_rgbd` to the `dso_dataset` command in `experiments_bash/tum.sh`. Also removed references to `pcalib.txt` and `vignette.png` which don't exist in TUM datasets.

---

## Config Updates

**Status:** ✅ Applied

All TUM configs updated with correct `depth_scale` and `sensor_type`:

| Config File | depth_scale | sensor_type |
|---|---|---|
| `tum_freiburg1_desk.yaml` | 5000 | rgbd |
| `tum_freiburg1_desk_quality.yaml` | 5000 | rgbd |
| `tum_freiburg2_xyz.yaml` | 5000 | rgbd |
| `tum_freiburg3_long_office_household.yaml` | 5000 | rgbd |
| `tum_kitchen_icl_10hz.yaml` | 1000 | rgbd |
| `tum_kitchen_icl_10hz_*.yaml` (5 variants) | 1000 | rgbd |

---

## Bug 3: DSO Frontend Tracks Monocular-Only (Scale Drift)

**Status:** ⚠️ Partially implemented, not validated

**Problem:** `trackNewCoarse()` in `FullSystem.cpp:933` performs purely photometric alignment. Depth is stored on frames (`fh->kf_depth`) but never used during pose estimation. DSO's poses drift in arbitrary monocular scale.

### Implemented: Option A — Post-Tracking Scale Correction

**Approach:** After `trackNewCoarse()` returns, compute the median scale ratio between DSO-derived idepth and RGB-D idepth at overlapping pixels in the reference frame. Smooth the ratio over time with exponential moving average (α=0.05). Correct the pose translation by dividing by the smoothed scale ratio.

**Files modified:**

| File | Changes |
|---|---|
| `src/util/settings.h` | Added `setting_rgbdTrackingWeight`, `setting_rgbdDepthScale`, `setting_rgbdTrackingMode` |
| `src/util/settings.cpp` | Added default values for RGB-D settings |
| `src/FullSystem/CoarseTracker.h` | Added `makeCoarseRGBDDepthL0()`, `computeRGBDScaleRatio()`, `rgbd_idepth[]`, `rgbd_weightSums[]` |
| `src/FullSystem/CoarseTracker.cpp` | Implemented RGB-D depth pyramid building, modified normalization to prefer RGB-D idepth, added `computeRGBDScaleRatio()` |
| `src/FullSystem/FullSystem.h` | Added `applyRGBDScaleCorrection()`, `rgbdSmoothedScaleRatio`, `rgbdScaleCorrectionCount` |
| `src/FullSystem/FullSystem.cpp` | Added scale correction logic called after `trackNewCoarse()` |
| `src/main_dso_pangolin.cpp` | Wired up RGB-D settings based on dataset name (lines 728-744) |

**Key implementation details:**

1. **`CoarseTracker::makeCoarseRGBDDepthL0()`** — Builds a depth pyramid from raw `MinimalImage<unsigned short>` using `setting_rgbdDepthScale`. Called from `setCoarseTrackingRef()` before `makeCoarseDepthL0()`.

2. **`makeCoarseDepthL0()` normalization** — Modified to prefer RGB-D idepth over DSO idepth where available. This injects metric-scale depth into the coarse point cloud.

3. **`CoarseTracker::computeRGBDScaleRatio()`** — Computes median ratio `dso_idepth / rgbd_idepth` at overlapping pixels in the reference frame's L0 depth map (sampled every 2 pixels, requires ≥50 samples).

4. **`FullSystem::applyRGBDScaleCorrection()`** — Called after every `trackNewCoarse()`. Smooths the scale ratio with EMA (α=0.05), then divides `fh->shell->camToWorld.translation()` by the smoothed ratio. Logs every 50 frames.

### Build Status

✅ Compiles successfully on branch `feature/rgbd-depth-aided-tracking`.

### Testing Status

⚠️ **Not validated on full sequences.** The GaussianMapper on this branch crashes with a segfault immediately on startup for both TUM freiburg1_desk and kitchen_icl_10hz datasets. This is a **pre-existing issue** on the branch (not caused by Bug 3 changes) — the original code runs fine with these datasets. The crash is in the branch's existing `gaussian_mapper.cpp` changes (~680 lines of diff), likely related to `kfSparseDepth` handling for RGB-D datasets.

**Previous test results (before scale correction):**

| Dataset | Config | Status | Frames | Keyframes | Scale | Vertices | Geometry Quality |
|---|---|---|---|---|---|---|---|
| Kitchen ICL 10Hz | depth_scale=1000 | ✅ Runs | 681 | 133 | ~0.44 | 15,597 | Exploded |
| Kitchen ICL 10Hz (200 frames) | depth_scale=1000 | ✅ Runs | 199 | 40 | ~0.51 | 17,518 | Exploded |
| Freiburg 1 Desk | depth_scale=5000 | ❌ Crashes | - | - | - | - | N/A |

---

## Pre-existing Branch Issue: GaussianMapper Crash

**Status:** ❌ Not addressed

The GaussianMapper crashes with a segfault on this branch when running TUM RGB-D datasets. The crash occurs in `GaussianMapper::run` shortly after startup, before any tracking output is produced. This affects both freiburg1_desk and kitchen_icl_10hz.

This is caused by the branch's existing changes to `gaussian_mapper.cpp` (large diff, ~680 lines), likely related to how `kfSparseDepth` or `kfDepth` is handled for RGB-D datasets. The original (upstream) code runs fine with these datasets.

**Impact:** Cannot validate Bug 3 scale correction on full sequences until this crash is fixed.

---

## Files Changed on Branch

### Core RGB-D fixes (Bugs 1, 2, 3):
- `src/GS/gaussian_mapper.cpp` — depth_scale config read, RGB-D seed geometry handling
- `src/util/settings.h` — RGB-D tracking settings
- `src/util/settings.cpp` — RGB-D tracking defaults
- `src/FullSystem/CoarseTracker.h` — RGB-D depth pyramid arrays
- `src/FullSystem/CoarseTracker.cpp` — RGB-D depth pyramid building, scale ratio computation
- `src/FullSystem/FullSystem.h` — scale correction state and method
- `src/FullSystem/FullSystem.cpp` — scale correction after tracking
- `src/FullSystem/HessianBlocks.h` — kf_depth, kfDepth members
- `experiments_bash/tum.sh` — which_dataset flag, removed pcalib/vignette refs

### Config files:
- `cfg/gaussian_mapper/Monocular/TUM/tum_freiburg1_desk.yaml`
- `cfg/gaussian_mapper/Monocular/TUM/tum_freiburg1_desk_quality.yaml`
- `cfg/gaussian_mapper/Monocular/TUM/tum_freiburg2_xyz.yaml`
- `cfg/gaussian_mapper/Monocular/TUM/tum_freiburg3_long_office_household.yaml`
- `cfg/gaussian_mapper/Monocular/TUM/tum_kitchen_icl_10hz.yaml` (+ 5 variants)

### Other branch changes (not RGB-D related):
- `include/camera.h`
- `include/gaussian_mapper.h`
- `src/GS/gaussian_mapper.cpp` (large diff beyond Bug 1 fix)
- `experiments_bash/replica.sh`
- `experiments_bash/replica_eval_depth.sh`
- `experiments_bash/replica_eval_rendering.sh`
- `experiments_bash/scripts/evaluate_ate_scale_replica.py`
- `experiments_bash/scripts/evaluate_ate_scale_tum.py`
- `docs/pipeline_detailed.md`
- `docs/view_results_in_jupyter.ipynb`
- `cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml`
- `dataset/preprocess.sh`

---

## What Is Not Working Currently

### 1. GaussianMapper Crashes on TUM RGB-D Datasets (Blocking)

**Symptom:** Immediate segfault after `[DEBUG] sensor_type_=3 (RGBD=3)` is printed. No tracking output is produced. Affects **all** TUM RGB-D datasets (freiburg1_desk, freiburg2_xyz, freiburg3_long_office_household, kitchen_icl_10hz).

**Root cause:** The branch's `gaussian_mapper.cpp` has ~680 lines of new code for RGB-D seed geometry handling. The crash likely occurs when accessing `fh->kfSparseDepth` or `fh->pointPixels` which are either empty or not properly populated for TUM datasets. The `appendSeedGeometry()` function (lines 199+ in the diff) has new code paths that check `!fh->kfSparseDepth.empty()` and `fh->pointPixels.size() == num_points * 2` — if these assumptions don't hold for TUM data, a null dereference or out-of-bounds access occurs.

**Impact:** **Blocks all testing.** Cannot run any TUM RGB-D dataset through the full pipeline. The DSO frontend may work, but the process dies before any meaningful output is produced.

**Note:** The original (upstream) code runs fine with these same datasets. This crash was introduced by the branch's gaussian_mapper.cpp changes.

### 2. Bug 3 Scale Correction Not Validated

**Symptom:** The post-tracking scale correction code compiles and is wired in, but has never been tested on a full sequence because of the GaussianMapper crash above.

**What's uncertain:**
- Whether `computeRGBDScaleRatio()` returns sensible values on real data
- Whether the EMA smoothing (α=0.05) is appropriate
- Whether dividing pose translation by the smoothed ratio actually fixes the exploded geometry
- Whether the scale correction interacts correctly with the DSO optimization loop (it runs after `trackNewCoarse()` but before the pose is used for keyframe creation)

### 3. RGB-D Idepth Preference in CoarseTracker Untested

**Symptom:** The modification to `makeCoarseDepthL0()` that prefers RGB-D idepth over DSO idepth was previously tested and found to be **insufficient** on its own. Debug output showed `rgbd=926506 dso=545` — RGB-D depth was being used heavily, but rendered geometry was still blurry/exploded. This is expected: replacing idepth values without geometric residuals in the optimization doesn't constrain the pose to metric scale.

**Current state:** This code is still in place and runs alongside the new scale correction. It may help or may interfere — unknown until tested.

### 4. Freiburg 1 Desk Crash (Separate Pre-existing Issue)

**Symptom:** Segfault in `GaussianMapper::run` — same root cause as #1 above, but specifically called out because freiburg1_desk was the primary test dataset in the original evaluation.

### 5. Replica Dataset Not Tested on This Branch

**Status:** Unknown. Replica datasets have `pcalib.txt` and `vignette.png` which TUM lacks, and use `mode=2` instead of `mode=1`. The `which_dataset=replica` path in `main_dso_pangolin.cpp` sets `setting_rgbdTrackingMode=1` and `setting_rgbdDepthScale=1000`, so the RGB-D tracking code would be active. Whether the GaussianMapper crash also affects Replica is untested.

---

## Next Steps

1. **Fix GaussianMapper crash** — Debug the pre-existing segfault in `gaussian_mapper.cpp` on this branch. Likely related to `kfSparseDepth`/`kfDepth` handling for TUM RGB-D datasets.

2. **Validate scale correction** — Once the mapper crash is fixed, run full sequences (kitchen_icl_10hz, freiburg1_desk) and verify:
   - `[RGB-D Scale]` log output shows stable smoothed scale near 1.0
   - Rendered geometry is no longer exploded
   - ATE evaluation shows improvement

3. **Alternative: test on Replica** — If TUM datasets continue to crash, try Replica datasets which may have different depth handling paths.
