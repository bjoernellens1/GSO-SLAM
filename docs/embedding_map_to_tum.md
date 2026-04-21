# Embedding Map Export to TUM

The dataset under `/home/jovyan/cps_persistent1_shared/datasets/christian/embedding_map/` is a ROS2 bag export, not a native TUM dataset. The useful export path is the `export/<rate>/kitchen_icl.tar.gz` archive, which contains:

- RGB frames under `rgb/`
- aligned depth PNGs under `depth/`
- camera intrinsics and depth scale in `icl.yaml`
- per-frame poses in `poses.gt.sim`

The archive already contains matched RGB/depth pairs by filename, so conversion to TUM format is a data-layout step, not a frame association problem.

## Converter

Use [dataset/convert_embedding_map_to_tum.py](/home/jovyan/work/GSO-SLAM/dataset/convert_embedding_map_to_tum.py) to convert either the tarball or an extracted export folder:

```bash
python3 dataset/convert_embedding_map_to_tum.py \
  --source /home/jovyan/cps_persistent1_shared/datasets/christian/embedding_map/export/10hz/kitchen_icl.tar.gz \
  --overwrite
```

By default the output is written under `dataset/TUM` in this repo, with the export rate appended when the source path contains `1hz`, `5hz`, or `10hz`.

The converter writes:

- `rgb/`
- `depth/`
- `rgb.txt`
- `depth.txt`
- `rgbd_association.txt`
- `groundtruth.txt`
- `camera.txt`
- `conversion_manifest.json`

`rgb.txt` and `depth.txt` follow the standard TUM list format. `rgbd_association.txt` is the pipeline-friendly 4-column association file if you want the loader to keep depth paths available.

## Pipeline Use

For the current pipeline, the safest default is to keep the converted dataset under `dataset/TUM/rgbd_dataset_kitchen_icl_10hz` and point the runtime at the association file:

- `files=.../rgb`
- `dataassociation=.../rgbd_association.txt`
- `calib=.../camera.txt`

If you want the RGB-D path in the loader, set the dataset mode to the RGB-D branch used by this repo and pass the 4-column association file. If you only need the existing mono-style TUM path, `rgb.txt` is still available.

## Notes

- `camera.txt` is emitted in the repository's `RadTanK3` format.
- `groundtruth.txt` is generated from `poses.gt.sim` using the RGB frame timestamps parsed from the exported filenames.
- The `10hz` export is usually the best candidate if you want the densest sequence for benchmarking.

## Findings

### Correct runtime path

For the imported kitchen scene, the runtime should use the RGB-D loader branch:

- `which_dataset=tum_rgbd`
- `files=.../rgb`
- `dataassociation=.../rgbd_association.txt`
- `calib=.../camera.txt`

Using `which_dataset=tum` keeps the mono-style TUM path and does not populate `depth_files` from the 4-column association file.

### Root cause of the collapsed map

The bad full-scene run was not caused by the dataset conversion itself. It was caused by depth-evidence pruning on top of an unstable per-keyframe RGB-D depth alignment step.

The aligned depth image is produced in [src/GS/gaussian_mapper.cpp](/home/jovyan/work/GSO-SLAM/src/GS/gaussian_mapper.cpp) by:

- `prepareAlignedRgbdDepthImage()`
- `estimateDepthAlignmentScale()`

`estimateDepthAlignmentScale()` recomputes a fresh scale for every keyframe as the median of:

- `sparse_depth / rgbd_depth`

where:

- `rgbd_depth` comes from the imported PNG depth converted with the fixed export scale from `conversion_manifest.json`
- `sparse_depth` comes from DSO keyframe points via `1 / ph->idepth_scaled`

That means the imported metric depth is being aligned to a sparse monocular DSO depth map whose scale and support pixels vary from keyframe to keyframe. On the kitchen import this produced alignment factors roughly between `0.29` and `0.88`, which is unstable enough to poison the free-space pruning signal.

The pruning path is:

- [src/GS/gaussian_mapper.cpp](/home/jovyan/work/GSO-SLAM/src/GS/gaussian_mapper.cpp): `gaussians_->pruneByDepthEvidence(...)`
- [src/GS/gaussian_model.cpp](/home/jovyan/work/GSO-SLAM/src/GS/gaussian_model.cpp): `GaussianModel::pruneByDepthEvidence(...)`

One important detail is that `pruneByDepthEvidence()` does not only prune by contradictory depth evidence. It also prunes on global anisotropy and scale thresholds inside the same pass. When the per-keyframe aligned depth is unstable, that combined prune can wipe almost the whole map.

### Reproduced results

Imported kitchen scene, same `tum_rgbd` loader, same full-scene input:

- `tum_kitchen_icl_10hz_rgbd_fullscene` with `tum_kitchen_icl_10hz_aligned_depthprune.yaml`: `1091` splats
- `tum_kitchen_icl_10hz_rgbd_noprune` with `tum_kitchen_icl_10hz.yaml`: `71997` splats

Native and baseline comparisons:

- `tum_freiburg1_desk`: `65617` splats
- `replica_room0_fullscene`: `111042` splats

Existing older kitchen runs in this repo also stayed large:

- `tum_kitchen_icl_10hz_headless`: `74789` splats
- `tum_kitchen_icl_10hz_aligned_depthprune`: `73763` splats

### Practical conclusion

For the imported kitchen scene, the safe default is the base config without depth-evidence pruning:

```bash
./build/bin/dso_dataset \
  files="dataset/TUM/rgbd_dataset_kitchen_icl_10hz/rgb" \
  calib="dataset/TUM/rgbd_dataset_kitchen_icl_10hz/camera.txt" \
  dataassociation="dataset/TUM/rgbd_dataset_kitchen_icl_10hz/rgbd_association.txt" \
  preset=0 mode=1 quiet=1 nogui=1 which_dataset=tum_rgbd \
  cfg_yaml="cfg/gaussian_mapper/Monocular/TUM/tum_kitchen_icl_10hz.yaml" \
  save_dir="experiments_bash/results/test/tum_kitchen_icl_10hz_rgbd_noprune" \
  use_gaussian_viewer=0
```

If depth-evidence pruning is re-enabled for imported RGB-D scenes, it should first be stabilized by either:

- smoothing or fixing the depth alignment scale across the sequence, or
- disabling the prune when the estimated alignment scale is unstable.
