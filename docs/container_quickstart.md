# Container Quickstart

GSO-SLAM now ships Compose examples for the published GHCR images:

- `ghcr.io/bjoernellens1/gso-slam-cuda:cuda-latest`
- `ghcr.io/bjoernellens1/gso-slam-rocm:rocm-latest`

The Compose files are:

- `compose.cuda.yml`
- `compose.rocm.yml`

Both mount the local `dataset/` directory at `/datasets` and write outputs to `docker/results/`.

## What Works Today

The published images are pushed from the Docker `builder` stage. They are suitable for:

- image smoke checks
- dataset preprocessing
- post-run evaluation on mounted result folders

Example commands:

```bash
docker compose -f compose.cuda.yml run --rm smoke
docker compose -f compose.cuda.yml run --rm preprocess-datasets
docker compose -f compose.cuda.yml run --rm -e GSO_REPLICA_SCENE=room0 replica-ate
docker compose -f compose.cuda.yml run --rm -e GSO_REPLICA_SCENE=room0 replica-rendering
docker compose -f compose.cuda.yml run --rm -e GSO_TUM_SCENE=freiburg1_desk tum-ate
```

ROCm uses the same service names:

```bash
docker compose -f compose.rocm.yml run --rm smoke
docker compose -f compose.rocm.yml run --rm -e GSO_REPLICA_SCENE=office0 replica-depth
```

## Dataset Layout

The Compose wrappers assume the repository dataset layout:

- `dataset/data/Replica/...` and `dataset/data/TUM/...` for raw downloads
- `dataset/Replica/...` and `dataset/TUM/...` for preprocessed runtime data

Run `preprocess-datasets` after downloading data with the existing repo scripts.

## Important Limitation

`dso_dataset` is only built when Pangolin is available during image build. The current published images do not install Pangolin, so they should be treated as evaluation and environment images, not guaranteed full SLAM runtime images. Run `smoke` first to confirm what the image contains.

## Podman / ROCm Notes

`compose.rocm.yml` passes `/dev/kfd` and `/dev/dri` through with `devices:` and adds the `video` group, which is the minimum setup for Podman and Docker ROCm containers. On Fedora or other SELinux-enabled hosts, add relabelled bind mounts if needed. If your environment uses the AMD Container Runtime Toolkit with CDI, adapt the ROCm service to your CDI device names instead of raw device paths.
