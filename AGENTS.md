# Repository Guidelines

## Project Structure & Module Organization
Core C++ and GPU code lives in `src/` and public headers in `include/`. The SLAM front-end and optimization back-end are mainly under `src/FullSystem/` and `src/OptimizationBackend/`; Gaussian Splatting code is under `src/GS/`; viewers and UI code live in `viewer/` and `src/IOWrapper/`. Runtime configs are in `cfg/gaussian_mapper/`, dataset helpers in `dataset/`, evaluation scripts in `experiments_bash/`, and longer technical docs in `docs/`. Third-party code is vendored in `thirdparty/`.

## Build, Test, and Development Commands
Configure and build locally with:

```bash
mkdir -p build && cd build
cmake ..
make -j"$(nproc)"
```

Use `cmake -DUSE_ROCM=ON ..` for AMD/ROCm builds. Run the common entrypoints from `experiments_bash/`, for example `bash experiments_bash/replica.sh` or `bash experiments_bash/tum.sh`. Dataset setup is script-driven: `bash dataset/download_replica.sh`, `bash dataset/download_tum.sh`, then `bash dataset/preprocess.sh`.

CI treats Docker builds as the main smoke test:

```bash
docker build --target configure -f docker/Dockerfile.cuda .
docker build --target builder -f docker/Dockerfile.cuda .
```

Equivalent ROCm commands use `docker/Dockerfile.rocm`.

## Coding Style & Naming Conventions
Follow the existing C++17/CUDA style in the tree: 4-space indentation, opening braces on the same line, and headers paired with nearby `.cpp` or `.cu` files. Preserve current naming patterns: `CamelCase` for classes (`GaussianMapper`), `snake_case` for scripts and YAML files (`replica_eval_depth.sh`), and descriptive config names under `cfg/`. Keep edits narrow in vendored directories unless the change explicitly targets third-party code.

## Testing Guidelines
There is no dedicated unit-test suite in the root project today; validation is primarily compile-focused. Before opening a PR, run at least one local build, one relevant experiment script, and the docs build if you touched `docs/`, `include/`, or `src/`:

```bash
mkdocs build --strict
doxygen Doxyfile
```

## Commit & Pull Request Guidelines
Recent history uses concise, scoped subjects such as `ci: add permissions...` and `docker/ci: fix ROCm cmake issues`. Prefer imperative commit messages with a clear scope prefix when useful. PRs should describe the affected subsystem, note CUDA vs. ROCm impact, link related issues, and include screenshots only for viewer or docs UI changes.
