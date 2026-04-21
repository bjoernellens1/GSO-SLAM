#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


FRAME_RE = re.compile(
    r"^(?:frame_)?(?P<date>\d{8})_(?P<time>\d{6})_(?P<micro>\d{6})_(?P<index>\d+)$"
)


@dataclass(frozen=True)
class FrameEntry:
    stamp: float
    stem: str
    rgb_name: str
    depth_name: str


@dataclass(frozen=True)
class SequenceData:
    sequence_name: str
    camera_yaml: str
    poses_sim: str
    rgb_entries: List[str]
    depth_entries: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert embedding_map RGB-D export tarballs into a TUM-style dataset."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to an export tarball (*.tar.gz) or an extracted sequence directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination directory for the TUM-style dataset root. Defaults to dataset/TUM inside this repo.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("auto", "copy", "hardlink", "symlink"),
        default="auto",
        help="How to materialize files when the source is an extracted directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory first if it already exists.",
    )
    return parser.parse_args()


def parse_frame_filename(name: str) -> FrameEntry:
    stem = Path(name).stem
    match = FRAME_RE.match(stem)
    if not match:
        raise ValueError(f"Unsupported frame filename: {name}")

    stamp_text = "{date}{time}{micro}".format(**match.groupdict())
    stamp = dt.datetime.strptime(stamp_text, "%Y%m%d%H%M%S%f").replace(
        tzinfo=dt.timezone.utc
    ).timestamp()
    return FrameEntry(
        stamp=stamp,
        stem=stem,
        rgb_name=f"rgb/{stamp:.6f}.jpg",
        depth_name=f"depth/{stamp:.6f}.png",
    )


def matrix_to_quaternion(rotation: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    r00, r01, r02 = rotation[0]
    r10, r11, r12 = rotation[1]
    r20, r21, r22 = rotation[2]

    trace = r00 + r11 + r22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r21 - r12) / s
        qy = (r02 - r20) / s
        qz = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        qw = (r21 - r12) / s
        qx = 0.25 * s
        qy = (r01 + r10) / s
        qz = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        qw = (r02 - r20) / s
        qx = (r01 + r10) / s
        qy = 0.25 * s
        qz = (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        qw = (r10 - r01) / s
        qx = (r02 + r20) / s
        qy = (r12 + r21) / s
        qz = 0.25 * s

    return qx, qy, qz, qw


def parse_camera_yaml(text: str) -> dict:
    values: dict[str, object] = {}

    def parse_scalar(value: str) -> object:
        if value.startswith("[") and value.endswith("]"):
            return [
                float(item.strip())
                for item in value[1:-1].split(",")
                if item.strip()
            ]
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        try:
            if any(ch in value for ch in ".eE"):
                return float(value)
            return int(value)
        except ValueError:
            return value

    in_camera_params = False
    in_distortion = False
    camera_params: dict[str, object] = {}
    distortion: List[float] = []

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if indent == 0:
            in_camera_params = False
            in_distortion = False
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                if key == "camera_params":
                    in_camera_params = True
                    continue
                values[key] = {}
            else:
                values[key] = parse_scalar(value)
            continue

        if not in_camera_params:
            continue

        if indent == 2 and line.startswith("distortion:"):
            in_distortion = True
            camera_params["distortion"] = distortion
            continue

        if in_distortion:
            if line.startswith("-"):
                distortion.append(float(line[1:].strip()))
                continue
            in_distortion = False

        if indent == 2 and ":" in line:
            key, value = line.split(":", 1)
            camera_params[key.strip()] = parse_scalar(value.strip())

    if distortion and "distortion" not in camera_params:
        camera_params["distortion"] = distortion
    if camera_params:
        values["camera_params"] = camera_params
    return values


def load_sequence_from_tar(source: Path) -> SequenceData:
    with tarfile.open(source, "r:gz") as tar:
        members = tar.getmembers()
        top_level = []
        for member in members:
            if not member.name or member.name.endswith("/"):
                continue
            parts = Path(member.name).parts
            if not parts:
                continue
            top_level.append(parts[0])
        if not top_level:
            raise RuntimeError(f"No sequence data found in {source}")
        sequence_name = sorted(set(top_level))[0]

        def read_text(member_name: str) -> str:
            extracted = tar.extractfile(member_name)
            if extracted is None:
                raise RuntimeError(f"Missing {member_name} in {source}")
            return extracted.read().decode("utf-8")

        camera_yaml = read_text(f"{sequence_name}/icl.yaml")
        poses_sim = read_text(f"{sequence_name}/poses.gt.sim")

        rgb_entries: List[str] = []
        depth_entries: List[str] = []
        for member in members:
            if not member.isfile():
                continue
            if member.name.startswith(f"{sequence_name}/rgb/") and member.name.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                rgb_entries.append(Path(member.name).name)
            elif member.name.startswith(f"{sequence_name}/depth/") and member.name.lower().endswith(
                ".png"
            ):
                depth_entries.append(Path(member.name).name)

        rgb_entries.sort()
        depth_entries.sort()
        return SequenceData(
            sequence_name=sequence_name,
            camera_yaml=camera_yaml,
            poses_sim=poses_sim,
            rgb_entries=rgb_entries,
            depth_entries=depth_entries,
        )


def locate_sequence_dir(source: Path) -> Path:
    if (source / "icl.yaml").is_file() and (source / "poses.gt.sim").is_file():
        return source
    for child in sorted(source.iterdir()):
        if not child.is_dir():
            continue
        if (child / "icl.yaml").is_file() and (child / "poses.gt.sim").is_file():
            return child
    raise RuntimeError(f"Could not locate a sequence root under {source}")


def read_sequence_from_dir(source: Path) -> SequenceData:
    root = locate_sequence_dir(source)
    rgb_dir = root / "rgb"
    depth_dir = root / "depth"
    if not rgb_dir.is_dir() or not depth_dir.is_dir():
        raise RuntimeError(f"Expected rgb/ and depth/ under {root}")

    rgb_entries = sorted([path.name for path in rgb_dir.iterdir() if path.is_file()])
    depth_entries = sorted([path.name for path in depth_dir.iterdir() if path.is_file()])
    return SequenceData(
        sequence_name=root.name,
        camera_yaml=(root / "icl.yaml").read_text(encoding="utf-8"),
        poses_sim=(root / "poses.gt.sim").read_text(encoding="utf-8"),
        rgb_entries=rgb_entries,
        depth_entries=depth_entries,
    )


def parse_poses(poses_text: str) -> List[List[List[float]]]:
    rows = [line.strip() for line in poses_text.splitlines() if line.strip()]
    if len(rows) % 3 != 0:
        raise RuntimeError("poses.gt.sim does not contain a multiple of 3 non-empty lines")

    matrices: List[List[List[float]]] = []
    for idx in range(0, len(rows), 3):
        matrix: List[List[float]] = []
        for line in rows[idx : idx + 3]:
            values = [float(item) for item in line.split()]
            if len(values) != 4:
                raise RuntimeError(f"Invalid pose matrix row: {line}")
            matrix.append(values)
        matrices.append(matrix)
    return matrices


def ensure_output_dir(output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise RuntimeError(f"Output directory already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def materialize_file(src: Path, dst: Path, link_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "copy":
        shutil.copy2(src, dst)
        return
    if link_mode == "hardlink":
        os.link(src, dst)
        return
    if link_mode == "symlink":
        os.symlink(src, dst)
        return
    if link_mode == "auto":
        try:
            os.link(src, dst)
            return
        except OSError:
            try:
                os.symlink(src, dst)
                return
            except OSError:
                shutil.copy2(src, dst)
                return
    raise ValueError(f"Unknown link mode: {link_mode}")


def write_camera_file(output: Path, camera_yaml: str) -> dict:
    data = parse_camera_yaml(camera_yaml)
    params = data.get("camera_params")
    if not isinstance(params, dict):
        raise RuntimeError("camera_params missing from icl.yaml")

    fx = float(params["fx"])
    fy = float(params["fy"])
    cx = float(params["cx"])
    cy = float(params["cy"])
    width = int(params["image_width"])
    height = int(params["image_height"])
    distortion = params.get("distortion", [])
    if not isinstance(distortion, list) or len(distortion) != 5:
        raise RuntimeError("Expected 5 distortion coefficients in icl.yaml")
    k1, k2, p1, p2, k3 = [float(x) for x in distortion]

    camera_text = "\n".join(
        [
            f"RadTanK3 {fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2} {k3}",
            f"{width} {height}",
            f"{fx / width} {fy / height} {cx / width} {cy / height} 0",
            f"{width} {height}",
            "",
        ]
    )
    (output / "camera.txt").write_text(camera_text, encoding="utf-8")
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
        "png_depth_scale": float(params.get("png_depth_scale", 1000)),
        "distortion": [k1, k2, p1, p2, k3],
    }


def write_association_files(output: Path, frames: List[FrameEntry]) -> None:
    rgb_lines = ["# timestamp filename", "# generated from embedding_map export"]
    depth_lines = ["# timestamp filename", "# generated from embedding_map export"]
    assoc_lines = [
        "# timestamp rgb_filename depth_timestamp depth_filename",
        "# generated from embedding_map export",
    ]
    for frame in frames:
        rgb_lines.append(f"{frame.stamp:.6f} {frame.rgb_name}")
        depth_lines.append(f"{frame.stamp:.6f} {frame.depth_name}")
        assoc_lines.append(
            f"{frame.stamp:.6f} {frame.rgb_name} {frame.stamp:.6f} {frame.depth_name}"
        )

    (output / "rgb.txt").write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")
    (output / "depth.txt").write_text("\n".join(depth_lines) + "\n", encoding="utf-8")
    (output / "rgbd_association.txt").write_text(
        "\n".join(assoc_lines) + "\n", encoding="utf-8"
    )


def write_groundtruth(output: Path, frames: List[FrameEntry], matrices: List[List[List[float]]]) -> None:
    if len(frames) != len(matrices):
        raise RuntimeError(
            f"Frame count mismatch: {len(frames)} images vs {len(matrices)} poses"
        )

    gt_lines = ["# ground truth trajectory", "# timestamp tx ty tz qx qy qz qw"]
    for frame, matrix in zip(frames, matrices):
        rotation = [row[:3] for row in matrix]
        translation = [row[3] for row in matrix]
        qx, qy, qz, qw = matrix_to_quaternion(rotation)
        gt_lines.append(
            f"{frame.stamp:.6f} {translation[0]:.9f} {translation[1]:.9f} {translation[2]:.9f} "
            f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"
        )
    (output / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")


def build_frames(sequence: SequenceData) -> List[FrameEntry]:
    if len(sequence.rgb_entries) != len(sequence.depth_entries):
        raise RuntimeError(
            f"RGB/depth count mismatch: {len(sequence.rgb_entries)} RGB vs {len(sequence.depth_entries)} depth images"
        )

    frames: List[FrameEntry] = []
    for rgb_name, depth_name in zip(
        sequence.rgb_entries, sequence.depth_entries
    ):
        rgb_entry = parse_frame_filename(rgb_name)
        depth_entry = parse_frame_filename(depth_name)
        if rgb_entry.stem.startswith("frame_"):
            rgb_stamp = rgb_entry.stem[len("frame_") :]
        else:
            rgb_stamp = rgb_entry.stem
        if depth_entry.stem != rgb_stamp:
            raise RuntimeError(
                f"RGB/depth filename mismatch: {rgb_name} vs {depth_name}"
            )
        if abs(rgb_entry.stamp - depth_entry.stamp) > 1e-6:
            raise RuntimeError(
                f"RGB/depth timestamp mismatch: {rgb_name} vs {depth_name}"
            )
        frames.append(
            FrameEntry(
                stamp=rgb_entry.stamp,
                stem=rgb_entry.stem,
                rgb_name=rgb_entry.rgb_name,
                depth_name=depth_entry.depth_name,
            )
        )
    return frames


def write_image_files(
    output: Path,
    sequence: SequenceData,
    frames: List[FrameEntry],
    link_mode: str,
    source_path: Path,
) -> None:
    rgb_dir = output / "rgb"
    depth_dir = output / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    if source_path.is_file() and tarfile.is_tarfile(source_path):
        with tarfile.open(source_path, "r:gz") as tar:
            rgb_member_prefix = f"{sequence.sequence_name}/rgb/"
            depth_member_prefix = f"{sequence.sequence_name}/depth/"
            for frame, rgb_name, depth_name in zip(
                frames, sequence.rgb_entries, sequence.depth_entries
            ):
                rgb_member = tar.extractfile(f"{rgb_member_prefix}{rgb_name}")
                depth_member = tar.extractfile(f"{depth_member_prefix}{depth_name}")
                if rgb_member is None or depth_member is None:
                    raise RuntimeError(f"Missing image member while writing {frame.stem}")
                (rgb_dir / Path(frame.rgb_name).name).write_bytes(rgb_member.read())
                (depth_dir / Path(frame.depth_name).name).write_bytes(depth_member.read())
    else:
        root = locate_sequence_dir(source_path)
        for frame, rgb_name, depth_name in zip(
            frames, sequence.rgb_entries, sequence.depth_entries
        ):
            src_rgb = root / "rgb" / rgb_name
            src_depth = root / "depth" / depth_name
            dst_rgb = rgb_dir / Path(frame.rgb_name).name
            dst_depth = depth_dir / Path(frame.depth_name).name
            materialize_file(src_rgb, dst_rgb, link_mode)
            materialize_file(src_depth, dst_depth, link_mode)


def write_manifest(
    output: Path,
    source: Path,
    sequence: SequenceData,
    frames: List[FrameEntry],
    camera_info: dict,
) -> None:
    manifest = {
        "source": str(source),
        "sequence_name": sequence.sequence_name,
        "num_frames": len(frames),
        "camera": camera_info,
        "files": {
            "rgb": "rgb.txt",
            "depth": "depth.txt",
            "association": "rgbd_association.txt",
            "groundtruth": "groundtruth.txt",
            "camera": "camera.txt",
        },
    }
    (output / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_source(source: Path) -> SequenceData:
    if source.is_file() and tarfile.is_tarfile(source):
        return load_sequence_from_tar(source)
    if source.is_dir():
        return read_sequence_from_dir(source)
    raise RuntimeError(f"Unsupported source path: {source}")


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    sequence = load_source(source)
    repo_root = Path(__file__).resolve().parents[1]
    if args.output is None:
        suffix = ""
        for part in source.parts[::-1]:
            if re.fullmatch(r"\d+hz", part):
                suffix = f"_{part}"
                break
        output = repo_root / "dataset" / "TUM" / f"rgbd_dataset_{sequence.sequence_name}{suffix}"
    else:
        output = Path(args.output).expanduser().resolve()

    ensure_output_dir(output, args.overwrite)
    frames = build_frames(sequence)
    camera_info = write_camera_file(output, sequence.camera_yaml)
    write_association_files(output, frames)
    write_groundtruth(output, frames, parse_poses(sequence.poses_sim))
    write_image_files(output, sequence, frames, args.link_mode, source)
    write_manifest(output, source, sequence, frames, camera_info)

    print(f"Converted {len(frames)} frames from {source} into {output}")


if __name__ == "__main__":
    main()
