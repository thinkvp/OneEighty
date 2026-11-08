"""Regenerate THIRD_PARTY_NOTICES.md from the licence files of the dependencies
that are compiled into OneEighty.aex or shipped with it. Run after
deps/setup_opencv.bat:

    python release/make_notices.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARE = ROOT / "deps" / "vcpkg" / "installed" / "x64-windows-static-md" / "share"
THIRD = ROOT / "third_party"


def text(path):
    return Path(path).read_text(encoding="utf-8", errors="replace").strip()


def block(title, body):
    return ["---", "", f"## {title}", "", "```", body, "```", ""]


parts = [
    "# Third-party notices",
    "",
    "OneEighty's own code is MIT licensed (see `LICENSE`). The released plugin",
    "binary (`OneEighty.aex`) also contains the following third-party software,",
    "compiled in statically, and ships with the RIFE model files:",
    "",
    "| Component | Used for | Licence |",
    "|---|---|---|",
    "| [OpenCV](https://opencv.org) 4 (core, imgproc, video) | DIS optical flow, CPU blur | Apache 2.0 (+ notices below) |",
    "| [zlib](https://zlib.net) | dependency of OpenCV core | zlib licence |",
    "| NVIDIA CUDA Runtime (`cudart_static`) | GPU processing | NVIDIA CUDA Toolkit EULA (redistributable component) |",
    "| [ncnn](https://github.com/Tencent/ncnn) | running the RIFE network on the GPU (Vulkan), Quality Extra High | BSD 3-clause (+ notices below) |",
    "| [glslang](https://github.com/KhronosGroup/glslang) | compiling ncnn's GPU shaders | BSD 3-clause and others (below) |",
    "| [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan) | RIFE's Warp layer shaders and the ncnn conversion of the model (`rife-v4.6`) | MIT |",
    "| [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) v4.6 model | in-between frames for Quality Extra High | MIT |",
    "",
    "The CUDA driver and the Vulkan loader (`vulkan-1.dll`) are part of the",
    "graphics driver and are not distributed with OneEighty.",
    "",
    "Building from source also needs the Adobe After Effects SDK, which is",
    "licensed separately by Adobe and is not included in this repository or in",
    "the release.",
    "",
    "---",
    "",
    "## NVIDIA CUDA Runtime",
    "",
    "This software contains the NVIDIA CUDA Runtime library, a redistributable",
    "component of the NVIDIA CUDA Toolkit, distributed under the NVIDIA CUDA",
    "Toolkit End User License Agreement: https://docs.nvidia.com/cuda/eula/",
    "",
]
parts += block("RIFE model (Practical-RIFE, MIT)", text(THIRD / "rife-model" / "LICENSE-Practical-RIFE"))
parts += block("rife-ncnn-vulkan (MIT)", text(THIRD / "rife-ncnn-vulkan" / "LICENSE"))
parts += block("ncnn", text(SHARE / "ncnn" / "copyright"))
parts += block("glslang", text(SHARE / "glslang" / "copyright"))
parts += block("zlib", text(SHARE / "zlib" / "copyright"))
parts += block("OpenCV (Apache License 2.0, with notices for bundled components)", text(SHARE / "opencv4" / "copyright"))
out = ROOT / "THIRD_PARTY_NOTICES.md"
out.write_text("\n".join(parts), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
