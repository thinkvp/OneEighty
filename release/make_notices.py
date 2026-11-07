"""Regenerate THIRD_PARTY_NOTICES.md from the licence files of the dependencies
that are compiled into OneEighty.aex. Run after deps/setup_*.bat:

    python release/make_notices.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARE = ROOT / "deps" / "vcpkg" / "installed" / "x64-windows-static-md" / "share"
NVOF = ROOT / "deps" / "nvof" / "nvOpticalFlowCommon.h"


def nvof_licence():
    text = NVOF.read_text(encoding="utf-8", errors="replace")
    block = text[text.index("/*") + 2:text.index("*/")]
    return "\n".join(line.lstrip("* ").rstrip() if line.strip() != "*" else "" for line in block.splitlines()).strip()


parts = [
    "# Third-party notices",
    "",
    "OneEighty's own code is MIT licensed (see `LICENSE`). The released plugin",
    "binary (`OneEighty.aex`) also contains the following third-party software,",
    "compiled in statically:",
    "",
    "| Component | Used for | Licence |",
    "|---|---|---|",
    "| [OpenCV](https://opencv.org) 4 (core, imgproc, video) | DIS optical flow, CPU blur | Apache 2.0 (+ notices below) |",
    "| [zlib](https://zlib.net) | dependency of OpenCV core | zlib licence |",
    "| NVIDIA CUDA Runtime (`cudart_static`) | GPU processing | NVIDIA CUDA Toolkit EULA (redistributable component) |",
    "| [NVIDIA Optical Flow SDK](https://github.com/NVIDIA/NVIDIAOpticalFlowSDK) headers | hardware optical flow (\"Fast\" quality) | BSD 3-clause |",
    "",
    "The NVIDIA Optical Flow runtime (`nvofapi64.dll`) and the CUDA driver are part",
    "of the NVIDIA display driver and are not distributed with OneEighty.",
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
    "---",
    "",
    "## NVIDIA Optical Flow SDK (headers)",
    "",
    "```",
    nvof_licence(),
    "```",
    "",
    "---",
    "",
    "## zlib",
    "",
    "```",
    (SHARE / "zlib" / "copyright").read_text(encoding="utf-8", errors="replace").strip(),
    "```",
    "",
    "---",
    "",
    "## OpenCV (Apache License 2.0, with notices for bundled components)",
    "",
    "```",
    (SHARE / "opencv4" / "copyright").read_text(encoding="utf-8", errors="replace").strip(),
    "```",
    "",
]
out = ROOT / "THIRD_PARTY_NOTICES.md"
out.write_text("\n".join(parts), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
