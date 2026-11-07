# OneEighty Motion Blur

A Premiere Pro (and After Effects) effect that gives high-frame-rate footage
(50p, 59.94p) the motion blur of a 180° shutter when it plays at 100% speed on a
lower-rate timeline (25p).

Premiere conforms 50p to 25p by dropping every other frame, so each frame keeps
its 1/100 s exposure. That's a 90° look at 25p: choppy, staccato motion. OneEighty
reads the neighbouring source frames, estimates per-pixel motion with optical
flow and smears each pixel along its path to add the missing exposure.

## Layout

| Folder | What |
|---|---|
| `plugin/` | The effect (C++, After Effects SDK + OpenCV). See [plugin/README.md](plugin/README.md). |
| `prototype/` | Python reference implementation, synthetic ground-truth test clip and scoring scripts. |
| `deps/` | Scripts that fetch third-party dependencies (vcpkg/OpenCV, NVIDIA Optical Flow headers). |
| `sdk/` | Adobe SDKs: not in git (Adobe licence), download from developer.adobe.com and unzip here. |

## Setup from a fresh clone

1. Visual Studio 2022 Build Tools with the C++ workload; CUDA Toolkit (13.x) for the GPU path.
2. Adobe SDKs, renamed to version-free folders so updating one is just replacing the folder:
   - After Effects SDK (built against 26.5) → `sdk\ae` (contains `Examples\`)
   - Premiere Pro C++ SDK (26.0) → `sdk\premiere`: optional, only for the unbuilt `plugin\experimental\` GPU filter
3. `deps\setup_opencv.bat` and `deps\setup_nvof.bat`, then `plugin\build.bat`, then `plugin\install.bat` (once, as admin).
4. Prototype: `cd prototype`, `python -m venv .venv`, `.venv\Scripts\pip install -r requirements.txt`,
   `python make_test_clip.py`, `python validate.py`, `python validate_plugin.py`.

## Results so far (synthetic clips, PSNR vs a real 25p camera at 180°, higher is better)

50p source → 25p:

| Method | PSNR |
|---|---|
| Premiere frame dropping | 28.1 dB |
| 2-frame blend (Echo / nested-sequence trick) | 24.1 dB |
| OneEighty, CPU (OpenCV optical flow) | 38.8 dB |
| OneEighty, GPU (NVIDIA optical flow) | 39.2–39.5 dB |

59.94p source → 25p (uneven 2/3-frame cadence). "Pan step variation" is how
unevenly the background moves from frame to frame (a real 25p camera: 0.02 px):

| Method | PSNR | Pan step variation |
|---|---|---|
| Premiere frame dropping | 24.9 dB | 9.8 px |
| OneEighty, exact frame timing off | 30.7 dB | 9.8 px |
| OneEighty v0.5, Quality Normal (default), exact timing on | 38.6 dB | 0.14 px |
| OneEighty v0.5, Quality Fast, exact timing on | 39.2 dB | 0.13 px |

In Premiere (RTX 4070, UHD 59.94p on 25p) the default takes ~1.35 s per frame on
long-GOP camera files (mostly Premiere decoding the extra source frames) and
~0.27 s on ProRes copies or proxies; see `plugin/README.md`.

Scripts: `prototype/validate.py`, `validate_plugin.py`, `validate_gpu.py`,
`validate_cadence.py`, `validate_timing.py`, `validate_edges.py`.
