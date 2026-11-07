# OneEighty Motion Blur (Premiere Pro / After Effects effect)

v0.3: optical-flow shutter blur. Estimates per-pixel motion between the current
source frame and its neighbours (Premiere lets the effect read the in-between
50p/59.94p frames on a 25p timeline) and smears each pixel along its path to add
the exposure a 180° shutter at the timeline rate would have recorded.

Two render paths in one plugin file:

| | When | Files |
|---|---|---|
| **GPU** | Premiere, project renderer = *Mercury Playback Engine GPU Acceleration (CUDA)*, NVIDIA GPU with optical-flow hardware (Turing or newer) | `OneEightyGPU.cpp` (Premiere GPU filter), `gpu/GpuMotionBlur.cu` |
| **CPU** | everything else (OpenCL/DirectX renderer, other GPUs, After Effects, "Blend 3 frames" mode) | `OneEighty.cpp` (AE API effect), `MotionBlur.cpp` |

The GPU filter asks Premiere for the frames one source-frame interval either side
(`GetFrameDependencies`); Premiere decodes them and passes them in GPU memory.

Accuracy on the synthetic test: CPU 38.8 dB, GPU 39.2–39.5 dB vs a real 180°
shutter, against 28.1 dB for Premiere's frame dropping.

## Build / install

```
..\deps\setup_opencv.bat   # once: vcpkg + static OpenCV (core/imgproc/video)
..\deps\setup_nvof.bat     # once: NVIDIA Optical Flow headers
build.bat                  # -> build\OneEighty.aex, build\oe_offline.exe, build\oe_gpu_offline.exe
install.bat                # once, as admin: junction Adobe\Common\Plug-ins\7.0\MediaCore\OneEighty -> build\
```
Close Premiere before rebuilding (it locks the .aex); restart it to load the new build.

## Effect controls

| Control | Meaning |
|---|---|
| Mode | **Optical flow blur** · Blend 3 frames (test) · Show previous / next source frame · Off |
| Source fps (0 = auto) | Clip's real frame rate; auto reads the original clip rate from Premiere |
| Source shutter angle | Shutter the footage was shot with (default 180°) |
| Target shutter angle | Look to simulate at the timeline rate (default 180°; 360° = maximum smear) |
| Motion analysis | CPU: analysis width 480 / 960 / 1920 px. GPU: ¼ width grid 4 / ½ width grid 2 / ½ width grid 1 |
| Write debug log | One line per rendered frame (timings, frame rates, `[gpu]` or CPU path) in `%TEMP%\OneEighty.log` |

## Performance

In Premiere 26 on an RTX 4070 (UHD 59.94p clip on a 25p timeline, 1,297 GPU frames,
before the popup fix so analysis ran at "Fast"): median **4.1 ms** per frame,
95th percentile 7.6 ms, no fallbacks or errors.

CPU path (OpenCV DIS optical flow, 24 threads):

| | Motion analysis | Blur | Accuracy (720p test) |
|---|---|---|---|
| 1280×720 | ~60 ms | ~25 ms | 38.8 dB |
| 3840×2160 | ~75 ms | ~325 ms | |

GPU path (`gpu/`, NVIDIA Optical Flow hardware + CUDA, RTX 4070), offline harness:

| | Analysis width / NVOF grid | Total per frame | Accuracy (720p test) |
|---|---|---|---|
| 1280×720 | 640 / 2 | 2.4 ms | 39.5 dB |
| 3840×2160 | 960 / 2 | 7.3 ms | |
| 3840×2160 | 1920 / 4 | 8.2 ms | |
| 3840×2160 | 1920 / 2 | 13.9 ms | |

Needs the CUDA toolkit to build (`build.bat` expects v13.4 or `CUDA_PATH`) and
`..\deps\setup_nvof.bat` for the NVIDIA Optical Flow headers; at runtime only the
NVIDIA driver (Turing or newer).

## Offline tests

- `oe_offline in.bgra W H srcFps outFps out.bgra [srcShutter targetShutter [mipThreshold]]` (CPU)
- `oe_gpu_offline in.bgra W H srcFps outFps out.bgra [flowWidth grid perf mipThreshold]` (GPU)

`../prototype/validate_plugin.py` and `../prototype/validate_gpu.py` score them against ground truth.
