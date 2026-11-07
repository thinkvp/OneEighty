# OneEighty Motion Blur (Premiere Pro / After Effects effect)

v0.2: optical-flow shutter blur. Estimates per-pixel motion between the current
source frame and its neighbours (Premiere lets the effect read the in-between
50p/59.94p frames on a 25p timeline) and smears each pixel along its path to add
the exposure a 180° shutter at the timeline rate would have recorded.

Accuracy on the synthetic test (`../prototype/validate_plugin.py`): 38.8 dB vs a
real 180° shutter, against 28.1 dB for Premiere's frame dropping.

## Build / install

```
..\deps\setup_opencv.bat   # once: vcpkg + static OpenCV (core/imgproc/video)
build.bat                  # -> build\OneEighty.aex, build\oe_offline.exe
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
| Motion analysis | Resolution of the motion estimate: Fast 480 px · Normal 960 px · High 1920 px wide |
| Write debug log | One line per rendered frame (timings, frame rates) in `%TEMP%\OneEighty.log` |

## Performance

CPU path (OpenCV DIS optical flow, 24 threads) — used in the current Premiere build:

| | Motion analysis | Blur | Accuracy (720p test) |
|---|---|---|---|
| 1280×720 | ~60 ms | ~25 ms | 38.8 dB |
| 3840×2160 | ~75 ms | ~325 ms | |

GPU path (`gpu/`, NVIDIA Optical Flow hardware + CUDA, RTX 4070) — offline only so far,
Premiere integration needs the Premiere Pro SDK's GPU filter API:

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
