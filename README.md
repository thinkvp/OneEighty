# OneEighty Motion Blur

A Premiere Pro effect that gives high-frame-rate footage the motion blur of a
**180° shutter** when it plays at 100% speed on a lower-rate timeline, and
evens out the stutter of uneven conversions like **59.94p → 25p**.

## The problem

B-roll is often shot at 50p or 59.94p with a 180° shutter so it can be slowed
down later. Used at normal speed on a 25p timeline, Premiere just drops frames:
each frame keeps its short 1/100 s (or 1/120 s) exposure, which looks like a
90° shutter: choppy, staccato motion. With 59.94p it also shows source frames
2, 3, 2, 3… apart, so pans advance in uneven steps.

OneEighty reads the in-between source frames, estimates per-pixel motion with
optical flow and smears each pixel along its path to add the exposure a 180°
shutter at 25p would have recorded. With **Exact frame timing** it also builds
each output frame at its true 1/25 s moment, so motion advances evenly.

## Install

1. Download `OneEighty-v…-win64.zip` from [Releases](https://github.com/thinkvp/OneEighty/releases) and unzip it.
2. Close Premiere Pro, After Effects and Media Encoder.
3. Double-click **Install OneEighty.bat** and allow it to make changes.
4. In Premiere: **Effects › Video Effects › OneEighty › OneEighty Motion Blur**.

**Requirements:** Windows 10/11 64-bit and Premiere Pro (tested with Premiere Pro
2026 / 26.x). An NVIDIA GeForce RTX 20-series or newer GPU is strongly recommended;
without one the effect still works, on the CPU, but much more slowly.

## Using it

Drop the effect on a 50p / 59.94p clip (at 100% speed) in a 25p sequence.

| Control | |
|---|---|
| **Shutter angle** | The blur to simulate at the timeline rate. 180° is the classic film look; 360° is the maximum smear. |
| **Quality** | **Normal** (default): best balance. **High**: cleaner edges where people pass in front of the background, slower. **Fast**: quickest (all on the GPU) but can shimmer on fine repeating patterns such as checked shirts. |
| **Exact frame timing** | On (default): even motion for 59.94p → 25p. Makes no difference for 50p → 25p. |
| *Advanced* | Source shutter angle and frame rate (auto by default), a **motion-vector view** that shows what the effect is tracking, Force CPU, and a debug log (`%TEMP%\OneEighty.log`). |

**Tips**

- It isn't real time: render the work area to preview (**Sequence › Render In to Out**).
- Long-GOP camera files (H.264/HEVC, especially 10-bit 4:2:2 like Sony XAVC) are
  slow for Premiere to decode, and that is most of the render time. **ProRes
  proxies** make previews about 4–5× faster.
- For 10-bit, log or graded footage, tick **Render at Maximum Depth** when
  exporting: the effect then works in 32-bit float end to end.
- Problem area on a shot (e.g. an edge that tears)? Mask the effect in Premiere,
  or try Quality **High**.

**Speed** (RTX 4070, UHD 59.94p on a 25p timeline, per frame): about **0.27 s**
with ProRes source at Normal quality, about 1.2 s with long-GOP camera files.

## How well does it work?

Measured on synthetic clips with a known-correct answer (what a real 25p camera
at 180° records), PSNR, higher is better:

| 59.94p → 25p | Accuracy | Pan-step unevenness (judder) |
|---|---|---|
| Premiere (frame dropping) | 24.9 dB | 9.8 px |
| OneEighty, exact frame timing off | 30.7 dB | 9.8 px |
| **OneEighty, Normal, exact timing on** | **38.6 dB** | **0.14 px** |

| 50p → 25p | Accuracy |
|---|---|
| Premiere (frame dropping) | 28.1 dB |
| 2-frame blend (Echo / nested-sequence trick) | 24.1 dB |
| **OneEighty** | **38.8–39.2 dB** |

## Known limitations

- **Edges:** where a subject moves against a differently moving background,
  optical flow can't know which is in front, so edges can smear or tear slightly.
  Quality High helps; masks are the fallback.
- **Speed:** frame fetching from long-GOP files dominates; use proxies to preview.
- **Exact frame timing** assumes a constant source frame rate and a clip in-point
  on a source frame; it is not applied to reversed clips.
- Tested in **Premiere Pro on Windows with an NVIDIA RTX 4070**. The plugin also
  loads in After Effects, but that's untested. No macOS build.

## Building from source

See [plugin/README.md](plugin/README.md) for how it works and the measurements.
In short:

1. Visual Studio 2022 Build Tools (C++ workload) and the CUDA Toolkit 13.x.
2. The Adobe After Effects SDK (free from [developer.adobe.com](https://developer.adobe.com/after-effects/),
   Adobe licence, not included) unzipped to `sdk\ae` (containing `Examples\`).
3. `deps\setup_opencv.bat` and `deps\setup_nvof.bat` (fetch static OpenCV via
   vcpkg and the NVIDIA Optical Flow headers).
4. `plugin\build.bat` → `plugin\build\OneEighty.aex`. For development,
   `plugin\install.bat` (once, as admin) links Premiere's plug-in folder to the
   build folder.
5. Release zip: `python release\make_release.py` → `dist\`.

| Folder | |
|---|---|
| `plugin/` | The effect (C++): After Effects SDK host code, CUDA engine (`gpu/`), CPU maths (`MotionBlur.cpp`), offline test tools (`tools/`) |
| `prototype/` | Python reference implementation, synthetic ground-truth clips and the validation scripts behind the numbers above |
| `release/` | Installer scripts and release packaging |
| `deps/` | Dependency setup scripts |

## Licence

MIT, see [LICENSE](LICENSE). The release binary includes OpenCV, zlib, the NVIDIA
CUDA runtime and NVIDIA Optical Flow SDK headers; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Not affiliated with Adobe or
NVIDIA.
