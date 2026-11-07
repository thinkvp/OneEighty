# OneEighty Motion Blur (Premiere Pro / After Effects effect)

v0.6. Gives high-frame-rate footage (50p, 59.94p) played at 100% speed on a
lower-rate timeline (25p) the motion blur a 180° shutter at the timeline rate
would have recorded, and (optionally) evens out the uneven frame steps of
non-integer conversions like 59.94p → 25p.

## Effect controls

| Control | Meaning |
|---|---|
| **Shutter angle** | Look to simulate at the timeline rate (default 180°; 360° = maximum smear). |
| **Quality** | **Fast**: motion and blur on the GPU (NVIDIA optical-flow hardware). Quickest, but can mis-track fine repeating patterns (checked shirts). **Normal** (default): motion estimated on the CPU with OpenCV DIS (FAST preset, 960 px wide), blur on the GPU. **High**: DIS MEDIUM preset; ~2.4× the motion time, more robust on fine patterns. |
| **Exact frame timing** | On (default): each output frame is built at its exact timeline moment from the two source frames either side, so motion advances evenly (removes the 2/3-frame judder of 59.94p → 25p). Off: Premiere's own frame choice, blur only. No visible difference for 50p → 25p. |
| *Advanced* ▸ Source shutter angle | Shutter the footage was shot with (default 180°). |
| *Advanced* ▸ Source fps (0 = auto) | Clip's real frame rate; auto reads the original clip rate from Premiere. |
| *Advanced* ▸ View | Result · Motion vectors (hue = direction, colour strength = speed, full at 1% of frame width per source frame; still areas grey). Shows why a shot artefacts. |
| *Advanced* ▸ Force CPU | Do everything on the CPU (troubleshooting). |
| *Advanced* ▸ Write debug log | One line per rendered frame in `%TEMP%\OneEighty.log`: path (`hybrid-dis`, `hybrid-nvof`, `cpu`), frame formats, timing plan, fetch / motion / total ms. |

"Advanced" is a twirl-down group; Premiere ignores the start-collapsed flag, so it shows open there.

The GPU is used automatically when an NVIDIA GPU with optical-flow hardware
(Turing or newer) and driver are present; otherwise everything runs on the CPU.

## How it works

1. **Fetch.** The effect checks out its input at sub-frame times through the
   After Effects API, which Premiere honours: on a 25p timeline it can read the
   in-between 50p/59.94p source frames. (Premiere's GPU filter API cannot: it
   floors frame requests to whole timeline frames. That route was built, tested
   and dropped in v0.5; the code is kept in `experimental/`.)
2. **Timing.** Exact frame timing: output time p (in source frames) lies
   between frames A = floor(p) and B = A+1. Off: Premiere's frame and its
   neighbours one source interval away.
3. **Motion.** Dense optical flow A→B and B→A (or cur→prev/next): OpenCV DIS on
   the CPU (Normal/High) or the NVIDIA optical-flow hardware (Fast).
4. **Blur.** Each pixel is averaged along its motion path over the missing
   shutter time (long smears sampled from downscaled copies), A blurred at the
   output time and B likewise, blended by closeness. CUDA when available
   (`gpu/GpuMotionBlur.cu`), else OpenCV (`MotionBlur.cpp`).

**Bit depth.** The effect declares 32-bit float BGRA to Premiere (preferred),
then 8-bit BGRA, so it shows Premiere's 32-bit badge and doesn't force a clip
into an 8-bit pipeline: no 8-bit banding on log/graded footage, and HDR values
above 1.0 pass through unclipped (checked offline: input x1.5 -> output max
1.500, nothing clipped). In float the GPU path copies frames straight into its
float buffers; the CPU maths runs in float too. Motion is always estimated on
an 8-bit greyscale copy (precision doesn't matter there). Float and 8-bit
renders of the same frame agree to 60 dB (0.05% of pixels differ by more than
8 levels, at motion edges). `../prototype/validate_float.py`.

Premiere can hand the input, the checked-out frames and the output in
different formats (seen: checkouts as ARGB while input/output were BGRA); the
effect asks for checkouts in the render format and converts anything that
still differs to the output's byte order and bit depth.

## Measured

Synthetic clips with a known-correct answer (`../prototype`), PSNR vs a real
25p camera at 180°:

| 59.94p → 25p | Accuracy | Frame-to-frame | Pan-step variation |
|---|---|---|---|
| Premiere | 24.9 dB | 21.8 dB | 9.8 px |
| Blur only (exact timing off) | 30.7 dB | 26.1 dB | 9.8 px |
| Exact timing, Normal (DIS motion) | 38.6 dB | 35.4 dB | 0.14 px |
| Exact timing, Fast (NVIDIA motion) | 39.2 dB | 36.1 dB | 0.13 px |

(Fast scores higher on the smooth synthetic textures, but mis-tracks real
repeating patterns, hence Normal is the default.)

In Premiere 26, RTX 4070, UHD 59.94p H.264/HEVC-style source on a 25p timeline,
per frame:

| | Fetch frames from Premiere | Motion | GPU blur etc. | Total |
|---|---|---|---|---|
| CPU only (v0.4) | 0.8 s | 2.6 s (motion + blur) | | 3.3 s |
| **Normal** | 1.1 s | 0.18 s | 0.04 s | **1.35 s** |
| High (v0.5: DIS MEDIUM @1920; now @960) | 1.1 s | 0.96 s | 0.04 s | 2.15 s |
| Fast | 0.6 s | – | 0.05 s | 0.68 s |

Frame fetching (Premiere decoding the extra source frames on the CPU) is ~80% of
that; decoding backwards in long-GOP footage is especially slow. With a ProRes
copy of the same clip fetching dropped to 0.12–0.14 s and the total to
0.45–0.5 s per frame.

With ProRes source and v0.5.1 (checked-out frames now arrive in the render
pixel format, so no byte reordering), medians per UHD frame:

| Quality | Fetch | Motion | Total |
|---|---|---|---|
| **Normal** (DIS FAST) | 0.14 s | 0.08 s | **0.27 s** |
| High (DIS MEDIUM) | 0.13 s | 0.28 s | 0.45 s |

Both kept the fine checked shirt clean; High gave cleaner edges where people
pass in front of the background.

DIS motion per UHD frame pair at 960 px (offline, 24 threads): MEDIUM 41 ms,
FAST 17 ms, ULTRAFAST 7 ms; on the synthetic 59.94p test FAST costs 0.4 dB and
ULTRAFAST 0.6 dB vs MEDIUM. Running the two directions in parallel gave no gain
(DIS already uses all cores).

**Edges (tried, not adopted):** depth-free "reconstruction filter" variants
(McGuire 2012 style) were 0.3–1.7 dB *worse* than the plain gather, a version
gated to motion boundaries only +0.1–0.2 dB (`../prototype/validate_edges.py`).
Proper edge handling needs to know which object is in front (RSMB Pro asks for
a matte); an effect mask on OneEighty is the practical workaround.

## Build / install

```
..\deps\setup_opencv.bat   # once: vcpkg + static OpenCV (core/imgproc/video)
..\deps\setup_nvof.bat     # once: NVIDIA Optical Flow headers
build.bat                  # -> build\OneEighty.aex, build\oe_offline.exe, build\oe_gpu_offline.exe
install.bat                # once, as admin: junction Adobe\Common\Plug-ins\7.0\MediaCore\OneEighty -> build\
```
Needs the VS 2022 Build Tools, the CUDA toolkit (v13.4 or `CUDA_PATH`) and the
After Effects SDK in `..\sdk\ae`. At runtime only the NVIDIA driver is needed
for the GPU path; the plugin loads and runs CPU-only without it. Close Premiere
before rebuilding (it locks the .aex); restart it to load the new build.

## Offline tests

- `oe_offline in.bgra W H srcFps outFps out.bgra [srcShutter targetShutter [mipThreshold]]` (CPU)
- `oe_gpu_offline in.bgra W H srcFps outFps out.bgra [flowWidth grid perf mipThreshold]` (GPU)
  - `OE_CADENCE=1|2|3|4`: single-frame shift / two-frame / hybrid host frames / hybrid with DIS motion
  - `OE_VIEW=1` motion-vector view, `OE_ARGB=1` feed the hybrid path ARGB bytes

`../prototype/validate_plugin.py`, `validate_gpu.py`, `validate_cadence.py`,
`validate_timing.py` and `validate_edges.py` score them against ground truth.
