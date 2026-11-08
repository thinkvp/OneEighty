# OneEighty Motion Blur: developer notes

Gives high-frame-rate footage (50p, 59.94p) on a lower-rate timeline (25p) the
motion blur a 180° shutter at the timeline rate would have recorded, and evens
out the uneven frame steps of non-integer conversions like 59.94p → 25p. Works
whenever Premiere would drop frames: clips at 100% speed, and slowed-down files
(e.g. S&Q 25p files holding 50p motion) sped back up.

## Effect controls

| Control | Meaning |
|---|---|
| **Shutter angle** | Look to simulate at the timeline rate (default 180°; 360° = maximum smear). Slider 90–360 (typed 0–720): at or below the footage's own exposure (source angle × timeline fps ÷ source frames per timeline second) `BlurHalfWidth` returns 0. |
| **Quality** | **Normal** (default) / **High**: motion estimated on the CPU with OpenCV DIS at 960 px wide (FAST / MEDIUM preset; High ~2.4× the motion time, cleaner edges where people pass in front of the background), blur on the GPU. **Extra High**: the shutter integrated from 6 RIFE in-between frames (see below); falls back to High without CUDA, Vulkan or the model. Appended to the popup, so saved Normal / High settings are unchanged. |
| **Exact frame timing** | On (default): each output frame is built at its exact timeline moment from the two source frames either side, so motion advances evenly (removes the 2/3-frame judder of 59.94p → 25p). Off: Premiere's own frame choice, blur only. No visible difference for 50p → 25p. |
| *Advanced* ▸ Source shutter angle | Shutter the footage was shot with (default 180°). |
| *Advanced* ▸ Source fps (0 = auto) | The clip's frame rate before any speed change; auto reads Premiere's original clip rate. |
| *Advanced* ▸ View | Result · Motion vectors (hue = direction, colour strength = speed, full at 1% of frame width per source frame; still areas grey). Shows why a shot artefacts. |
| *Advanced* ▸ Force CPU | Do everything on the CPU (troubleshooting). |
| *Advanced* ▸ Write debug log | One line per rendered frame in `%TEMP%\OneEighty.log`: path (`hybrid` = CPU motion + GPU blur, `extra` = RIFE, `cpu`), frame formats, the frame rate / speed Premiere reports and source frames per output frame, timing plan and frame offset, how the fetched frames compare with Premiere's own, the fetch times, motion and total ms. |
| *Advanced* ▸ Updates [Check for updates] | Checks now and shows the result in a Windows message box (offers to open the release page if newer). When a newer release is known, the Advanced group and this button are renamed ("Advanced - update available", "Update vX available") via `PF_UpdateParamUI`. |

"Advanced" is a twirl-down group; Premiere ignores the start-collapsed flag, so
it shows open there. Parameter names are limited to 31 characters.

Saved projects identify controls by disk ID (`OneEightyParams.h`); retired IDs
are never reused. v0.8 replaced the Fast | Normal | High popup (ID 6) with
Normal | High (ID 15), so clips saved on Fast open on Normal.

**Update check** (`UpdateCheck.cpp`): about once a day, one anonymous GET of
`api.github.com/repos/thinkvp/OneEighty/releases/latest` on a background thread
(User-Agent `OneEighty-UpdateCheck/<version>`, nothing else sent); the result is
cached in `%LOCALAPPDATA%\OneEighty\update.txt`. Opt-out for studios / IT:
environment variable `ONEEIGHTY_DISABLE_UPDATE_CHECK=1`, or DWORD
`DisableUpdateCheck=1` under `HKCU` or `HKLM\Software\thinkvp\OneEighty` (the
button always works).

The GPU is used automatically when an NVIDIA GPU (RTX 20-series / Turing or
newer) and driver are present; otherwise everything runs on the CPU.

**Extra High** (`rife/`): for each output frame, 6 samples are spread evenly
over the shutter window around the output time; each sample is an in-between
frame from the RIFE v4.6 network (Practical-RIFE, MIT) between the two source
frames around its time (the frame before A / after B are fetched when the
window reaches them), and the samples are averaged. The network runs with ncnn
on the GPU through Vulkan (`RifeFlow.cpp`, which also carries RIFE's custom
Warp layer). It only supplies the motion and the blend mask (the network's own
last step is `mask × warp(frame0) + (1 − mask) × warp(frame1)`): the warping and
blending are done in CUDA on the original frames (`KRifeAccum`), so 10-bit /
log / 32-bit float footage keeps full precision and values above 1.0 pass
through (float and 8-bit renders agree to 102 dB; input ×1.5 → output max
1.500). RIFE sees an RGB copy clamped to 0..1, scaled to at most ~2.1 MP.

## How it works

1. **Fetch.** The effect checks out its input at sub-frame times through the
   After Effects API, which Premiere honours: on a 25p timeline it can read the
   in-between 50p/59.94p source frames. (Premiere's GPU filter API cannot: it
   floors frame requests to whole timeline frames, so that route was dropped
   in v0.5.) On a speed-changed clip Premiere gives the effect times in the
   clip's own time, so frames are requested one source frame apart (clip
   rate), while the blur uses source frames per timeline second (clip rate ×
   speed).
2. **Timing.** Exact frame timing: output time p (in source frames) lies
   between frames A = floor(p) and B = A+1. Off: Premiere's frame and its
   neighbours one source interval away. On some speed-changed clip instances
   the computed A is one source frame away from the frame Premiere itself
   shows (measured: a 30p file at 200%, where it hid the frame of a ball's
   bounce). So on frames that sit exactly on a source frame the effect
   compares Premiere's frame with A, learns the offset (±1) per effect
   instance and shifts the plan; the debug log shows `offset` / `anchor`.
3. **Motion.** Dense optical flow A→B and B→A (or cur→prev/next) with OpenCV
   DIS on the CPU, at 960 px wide. DIS motion is smooth, so a fast object's
   motion spills 20–40 px past its outline and the blur would smear the
   background next to it (a court line beside a bouncing ball). Near motion
   boundaries each pixel therefore takes the colour-weighted median motion of
   nearby pixels (`RefineFlowEdges` in `MotionBlur.cpp`): background and line
   pixels side with the background around them, object pixels with the
   object. About +20 ms per UHD frame.
4. **Blur.** Each pixel is averaged along its motion path over the missing
   shutter time (long smears sampled from downscaled copies), A blurred at the
   output time and B likewise, blended by closeness. CUDA when available
   (`gpu/GpuMotionBlur.cu`), else OpenCV (`MotionBlur.cpp`).

**Bit depth.** The effect declares 32-bit float BGRA to Premiere (preferred),
then 8-bit BGRA, so it shows Premiere's 32-bit badge and doesn't force a clip
into an 8-bit pipeline: no 8-bit banding on log/graded footage, and HDR values
above 1.0 pass through unclipped. In float the GPU path copies frames straight
into its float buffers; the CPU maths runs in float too. Motion is always
estimated on an 8-bit greyscale copy. Float and 8-bit renders of the same frame
agree to about 60 dB (`../prototype/validate_float.py`).

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
| Blur only (exact timing off) | 30.5 dB | 26.0 dB | 9.8 px |
| Exact timing, Normal | 38.1 dB | 35.0 dB | 0.16 px |
| Exact timing, High | 38.6 dB | 35.4 dB | 0.14 px |
| Exact timing, Extra High | 40.2 dB | 37.2 dB | 0.24 px |

50p → 25p: Premiere 28.1 dB, Normal 38.8 dB, High 39.2 dB, Extra High 40.5 dB.
GPU and CPU paths agree (`validate_cadence.py`). Extra High offline at UHD
(RTX 4070): ~0.20 s RIFE + 0.03 s warping per frame (High: ~0.08 s). Samples:
8 scored 40.4 / 40.5 dB (59.94p / 50p) and look the same as 6 on a fast ball; 4
drop to 39.8 dB on 59.94p. Shrinking RIFE's analysis size below ~2 MP at UHD
costs 0.9-1.5 dB, so it stays. Speed-ups that didn't change the output: RIFE's
time inputs are cached on the GPU, the motion map goes to CUDA in ncnn's own
layout (no conversion), input preparation runs on all cores and outside the
GPU lock (Premiere renders frames in parallel).

Premiere 26, RTX 4070, UHD on a 25p timeline, per frame:

| Source | Quality | Fetch | Motion | Total |
|---|---|---|---|---|
| ProRes 59.94p | **Normal** | 0.14 s | 0.08 s | **0.27 s** |
| ProRes 59.94p | High | 0.13 s | 0.28 s | 0.45 s |
| Long-GOP 10-bit 4:2:2 (XAVC) | Normal | ~1 s | 0.06–0.1 s | ~1.2 s |

Fetching dominates on long-GOP footage. Of the two frames fetched per output
frame, A is Premiere's own frame and comes from its cache (~35 ms); B is the
in-between frame Premiere would have dropped and has to be decoded just for
the effect (0.8–1.3 s). Reusing Premiere's input frame instead of fetching A
was tried and saved only those ~35 ms. ProRes or proxies are the fix.

DIS motion per UHD frame pair at 960 px (offline, 24 threads): MEDIUM 41 ms,
FAST 17 ms; on the synthetic 59.94p test FAST costs 0.4 dB vs MEDIUM. Running
the two directions in parallel gave no gain (DIS already uses all cores).

**Tried, not adopted:**
- *NVIDIA optical-flow hardware for motion* (the "Fast" quality up to v0.7):
  scored well on smooth synthetic textures but mis-tracked fine repeating
  patterns (checked shirts), and wasn't faster in Premiere because fetching
  dominates.
- *Edge reconstruction filters* (McGuire 2012 style): 0.3–1.7 dB *worse* than
  the plain gather; gated to motion boundaries only +0.1–0.2 dB
  (`../prototype/validate_edges.py`). Proper edge handling needs to know which
  object is in front (RSMB Pro asks for a matte); Extra High handles it
  instead by building real in-between frames.
- *Three-frame motion with occlusion handling* (motion from the frame before A
  too, for a two-sided blur and for background hidden on one side): a fast
  object's streak ended in a hard edge wherever a pixel's motion switched;
  per-pixel motion can't draw an object's streak over a sharp background.

## Build / install

```
..\deps\setup_opencv.bat   # once: vcpkg + static OpenCV (core/imgproc/video) + ncnn (Vulkan)
..\deps\setup_rife.bat     # once: RIFE v4.6 model -> ..\deps\rife (build.bat copies it next to the .aex)
build.bat                  # -> build\OneEighty.aex + offline test tools
install.bat                # once, as admin: junction Adobe\Common\Plug-ins\7.0\MediaCore\OneEighty -> build\
```
Needs the VS 2022 Build Tools, the CUDA toolkit (v13.4 or `CUDA_PATH`) and the
After Effects SDK in `..\sdk\ae`. At runtime only the NVIDIA driver is needed
for the GPU path (Vulkan for Extra High comes with it); the plugin loads and
runs CPU-only without it. Close Premiere
before rebuilding (it locks the .aex); restart it to load the new build.

## Offline tests

Raw BGRA frames in and out, processed as the effect does (`build\`):

- `oe_offline in.bgra W H srcFps outFps out.bgra [srcShutter targetShutter [mipThreshold]]`: CPU path
- `oe_gpu_offline in.bgra W H srcFps outFps out.bgra [mipThreshold]`: GPU path (DIS motion, CUDA blur)
- `oe_update_test`: the update check against GitHub

Environment: `OE_EXACT=0` exact frame timing off; `OE_QUALITY=high` / `extra`
(`OE_SAMPLES=n`, `OE_RIFE_MODEL=dir`); `OE_EDGEFIX=0`
skip the motion-boundary clean-up; `OE_FLOAT=1`
32-bit float (`2` on the GPU tool: input ×1.5 to check values above 1.0 pass
through); GPU tool only: `OE_ARGB=1` ARGB bytes, `OE_VIEW=1` motion-vector view.

`../prototype/validate_plugin.py`, `validate_cadence.py`, `validate_float.py`,
`validate_timing.py` and `validate_edges.py` score them against ground truth.
