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
   - Premiere Pro C++ SDK (built against 26.0) → `sdk\premiere` (contains `Examples\`)
3. `deps\setup_opencv.bat` and `deps\setup_nvof.bat`, then `plugin\build.bat`, then `plugin\install.bat` (once, as admin).
4. Prototype: `cd prototype`, `python -m venv .venv`, `.venv\Scripts\pip install -r requirements.txt`,
   `python make_test_clip.py`, `python validate.py`, `python validate_plugin.py`.

## Results so far (synthetic clip, PSNR vs a real 180° shutter, higher is better)

| Method | PSNR |
|---|---|
| Premiere frame dropping | 28.1 dB |
| 2-frame blend (Echo / nested-sequence trick) | 24.1 dB |
| OneEighty optical-flow blur | 38.8 dB |
