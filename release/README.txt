OneEighty Motion Blur {VERSION}
==============================

Adds the motion blur of a 180-degree shutter to high-frame-rate footage
(50p, 59.94p) played at 100% speed on a lower-rate timeline (25p), and evens
out the stutter of uneven conversions like 59.94p -> 25p.

https://github.com/thinkvp/OneEighty


REQUIREMENTS
  - Windows 10 or 11 (64-bit)
  - Adobe Premiere Pro (tested with Premiere Pro 2026 / 26.x)
  - Recommended: NVIDIA GeForce RTX 20-series or newer with a current driver
    (the effect falls back to the CPU otherwise - correct, but much slower)
  - The Microsoft Visual C++ runtime (installed with Premiere Pro)


INSTALL
  1. Close Premiere Pro, After Effects and Media Encoder.
  2. Double-click "Install OneEighty.bat" and allow it to make changes.
  3. Start Premiere Pro: Effects > Video Effects > OneEighty >
     OneEighty Motion Blur.

  To remove it, run "Uninstall OneEighty.bat".


USING IT
  Drop the effect on a 50p / 59.94p clip in a 25p sequence (clip at 100%).

  Shutter angle       Amount of blur to simulate (180 = normal film look).
  Quality             Normal (default): best balance.
                      High: cleaner edges where people pass in front of the
                      background; slower.
                      Fast: quickest (all on the GPU), but can shimmer on fine
                      repeating patterns such as checked shirts.
  Exact frame timing  On (default): even motion for 59.94p -> 25p.
  Advanced            Source shutter angle / frame rate, motion-vector view,
                      Force CPU, debug log (%TEMP%\OneEighty.log).


TIPS
  - Previewing is not real time: render the work area (Sequence > Render
    In to Out).
  - Long-GOP camera files (H.264 / HEVC, especially 10-bit 4:2:2) are slow
    for Premiere to decode. ProRes proxies make OneEighty roughly 4-5x faster
    to preview.
  - For 10-bit, log or graded footage, tick "Render at Maximum Depth" when
    exporting so the effect works in 32-bit float.


LICENCE
  MIT - see LICENSE.txt. Third-party components: see THIRD_PARTY_NOTICES.md.
