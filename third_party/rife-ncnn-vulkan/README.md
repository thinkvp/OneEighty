# rife-ncnn-vulkan (MIT, Copyright (c) 2020 nihui)

Source files from https://github.com/nihui/rife-ncnn-vulkan at commit
`a7532fc3f9f8f008cd6eecd6f2ffe2a9698e0cf7`, kept for reference and licensing.

- `warp.cpp`, `warp.comp`, `warp_pack4.comp`: RIFE's custom Warp layer. OneEighty's
  copy is `plugin/rife/RifeFlow.cpp`, with the shaders embedded in
  `plugin/rife/RifeWarpShaders.inl` (current ncnn no longer uses the pack8 variant).
- The ncnn model files (`rife-v4.6/flownet.param`, `flownet.bin`) come from the same
  repository: `deps/setup_rife.bat` downloads them at a pinned commit and checks the
  SHA-256.
