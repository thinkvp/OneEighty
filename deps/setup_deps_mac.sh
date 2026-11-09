#!/bin/sh
# macOS (Apple Silicon): fetches and builds the plugin's dependencies. Needs the
# Xcode command line tools (xcode-select --install) and CMake.
#   vcpkg/installed/arm64-osx-oneeighty  static OpenCV (core/imgproc/video)
#   ncnn-mac/      static ncnn with Vulkan, for RIFE (Quality Extra High)
#   moltenvk/      libMoltenVK.dylib, Vulkan on Metal: shipped inside the plugin
#   rife/rife-v4.6 RIFE v4.6 model (same as deps/setup_rife.bat)
# Everything is pinned and checked by SHA-256.
set -e
JOBS=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
# curl: retry transient GitHub errors (502/504)
CURL="curl -fsSL --retry 5 --retry-delay 5 --retry-all-errors"
cd "$(dirname "$0")"

# ---------------------------------------------------------------- OpenCV (vcpkg)
# Pinned so CI's cached builds stay valid; bump deliberately.
VCPKG_TAG=2026.10.16
if [ ! -d vcpkg/.git ]; then
    git clone --depth 1 --branch "$VCPKG_TAG" https://github.com/microsoft/vcpkg.git vcpkg
fi
if [ ! -x vcpkg/vcpkg ]; then
    vcpkg/bootstrap-vcpkg.sh -disableMetrics
fi
# [core] = no default features (no codecs, dnn, etc.)
vcpkg/vcpkg install "opencv4[core]:arm64-osx-oneeighty" --overlay-triplets=triplets --disable-metrics
echo "OPENCV READY"

# ---------------------------------------------------------------- ncnn
# Built from the release source rather than vcpkg: vcpkg turns off ncnn's own
# Vulkan loader (NCNN_SIMPLEVK) on macOS, and the plugin needs it to load the
# MoltenVK copy in its bundle (no Vulkan install on users' Macs). OpenMP off: a
# second OpenMP runtime in Premiere's process can abort it, and RIFE runs on
# the GPU anyway. The source folder stays: ncnn's CMake package refers to its
# simplevk.tbd (weak Vulkan symbols).
NCNN_VER=20261020
NCNN_SHA256=699d9acc67489eee0993a551ca1ebc247bbf0f4e33babe1dec810e6a9a52b988
if [ ! -f ncnn-mac/lib/libncnn.a ]; then
    rm -rf ncnn-src ncnn-mac
    $CURL -o ncnn-src.zip "https://github.com/Tencent/ncnn/releases/download/$NCNN_VER/ncnn-$NCNN_VER-full-source.zip"
    echo "$NCNN_SHA256  ncnn-src.zip" | shasum -a 256 -c -
    mkdir ncnn-src
    (cd ncnn-src && unzip -q ../ncnn-src.zip)
    rm ncnn-src.zip
    cmake -S ncnn-src -B ncnn-src/build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 \
        -DCMAKE_INSTALL_PREFIX="$PWD/ncnn-mac" \
        -DNCNN_VULKAN=ON -DNCNN_SIMPLEVK=ON -DNCNN_SYSTEM_GLSLANG=OFF -DNCNN_OPENMP=OFF \
        -DNCNN_SHARED_LIB=OFF -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
        -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF -DNCNN_PYTHON=OFF
    # bounded: a bare --parallel means make -j (no limit) and runs hundreds of
    # compilers at once, swapping an 8 GB Mac for hours
    cmake --build ncnn-src/build --parallel "$JOBS"
    cmake --install ncnn-src/build
    rm -rf ncnn-src/build
fi
echo "NCNN READY"

# ---------------------------------------------------------------- MoltenVK
MVK_VER=v1.4.2
MVK_SHA256=f95765a6229cb7b915990a2890ce12ebe36a730b021545d3d52ae69ce4c4024e
if [ ! -f moltenvk/libMoltenVK.dylib ]; then
    rm -rf moltenvk MoltenVK
    $CURL -o moltenvk.tar "https://github.com/KhronosGroup/MoltenVK/releases/download/$MVK_VER/MoltenVK-macos.tar"
    echo "$MVK_SHA256  moltenvk.tar" | shasum -a 256 -c -
    tar -xf moltenvk.tar MoltenVK/LICENSE MoltenVK/MoltenVK/dynamic/dylib/macOS/libMoltenVK.dylib
    mkdir moltenvk
    LIB=MoltenVK/MoltenVK/dynamic/dylib/macOS/libMoltenVK.dylib
    # arm64 only, like the plugin
    if lipo -archs "$LIB" | grep -q x86_64; then
        lipo "$LIB" -thin arm64 -output moltenvk/libMoltenVK.dylib
    else
        cp "$LIB" moltenvk/libMoltenVK.dylib
    fi
    cp MoltenVK/LICENSE moltenvk/LICENSE
    rm -rf MoltenVK moltenvk.tar
    codesign --force --sign - --timestamp=none moltenvk/libMoltenVK.dylib # thinned: re-sign (ad hoc)
fi
echo "MOLTENVK READY"

# ---------------------------------------------------------------- RIFE model
# github.com/nihui/rife-ncnn-vulkan, pinned commit (as deps/setup_rife.bat)
RIFE_COMMIT=5ca8c773f2e9ae78ba971bb24c35c1bad5fbaf31
RIFE_BIN_SHA256=f334ed2260149ce0188a6dcf049844e8b0cdd912e01cbcfb63553157d2508958
if [ ! -f rife/rife-v4.6/flownet.bin ]; then
    mkdir -p rife/rife-v4.6
    URL="https://raw.githubusercontent.com/nihui/rife-ncnn-vulkan/$RIFE_COMMIT/models/rife-v4.6"
    $CURL -o rife/rife-v4.6/flownet.param "$URL/flownet.param"
    $CURL -o rife/rife-v4.6/flownet.bin "$URL/flownet.bin"
    echo "$RIFE_BIN_SHA256  rife/rife-v4.6/flownet.bin" | shasum -a 256 -c -
fi
echo "RIFE MODEL READY"
