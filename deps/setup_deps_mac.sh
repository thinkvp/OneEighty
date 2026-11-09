#!/bin/sh
# macOS: fetches vcpkg and builds the plugin's static libraries (a minimal
# OpenCV: core/imgproc/video) for Apple Silicon. Needs the Xcode command line
# tools (xcode-select --install) and CMake.
# ncnn (Quality "Extra High") comes later, with the Metal GPU engine.
set -e
cd "$(dirname "$0")"
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
