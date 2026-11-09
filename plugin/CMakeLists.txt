# macOS build (Apple Silicon): OneEighty.plugin + offline test tools.
# Windows builds with build.bat.
#
#   ../deps/setup_deps_mac.sh                         # once: vcpkg + static OpenCV
#   cmake -S . -B build-mac -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-mac                           # -> build-mac/OneEighty.plugin
#
# Needs the Xcode command line tools, CMake 3.21+, and the After Effects SDK in
# ../sdk/ae (the same SDK download as Windows; it carries the Mac headers).
#
# GPU: Metal (gpu/MetalMotionBlur.mm, the port of gpu/GpuMotionBlur.cu).
# Quality Extra High needs RIFE, not ported yet (rife/NoRife.cpp): it renders
# as High, as on a Windows PC without Vulkan.
cmake_minimum_required(VERSION 3.21)

if(NOT APPLE)
    message(FATAL_ERROR "CMakeLists.txt is the macOS build; on Windows use build.bat")
endif()

set(CMAKE_OSX_ARCHITECTURES arm64 CACHE STRING "")
set(CMAKE_OSX_DEPLOYMENT_TARGET 12.0 CACHE STRING "") # keep in sync with deps/triplets/arm64-osx-oneeighty.cmake

# Static OpenCV from vcpkg (deps/setup_deps_mac.sh), classic mode.
set(OE_VCPKG ${CMAKE_CURRENT_SOURCE_DIR}/../deps/vcpkg)
if(NOT CMAKE_TOOLCHAIN_FILE AND EXISTS ${OE_VCPKG}/scripts/buildsystems/vcpkg.cmake)
    set(CMAKE_TOOLCHAIN_FILE ${OE_VCPKG}/scripts/buildsystems/vcpkg.cmake)
endif()
set(VCPKG_TARGET_TRIPLET arm64-osx-oneeighty CACHE STRING "")
set(VCPKG_OVERLAY_TRIPLETS ${CMAKE_CURRENT_SOURCE_DIR}/../deps/triplets CACHE STRING "")
set(VCPKG_MANIFEST_MODE OFF CACHE BOOL "")

project(OneEighty LANGUAGES C CXX OBJCXX)

# Version from OneEightyParams.h (single source of truth)
file(STRINGS OneEightyParams.h _ver REGEX "#define (MAJOR|MINOR|BUG)_VERSION")
string(REGEX MATCH "MAJOR_VERSION +([0-9]+)" _m "${_ver}")
set(_major ${CMAKE_MATCH_1})
string(REGEX MATCH "MINOR_VERSION +([0-9]+)" _m "${_ver}")
set(_minor ${CMAKE_MATCH_1})
string(REGEX MATCH "BUG_VERSION +([0-9]+)" _m "${_ver}")
set(OE_VERSION ${_major}.${_minor}.${CMAKE_MATCH_1})
message(STATUS "OneEighty ${OE_VERSION}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
# Only EffectMain / PluginDataEntryFunction2 (DllExport) leave the plugin.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_OBJCXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall)

set(AE_SDK ${CMAKE_CURRENT_SOURCE_DIR}/../sdk/ae/Examples CACHE PATH "After Effects SDK Examples folder")
if(NOT EXISTS ${AE_SDK}/Headers/AE_Effect.h)
    message(FATAL_ERROR "After Effects SDK not found at ${AE_SDK} (unzip it to sdk/ae)")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc video)

# ---------------------------------------------------------------- shared code
add_library(oe_motion STATIC MotionBlur.cpp)
target_link_libraries(oe_motion PUBLIC ${OpenCV_LIBS})
target_include_directories(oe_motion PUBLIC ${OpenCV_INCLUDE_DIRS})

add_library(oe_metal STATIC gpu/MetalMotionBlur.mm rife/RifeShutter.cpp rife/NoRife.cpp)
target_compile_options(oe_metal PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_link_libraries(oe_metal PUBLIC oe_motion "-framework Metal" "-framework Foundation")

add_library(oe_platform STATIC Log.cpp UpdateCheck.cpp Platform_mac.mm)
target_compile_options(oe_platform PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_link_libraries(oe_platform PUBLIC "-framework Foundation" "-framework AppKit")

# ---------------------------------------------------------------- PiPL resource
# Rez compiles the PiPL into OneEighty.rsrc, which the host reads from
# Contents/Resources. Rez isn't a C compiler, so the AE headers' processor
# check needs __arm64__ spelled out.
set(OE_RSRC ${CMAKE_CURRENT_BINARY_DIR}/OneEighty.rsrc)
add_custom_command(
    OUTPUT ${OE_RSRC}
    COMMAND xcrun Rez -o ${OE_RSRC} -useDF -d __MACH__ -d __arm64__
            -i ${AE_SDK}/Headers -i ${AE_SDK}/Headers/SP -i ${AE_SDK}/Resources -i ${AE_SDK}/Util
            ${CMAKE_CURRENT_SOURCE_DIR}/OneEightyPiPL.r
    DEPENDS OneEightyPiPL.r
    COMMENT "Rez OneEightyPiPL.r"
    VERBATIM)
set_source_files_properties(${OE_RSRC} PROPERTIES MACOSX_PACKAGE_LOCATION Resources)

# ---------------------------------------------------------------- plugin
configure_file(mac/Info.plist.in ${CMAKE_CURRENT_BINARY_DIR}/Info.plist @ONLY)

add_library(OneEighty MODULE
    OneEighty.cpp
    ${OE_RSRC})
target_include_directories(OneEighty PRIVATE
    ${AE_SDK}/Headers ${AE_SDK}/Headers/SP ${AE_SDK}/Util ${AE_SDK}/Resources)
target_link_libraries(OneEighty PRIVATE oe_metal oe_motion oe_platform)
target_link_options(OneEighty PRIVATE -Wl,-dead_strip)
set_target_properties(OneEighty PROPERTIES
    BUNDLE TRUE
    BUNDLE_EXTENSION plugin
    MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_BINARY_DIR}/Info.plist)

# PkgInfo (type + creator, as Xcode writes for the SDK samples), then an ad-hoc
# signature over the whole bundle: Apple Silicon refuses unsigned code, and
# releases are deliberately not signed with a Developer ID.
add_custom_command(TARGET OneEighty POST_BUILD
    COMMAND /bin/sh -c "printf eFKTFXTC > \"$<TARGET_BUNDLE_CONTENT_DIR:OneEighty>/PkgInfo\""
    COMMAND codesign --force --sign - --timestamp=none $<TARGET_BUNDLE_DIR:OneEighty>
    COMMENT "PkgInfo + ad-hoc signature"
    VERBATIM)

# ---------------------------------------------------------------- offline tools
add_executable(oe_offline tools/oe_offline.cpp)
target_link_libraries(oe_offline PRIVATE oe_motion)

add_executable(oe_gpu_offline tools/oe_gpu_offline.cpp)
target_link_libraries(oe_gpu_offline PRIVATE oe_metal)

add_executable(oe_update_test tools/oe_update_test.cpp)
target_link_libraries(oe_update_test PRIVATE oe_platform)
