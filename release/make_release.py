"""Package a release from the current build:

    plugin\\build.bat
    python release/make_release.py

-> dist/OneEighty-v<version>-setup.exe   Inno Setup installer (needs Inno Setup 6)
-> dist/OneEighty-v<version>-win64.zip   manual install: .aex + RIFE model + install/
                                         uninstall scripts, README.txt, LICENSE, notices
--mac: also -> dist/OneEighty-v<version>-mac-arm64.zip, from the zip the macOS
CI job uploads (OneEighty.plugin), with README.txt, LICENSE and notices.
The version is read from plugin/OneEightyParams.h.
"""
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REL = ROOT / "release"
AEX = ROOT / "plugin" / "build" / "OneEighty.aex"
MODEL = ROOT / "plugin" / "build" / "rife-v4.6"  # Quality "Extra High" (deps/setup_rife.bat, copied by build.bat)

params = (ROOT / "plugin" / "OneEightyParams.h").read_text(encoding="utf-8")
ver = ".".join(re.search(rf"#define {k}\s+(\d+)", params).group(1) for k in ("MAJOR_VERSION", "MINOR_VERSION", "BUG_VERSION"))
if not AEX.exists():
    raise SystemExit(f"{AEX} not found - run plugin\\build.bat first")
if not (MODEL / "flownet.bin").exists() or not (MODEL / "flownet.param").exists():
    raise SystemExit(f"{MODEL} incomplete - run deps\\setup_rife.bat, then plugin\\build.bat")

dist = ROOT / "dist"
dist.mkdir(exist_ok=True)
readme = (REL / "README.txt").read_text(encoding="utf-8").replace("{VERSION}", f"v{ver}")
(dist / "README.txt").write_text(readme.replace("\n", "\r\n"), encoding="utf-8", newline="")

# ---- zip (manual install)
out = dist / f"OneEighty-v{ver}-win64.zip"
folder = f"OneEighty-v{ver}"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(AEX, f"{folder}/OneEighty.aex")
    for name in ("flownet.param", "flownet.bin"):
        z.write(MODEL / name, f"{folder}/rife-v4.6/{name}")
    for name in ("Install OneEighty.bat", "Uninstall OneEighty.bat"):
        z.write(REL / name, f"{folder}/{name}")
    z.write(dist / "README.txt", f"{folder}/README.txt")
    z.write(ROOT / "LICENSE", f"{folder}/LICENSE.txt")
    z.write(ROOT / "THIRD_PARTY_NOTICES.md", f"{folder}/THIRD_PARTY_NOTICES.md")
print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

# ---- installer (Inno Setup)
candidates = [shutil.which("ISCC"),
              Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",
              Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
              Path(r"C:\Program Files\Inno Setup 6\ISCC.exe")]
iscc = next((Path(c) for c in candidates if c and Path(c).exists()), None)
if not iscc:
    print("Inno Setup 6 not found: skipped the installer")
else:
    r = subprocess.run([str(iscc), "/Q", f"/DAppVersion={ver}", f"/DRootDir={ROOT}", str(REL / "OneEighty.iss")],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stdout + r.stderr)
    exe = dist / f"OneEighty-v{ver}-setup.exe"
    print(f"wrote {exe} ({exe.stat().st_size / 1e6:.1f} MB)")

# ---- macOS (repackage the CI build, keeping its Unix permissions)
if "--mac" in sys.argv:
    src = Path(sys.argv[sys.argv.index("--mac") + 1])
    out = dist / f"OneEighty-v{ver}-mac-arm64.zip"
    folder = f"OneEighty-v{ver}-mac"
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        names = zin.namelist()
        if "OneEighty.plugin/Contents/Info.plist" not in names:
            raise SystemExit(f"{src}: no OneEighty.plugin/ inside")
        plist = zin.read("OneEighty.plugin/Contents/Info.plist").decode()
        if f"<string>{ver}</string>" not in plist:
            raise SystemExit(f"{src}: Info.plist isn't version {ver}")
        for info in zin.infolist():
            if info.filename.startswith("__MACOSX/"):
                continue
            new = zipfile.ZipInfo(f"{folder}/{info.filename}", info.date_time)
            new.external_attr, new.create_system = info.external_attr, 3  # Unix mode bits
            new.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(new, zin.read(info.filename))
        def add(name, data, mode=0o644):
            i = zipfile.ZipInfo(f"{folder}/{name}", (2026, 1, 1, 0, 0, 0))
            i.external_attr, i.create_system, i.compress_type = (0o100000 | mode) << 16, 3, zipfile.ZIP_DEFLATED
            z.writestr(i, data)
        add("README.txt", readme)
        add("LICENSE.txt", (ROOT / "LICENSE").read_bytes())
        add("THIRD_PARTY_NOTICES.md", (ROOT / "THIRD_PARTY_NOTICES.md").read_bytes())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
print(f"version {ver}")
