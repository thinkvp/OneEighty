"""Package a release from the current build:

    plugin\\build.bat
    python release/make_release.py

-> dist/OneEighty-v<version>-setup.exe   Inno Setup installer (needs Inno Setup 6)
-> dist/OneEighty-v<version>-win64.zip   manual install: .aex + install/uninstall
                                         scripts, README.txt, LICENSE, notices
The version is read from plugin/OneEightyParams.h.
"""
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REL = ROOT / "release"
AEX = ROOT / "plugin" / "build" / "OneEighty.aex"

params = (ROOT / "plugin" / "OneEightyParams.h").read_text(encoding="utf-8")
ver = ".".join(re.search(rf"#define {k}\s+(\d+)", params).group(1) for k in ("MAJOR_VERSION", "MINOR_VERSION", "BUG_VERSION"))
if not AEX.exists():
    raise SystemExit(f"{AEX} not found - run plugin\\build.bat first")

dist = ROOT / "dist"
dist.mkdir(exist_ok=True)
readme = (REL / "README.txt").read_text(encoding="utf-8").replace("{VERSION}", f"v{ver}")
(dist / "README.txt").write_text(readme.replace("\n", "\r\n"), encoding="utf-8", newline="")

# ---- zip (manual install)
out = dist / f"OneEighty-v{ver}-win64.zip"
folder = f"OneEighty-v{ver}"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(AEX, f"{folder}/OneEighty.aex")
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
print(f"version {ver}")
