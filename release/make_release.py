"""Package a release zip from the current build:

    plugin\\build.bat
    python release/make_release.py

-> dist/OneEighty-v<version>-win64.zip containing OneEighty.aex, the install /
uninstall scripts, README.txt, LICENSE.txt and THIRD_PARTY_NOTICES.md.
The version is read from plugin/OneEightyParams.h.
"""
import re
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
out = dist / f"OneEighty-v{ver}-win64.zip"
folder = f"OneEighty-v{ver}"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(AEX, f"{folder}/OneEighty.aex")
    for name in ("Install OneEighty.bat", "Uninstall OneEighty.bat"):
        z.write(REL / name, f"{folder}/{name}")
    z.writestr(f"{folder}/README.txt", (REL / "README.txt").read_text(encoding="utf-8").replace("{VERSION}", f"v{ver}"))
    z.write(ROOT / "LICENSE", f"{folder}/LICENSE.txt")
    z.write(ROOT / "THIRD_PARTY_NOTICES.md", f"{folder}/THIRD_PARTY_NOTICES.md")
print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), version {ver}")
