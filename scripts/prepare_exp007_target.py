"""Export the native CvNA2 plot bitmap; no raster resampling or tracing.

Requires optional pypdf==6.10.0 (not needed for analysis or core tests).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/exp006_motor/gorko2024_supplement.pdf"
SOURCE_SHA = "0d79fcef0205b2eb27cd0f1e205862ecebea7610fb5bbb43fa6421c370bdc7d2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/exp007_motor_boundary")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / "data"):
        raise ValueError("Native source export must remain under ignored project data/")
    from pypdf import PdfReader
    import pypdf

    if pypdf.__version__ != "6.10.0":
        raise ValueError("Use pypdf==6.10.0 for the frozen native bitmap export")
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("Supplement PDF differs from frozen EXP-006 source")
    page = PdfReader(SOURCE).pages[14]
    native = next(image for image in page.images if image.name == "Im6.png")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "CvNA2_velocity_native.png"
    if path.exists() and path.read_bytes() != native.data:
        raise ValueError("Refusing to replace a different existing source bitmap")
    path.write_bytes(native.data)
    print(json.dumps({"path": path.relative_to(ROOT).as_posix(),
                      "bytes": len(native.data),
                      "sha256": hashlib.sha256(native.data).hexdigest(),
                      "source_page_one_based": 15,
                      "figure": "Supplementary Figure 10g-i", "object": "/Im6",
                      "width": native.image.width, "height": native.image.height},
                     indent=2))


if __name__ == "__main__":
    main()
