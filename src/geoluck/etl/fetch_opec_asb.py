from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from geoluck.config import ProjectPaths, get_paths

OPEC_ASB_PAGE_URL = "https://www.opec.org/assets/assetdb/asb-2025.pdf"
OPEC_ASB_RAW_FILENAME = "asb-2025.pdf"
OPEC_ASB_TABLE_MARKER = "By country (b/tonne)"
BARRELS_PER_CUBIC_METER_WATER = 6.289
OPEC_ASB_COUNTRY_TO_ISO3 = {
    "Algeria": "DZA",
    "Congo": "COG",
    "Equatorial Guinea": "GNQ",
    "Gabon": "GAB",
    "IR Iran": "IRN",
    "Iraq": "IRQ",
    "Kuwait": "KWT",
    "Libya": "LBY",
    "Nigeria": "NGA",
    "Saudi Arabia": "SAU",
    "United Arab Emirates": "ARE",
    "Venezuela": "VEN",
}
OPEC_ASB_INTERMEDIATE_COLUMNS = [
    "iso3",
    "country_name",
    "opec_asb_source_country_name",
    "opec_asb_barrels_per_tonne",
    "opec_asb_implied_specific_gravity",
    "opec_asb_implied_density_kg_m3",
    "opec_asb_implied_api_gravity",
    "opec_asb_page_number",
]


@dataclass(frozen=True)
class OpecAsbFetchResult:
    raw_path: Path
    tidy_path: Path
    provenance_path: Path
    row_count: int
    country_count: int
    page_number: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def barrels_per_tonne_to_specific_gravity(barrels_per_tonne: float) -> float:
    if barrels_per_tonne <= 0:
        raise ValueError("Barrels-per-tonne conversion factor must be positive.")
    return BARRELS_PER_CUBIC_METER_WATER / barrels_per_tonne


def specific_gravity_to_api_gravity(specific_gravity: float) -> float:
    if specific_gravity <= 0:
        raise ValueError("Specific gravity must be positive.")
    return 141.5 / specific_gravity - 131.5


def parse_country_conversion_table(text: str) -> pd.DataFrame:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    try:
        start = lines.index(OPEC_ASB_TABLE_MARKER) + 1
    except ValueError as exc:
        raise ValueError("Could not find the OPEC ASB country conversion-factor marker.") from exc

    records: list[dict[str, object]] = []
    for line in lines[start:]:
        if line == "OPEC":
            break
        for source_country_name, iso3 in OPEC_ASB_COUNTRY_TO_ISO3.items():
            prefix = f"{source_country_name} "
            if not line.startswith(prefix):
                continue
            barrels_per_tonne = float(pd.to_numeric(line.removeprefix(prefix), errors="raise"))
            specific_gravity = barrels_per_tonne_to_specific_gravity(barrels_per_tonne)
            records.append(
                {
                    "iso3": iso3,
                    "country_name": source_country_name,
                    "opec_asb_source_country_name": source_country_name,
                    "opec_asb_barrels_per_tonne": barrels_per_tonne,
                    "opec_asb_implied_specific_gravity": specific_gravity,
                    "opec_asb_implied_density_kg_m3": specific_gravity * 1000.0,
                    "opec_asb_implied_api_gravity": specific_gravity_to_api_gravity(
                        specific_gravity
                    ),
                }
            )
            break

    frame = pd.DataFrame.from_records(records)
    expected = set(OPEC_ASB_COUNTRY_TO_ISO3.values())
    actual = set(frame["iso3"].tolist())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "Parsed OPEC ASB conversion factors did not match the expected members. "
            f"Missing={missing}; extra={extra}"
        )
    duplicates = frame.duplicated(subset=["iso3"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate iso3 rows found in parsed OPEC ASB conversion factors.")
    return frame.sort_values("iso3", kind="stable").reset_index(drop=True)


def extract_country_conversion_table(pdf_path: Path) -> tuple[pd.DataFrame, int]:
    reader = PdfReader(str(pdf_path))
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if OPEC_ASB_TABLE_MARKER not in text:
            continue
        return parse_country_conversion_table(text), page_number
    raise ValueError(
        "Could not find the OPEC ASB crude-oil conversion-factor table in the PDF text."
    )


def write_provenance(
    *,
    paths: ProjectPaths,
    raw_path: Path,
    tidy_path: Path,
    page_number: int,
) -> Path:
    provenance_path = paths.data_intermediate / "opec_asb" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(tidy_path)
    payload = {
        "source_page_url": OPEC_ASB_PAGE_URL,
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_path": str(raw_path),
        "raw_sha256": file_sha256(raw_path),
        "tidy_path": str(tidy_path),
        "tidy_rows": int(len(frame)),
        "country_count": int(frame["iso3"].nunique()),
        "page_number": int(page_number),
        "table_marker": OPEC_ASB_TABLE_MARKER,
        "note": (
            "This adapter extracts OPEC crude barrels-per-tonne conversion factors from the "
            "ASB PDF and derives implied density and API gravity. The PDF did not expose a "
            "machine-readable sulfur table in this implementation."
        ),
    }
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return provenance_path


def run_fetch(
    paths: ProjectPaths | None = None,
    *,
    force: bool = False,
) -> OpecAsbFetchResult:
    resolved_paths = paths or get_paths()
    raw_dir = resolved_paths.data_raw / "opec_asb"
    tidy_dir = resolved_paths.data_intermediate / "opec_asb"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tidy_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / OPEC_ASB_RAW_FILENAME
    tidy_path = tidy_dir / "country_crude_conversion_factors.parquet"
    provenance_path = tidy_dir / "provenance.json"

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Expected OPEC ASB PDF not found: {raw_path}. Download {OPEC_ASB_PAGE_URL} first."
        )

    if tidy_path.exists() and provenance_path.exists() and not force:
        frame = pd.read_parquet(tidy_path)
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        return OpecAsbFetchResult(
            raw_path=raw_path,
            tidy_path=tidy_path,
            provenance_path=provenance_path,
            row_count=len(frame),
            country_count=int(frame["iso3"].nunique()),
            page_number=int(payload.get("page_number", -1)),
        )

    frame, page_number = extract_country_conversion_table(raw_path)
    frame["opec_asb_page_number"] = int(page_number)
    frame = frame.loc[:, OPEC_ASB_INTERMEDIATE_COLUMNS].copy()
    frame.to_parquet(tidy_path, index=False)
    provenance_path = write_provenance(
        paths=resolved_paths,
        raw_path=raw_path,
        tidy_path=tidy_path,
        page_number=page_number,
    )
    return OpecAsbFetchResult(
        raw_path=raw_path,
        tidy_path=tidy_path,
        provenance_path=provenance_path,
        row_count=len(frame),
        country_count=int(frame["iso3"].nunique()),
        page_number=page_number,
    )
