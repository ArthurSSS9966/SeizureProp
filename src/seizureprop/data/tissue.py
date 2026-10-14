"""Conservative contact tissue mapping, with source provenance and QC flags.

Explicit tissue annotations take precedence over atlas regions. Discordance is
reported, never resolved using seizure labels. The ROI fallback recognizes the
FreeSurfer Desikan-Killiany cortex and selected aseg tissue structures only;
unrecognized regions are unknown, not presumed grey matter.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping

POLICY_VERSION = "contact-tissue-v2"
_DK = dict(enumerate([
    "unknown", "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal",
    "corpuscallosum", "cuneus", "entorhinal", "fusiform", "inferiorparietal",
    "inferiortemporal", "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal",
    "lingual", "medialorbitofrontal", "middletemporal", "parahippocampal",
    "paracentral", "parsopercularis", "parsorbitalis", "parstriangularis",
    "pericalcarine", "postcentral", "posteriorcingulate", "precentral",
    "precuneus", "rostralanteriorcingulate", "rostralmiddlefrontal",
    "superiorfrontal", "superiorparietal", "superiortemporal", "supramarginal",
    "frontalpole", "temporalpole", "transversetemporal", "insula",
]))
_ROI: dict[str, tuple[str, int | None]] = {
    f"ctx-{side}-{name}": ("GM", base + index)
    for side, base in [("lh", 1000), ("rh", 2000)]
    for index, name in _DK.items() if index not in (0, 4)
}
for _name, _left, _right in [("thalamus-proper", 10, 49), ("caudate", 11, 50),
                             ("putamen", 12, 51), ("pallidum", 13, 52),
                             ("hippocampus", 17, 53), ("amygdala", 18, 54),
                             ("accumbens-area", 26, 58)]:
    _ROI[f"left-{_name}"] = ("GM", _left)
    _ROI[f"right-{_name}"] = ("GM", _right)
_ROI.update({"left-cerebellum-cortex": ("GM", 8),
             "right-cerebellum-cortex": ("GM", 47),
             "left-cerebellum-white-matter": ("WM", 7),
             "right-cerebellum-white-matter": ("WM", 46)})
for _name, _code in [("left-cerebral-white-matter", 2),
                     ("right-cerebral-white-matter", 41),
                     ("wm-hypointensities", 77),
                     ("cc_posterior", 251), ("cc_mid_posterior", 252),
                     ("cc_central", 253), ("cc_mid_anterior", 254),
                     ("cc_anterior", 255)]:
    _ROI[_name] = ("WM", _code)
_ROI["white-matter"] = ("WM", None)  # Dataset's explicit HUP sentinel.
for _name, _code in [("left-lateral-ventricle", 4), ("left-inf-lat-vent", 5),
                     ("right-lateral-ventricle", 43), ("right-inf-lat-vent", 44),
                     ("3rd-ventricle", 14), ("4th-ventricle", 15), ("csf", 24)]:
    _ROI[_name] = ("CSF", _code)
_UNKNOWN = {"", "nan", "none", "unknown", "outside-brain", "emptyroom"}


def _text(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def contact_key(name: str) -> tuple[str, int] | None:
    """Canonicalize case and zero padding without stripping arbitrary prefixes."""
    match = re.fullmatch(r"([A-Za-z](?:[A-Za-z0-9]*[A-Za-z])?)0*(\d+)", str(name).strip())
    return (match[1].upper(), int(match[2])) if match else None


def atlas_tissue(region: Any, number: Any = None) -> tuple[str, tuple[str, ...]]:
    """Return atlas tissue and flags, rejecting known name/number mismatches."""
    name = _text(region)
    if name in _UNKNOWN:
        return "UNKNOWN", ()
    if name not in _ROI:
        return "UNKNOWN", ("unrecognized_roi",)
    tissue, expected = _ROI[name]
    if _text(number) not in _UNKNOWN and expected is not None:
        try:
            numeric = float(number)
        except (TypeError, ValueError):
            return "UNKNOWN", ("invalid_roi_number",)
        if not math.isfinite(numeric) or numeric != expected:
            return "UNKNOWN", ("roi_name_number_mismatch",)
    return tissue, ()


@dataclass(frozen=True)
class ContactTissue:
    label: str
    tissue: str
    source: str
    atlas_tissue: str
    flags: tuple[str, ...]


def map_contacts(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int], ContactTissue]:
    """Map electrode rows, rejecting unparsable or colliding contact identities.

    A present ``matter`` column is authoritative, including unknown/CSF values.
    G and A are grey aliases by the project annotation policy. Atlas disagreement
    remains a QC flag and cannot silently override explicit tissue annotations.
    """
    output = {}
    for row in rows:
        label = str(row["label"])
        key = contact_key(label)
        if key is None or key in output:
            raise ValueError(f"Invalid or duplicate canonical contact identity: {label!r}")
        explicit = "matter" in row
        atlas, flags = atlas_tissue(row.get("brain_area") if explicit else row.get("roi"),
                                    None if explicit else row.get("roiNum"))
        if explicit:
            matter = _text(row["matter"])
            tissue = {"grey": "GM", "gray": "GM", "g": "GM", "a": "GM",
                      "white": "WM", "w": "WM", "csf": "CSF"}.get(matter, "UNKNOWN")
            if tissue == "UNKNOWN" and matter not in _UNKNOWN:
                flags += ("unrecognized_matter",)
            if tissue != "UNKNOWN" and atlas != "UNKNOWN" and tissue != atlas:
                flags += ("matter_atlas_conflict",)
        else:
            tissue = atlas
        output[key] = ContactTissue(label, tissue if tissue in ("GM", "WM") else "UNKNOWN",
                                   "matter" if explicit else "roi", atlas, flags)
    return output


def pair_tissue(left: ContactTissue | None, right: ContactTissue | None) -> int:
    """Code pure GM=1, pure WM=0, and mixed/missing/non-tissue=-1."""
    if left is None or right is None or left.tissue != right.tissue:
        return -1
    return {"GM": 1, "WM": 0}.get(left.tissue, -1)


def pair_composition(left: str, right: str, conflicted: bool = False) -> int:
    """Return pure WM=0, pure GM=1, known GM/WM mixed=2, or unknown=-1.

    Unknown, missing, non-tissue and quarantined conflicting annotations must
    never become the mixed training class. Inputs are canonical contact tissues.
    """
    if conflicted or left not in {"GM", "WM"} or right not in {"GM", "WM"}:
        return -1
    if left != right:
        return 2
    return 1 if left == "GM" else 0
