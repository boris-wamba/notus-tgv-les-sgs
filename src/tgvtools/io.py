"""Readers and small data-cleaning utilities (replaces sorted_data.py and convert_comma_to_point.py)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

# Column layouts of the Notus post-processing files (index of each quantity).
# They differ between file types: check them against the `post_processing` block of your .nts file.
LAYOUT_LES = {"time": 0, "mean_kedr": 2, "nu_sgs": 4, "production": 5, "s": 6}
LAYOUT_DNS_LOG = {"time": 0, "mean_kedr": 2, "mean_kedr_filtered": 4}
LAYOUT_DNS_FILTERED = {"time": 0, "mean_kedr": 2, "mean_kedr_filtered": 6}


def load_columns(path, columns: Sequence[int] = (0, 2)) -> Tuple[np.ndarray, ...]:
    """Load selected columns of a whitespace-separated .dat file ('#' lines ignored), sorted by time."""
    data = np.loadtxt(path, comments="#", usecols=tuple(columns), ndmin=2)
    data = data[np.argsort(data[:, 0], kind="stable")]
    return tuple(data[:, i] for i in range(data.shape[1]))


def load_layout(path, layout: dict, *names: str) -> Tuple[np.ndarray, ...]:
    """Load named columns using one of the LAYOUT_* dictionaries (time is always returned first)."""
    cols = [layout["time"]] + [layout[n] for n in names]
    return load_columns(path, cols)


def sort_time_series(in_file, out_file=None, header: str = "# time   value") -> Path:
    """Sort a 'time value' file by increasing time (restarted runs append out of order)."""
    rows = []
    for line in Path(in_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            parts = line.split()
            rows.append((float(parts[0]), float(parts[1])))
    rows.sort(key=lambda r: r[0])
    out = Path(out_file) if out_file else Path(in_file).with_suffix(".dat")
    out.write_text(header + "\n" + "\n".join(f"{t:.4f}    {v:.7f}" for t, v in rows) + "\n", encoding="utf-8")
    return out


def convert_decimal_comma(in_file, out_file=None) -> Path:
    """Replace decimal commas by points (data digitised from papers, e.g. the Van Rees reference)."""
    out = Path(out_file) if out_file else Path(str(in_file) + ".dat")
    out.write_text(Path(in_file).read_text(encoding="utf-8").replace(",", "."), encoding="utf-8")
    return out


def read_adios_velocity(path):
    """Read velocity_u/v/w from a Notus ADIOS .bp file (requires the optional `adios` package)."""
    try:
        import adios  # noqa: WPS433 (optional dependency)
    except ImportError as exc:
        raise ImportError("reading .bp files needs the 'adios' Python package") from exc
    fp = adios.File(str(path))
    try:
        return tuple(fp[f"velocity_{c}"][:] for c in "uvw")
    finally:
        fp.close()
