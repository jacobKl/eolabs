from __future__ import annotations

from pathlib import Path

import numpy as np
import spectral.io.envi as envi

FALLBACK_RGB = (30, 20, 10)


def find_hdr_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.hdr"))


def parse_wavelengths(metadata: dict) -> np.ndarray | None:
    raw_wavelengths = metadata.get("wavelength")
    if not raw_wavelengths:
        return None
    try:
        return np.asarray([float(w) for w in raw_wavelengths], dtype=np.float64)
    except (TypeError, ValueError):
        return None


def parse_ignore_value(metadata: dict) -> float | None:
    raw_ignore = metadata.get("data ignore value")
    if raw_ignore is None:
        return None
    try:
        return float(str(raw_ignore).strip())
    except ValueError:
        return None


def parse_rgb_bands(metadata: dict, band_count: int) -> tuple[int, int, int]:
    raw_default_bands = metadata.get("default bands")
    if raw_default_bands and len(raw_default_bands) >= 3:
        try:
            parsed = tuple(int(float(v)) - 1 for v in raw_default_bands[:3])
            if all(0 <= band < band_count for band in parsed):
                return parsed
        except (TypeError, ValueError):
            pass
    return tuple(np.clip(np.asarray(FALLBACK_RGB), 0, max(0, band_count - 1)))  # type: ignore[return-value]


def load_dataset(hdr_path: Path):
    return envi.open(str(hdr_path))


def normalize_rgb_channels(rgb: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    for channel_idx in range(3):
        channel = rgb[:, :, channel_idx]
        finite = np.isfinite(channel)
        if not np.any(finite):
            rgb[:, :, channel_idx] = 0.0
            continue

        p2, p98 = np.nanpercentile(channel, [low_pct, high_pct])
        denom = max(p98 - p2, 1e-6)
        rgb[:, :, channel_idx] = np.clip((channel - p2) / denom, 0.0, 1.0)
    return rgb


def read_rgb_image(
    img,
    rgb_bands: tuple[int, int, int],
    ignore_value: float | None,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> np.ndarray:
    rgb = img.read_bands(list(rgb_bands)).astype(np.float32)
    if ignore_value is not None:
        rgb[rgb >= ignore_value] = np.nan
    rgb[rgb < 0] = np.nan
    return np.nan_to_num(normalize_rgb_channels(rgb, low_pct=low_pct, high_pct=high_pct), nan=0.0)


def read_pixel_spectrum(img, row: int, col: int, ignore_value: float | None) -> np.ndarray:
    spectrum = img.read_pixel(row, col).astype(np.float64)
    if ignore_value is not None:
        spectrum[spectrum >= ignore_value] = np.nan
    spectrum[spectrum < 0] = np.nan
    return spectrum


def _map_info_tokens(metadata: dict) -> list[str]:
    raw_map_info = metadata.get("map info")
    if raw_map_info is None:
        return []
    if isinstance(raw_map_info, (list, tuple)):
        return [str(token).strip() for token in raw_map_info]
    text = str(raw_map_info).strip().strip("{}")
    return [part.strip() for part in text.split(",") if part.strip()]


def _pixel_to_map_coordinates(metadata: dict, row: int, col: int) -> tuple[float, float] | None:
    tokens = _map_info_tokens(metadata)
    if len(tokens) < 7:
        return None
    try:
        ref_col_1based = float(tokens[1])
        ref_row_1based = float(tokens[2])
        ref_x = float(tokens[3])
        ref_y = float(tokens[4])
        px_x = float(tokens[5])
        px_y = float(tokens[6])
    except ValueError:
        return None

    col_1based = float(col + 1)
    row_1based = float(row + 1)
    row_step = -abs(px_y) if px_y > 0 else px_y

    x = ref_x + (col_1based - ref_col_1based) * px_x
    y = ref_y + (row_1based - ref_row_1based) * row_step
    return x, y


def format_map_coordinates(metadata: dict, row: int, col: int) -> str:
    coords = _pixel_to_map_coordinates(metadata, row, col)
    if coords is None:
        return "-"
    x, y = coords
    lower = " ".join(token.lower() for token in _map_info_tokens(metadata))
    is_geographic = "geographic" in lower or "degree" in lower
    if is_geographic:
        return f"lon={x:.6f}, lat={y:.6f}"
    return f"x={x:.3f}, y={y:.3f}"
