#!/usr/bin/env python3
"""
Hyperspectral BSQ viewer.

Features:
- Open ENVI `.hdr` files (with companion BSQ/BIL/BIP binary file)
- Render an RGB preview from default/fallback bands
- Click any pixel to inspect and plot its spectral signature
- Export the selected spectrum to CSV

Usage:
    python viewer.py [path/to/file.hdr]
"""

from __future__ import annotations

import csv
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

matplotlib.use("TkAgg")

try:
    import spectral.io.envi as envi  # kept for dependency check only
except ImportError:
    sys.exit("The 'spectral' library is missing.\nInstall it with: pip install spectral")

from viewer_logic import (
    format_map_coordinates,
    find_hdr_files,
    load_dataset,
    parse_ignore_value,
    parse_rgb_bands,
    parse_wavelengths,
    read_pixel_spectrum,
    read_rgb_image,
)


# Default search directory (relative to this script).
DATA_DIR = Path(__file__).parent / "data"
DARK_BG = "#121417"
PANEL_BG = "#1b1f24"
TEXT_FG = "#e6e6e6"
MUTED_FG = "#a6b0ba"
ACCENT = "#5da9ff"


class HyperspectralViewer:
    """Presentation/UI layer for ENVI hyperspectral data viewer."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hyperspectral Cube Viewer")
        self.root.geometry("1360x780")
        self.root.minsize(1100, 640)

        self.img = None
        self.current_path: Path | None = None
        self.wavelengths: np.ndarray | None = None
        self.ignore_value: float | None = None
        self.rgb_display: np.ndarray | None = None
        self.spectrum: np.ndarray | None = None
        self.pixel_pos: tuple[int, int] | None = None
        self.rgb_bands: tuple[int, int, int] | None = None
        self.zoom_factor: float = 1.0
        self.zoom_center: tuple[float, float] | None = None

        self.status_var = tk.StringVar(value="No file loaded.")
        self.file_var = tk.StringVar(value="File: -")
        self.shape_var = tk.StringVar(value="Shape: -")
        self.pixel_var = tk.StringVar(value="Pixel: -")
        self.map_coords_var = tk.StringVar(value="Map coords: -")
        self.range_var = tk.StringVar(value="Range: -")
        self.available_hdr_files: list[Path] = []
        self.file_listbox: tk.Listbox | None = None
        self._build_ui()
        self._auto_load_initial_file()

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        self.root.configure(bg=DARK_BG)
        style.configure(".", background=DARK_BG, foreground=TEXT_FG)
        style.configure("TFrame", background=DARK_BG)
        style.configure("TLabelframe", background=PANEL_BG, foreground=TEXT_FG, borderwidth=1)
        style.configure("TLabelframe.Label", background=PANEL_BG, foreground=TEXT_FG)
        style.configure("TLabel", background=DARK_BG, foreground=TEXT_FG)
        style.configure("Status.TLabel", background=DARK_BG, foreground=MUTED_FG)
        style.configure("TButton", padding=(10, 6), background="#2a3038", foreground=TEXT_FG)
        style.map("TButton", background=[("active", "#343b44")])

        center = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        center.pack(fill=tk.BOTH, expand=True)

        split = ttk.Panedwindow(center, orient=tk.HORIZONTAL)
        split.pack(fill=tk.BOTH, expand=True)

        plot_frame = ttk.Frame(split)
        side_panel = ttk.Frame(split, padding=(12, 10))
        split.add(plot_frame, weight=5)
        split.add(side_panel, weight=2)

        self.fig = Figure(figsize=(13.5, 7), dpi=100, facecolor=DARK_BG)
        self.ax_rgb = self.fig.add_subplot(1, 2, 1)
        self.ax_spec = self.fig.add_subplot(1, 2, 2)
        self.fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08, wspace=0.20)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)
        self._build_side_panel(side_panel)

    def _build_side_panel(self, panel: ttk.Frame) -> None:
        info = ttk.LabelFrame(panel, text="Dataset Info", padding=10)
        info.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(info, textvariable=self.file_var).pack(anchor=tk.W)
        ttk.Label(info, textvariable=self.shape_var).pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(info, textvariable=self.range_var).pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(info, textvariable=self.pixel_var).pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(info, textvariable=self.map_coords_var).pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(info, textvariable=self.status_var, style="Status.TLabel", wraplength=280).pack(
            anchor=tk.W, pady=(6, 0)
        )

        files = ttk.LabelFrame(panel, text="Files", padding=10)
        files.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.file_listbox = tk.Listbox(
            files,
            height=12,
            bg="#20252c",
            fg=TEXT_FG,
            selectbackground=ACCENT,
            selectforeground="#111",
            highlightthickness=0,
            activestyle="none",
        )
        self.file_listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.file_listbox.bind("<Double-Button-1>", lambda _: self._load_selected_sidebar_file())
        ttk.Button(files, text="Open Selected", command=self._load_selected_sidebar_file).pack(fill=tk.X)
        ttk.Button(files, text="Refresh List", command=self._refresh_sidebar_file_list).pack(
            fill=tk.X, pady=(6, 0)
        )

        controls = ttk.LabelFrame(panel, text="Controls", padding=10)
        controls.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(controls, text="Open Any .hdr", command=self._open_file_dialog).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(controls, text="Export CSV", command=self._export_csv).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(controls, text="Zoom In", command=self._zoom_in).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(controls, text="Zoom Out", command=self._zoom_out).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(controls, text="Reset Zoom", command=self._reset_zoom).pack(fill=tk.X, pady=(6, 0))

        hints = ttk.LabelFrame(panel, text="Tips", padding=10)
        hints.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            hints,
            text=(
                "- Click image to inspect full spectrum\n"
                "- Double-click a file to load it\n"
                "- Export selected spectrum to CSV"
            ),
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

    def _refresh_sidebar_file_list(self) -> None:
        if self.file_listbox is None:
            return
        search_dir = self.current_path.parent if self.current_path is not None else DATA_DIR
        if not search_dir.exists():
            self.available_hdr_files = []
        else:
            self.available_hdr_files = find_hdr_files(search_dir)
        self.file_listbox.delete(0, tk.END)
        for path in self.available_hdr_files:
            self.file_listbox.insert(tk.END, path.name)
        if self.current_path is not None:
            for idx, path in enumerate(self.available_hdr_files):
                if path.name == self.current_path.name:
                    self.file_listbox.selection_set(idx)
                    self.file_listbox.see(idx)
                    break

    def _load_selected_sidebar_file(self) -> None:
        if self.file_listbox is None:
            return
        selected = self.file_listbox.curselection()
        if not selected:
            return
        path = self.available_hdr_files[selected[0]]
        self._load_dataset(path)

    def _auto_load_initial_file(self) -> None:
        """Load file from argv or auto-discover in default directory."""
        if len(sys.argv) > 1:
            self._load_dataset(Path(sys.argv[1]))
            self._refresh_sidebar_file_list()
            return

        if not DATA_DIR.exists():
            self.status_var.set(f"Default data folder not found: {DATA_DIR}")
            self._refresh_plots()
            self._refresh_sidebar_file_list()
            return

        hdr_files = find_hdr_files(DATA_DIR)
        if not hdr_files:
            self.status_var.set(f"No .hdr files found in {DATA_DIR}")
            self._refresh_plots()
            self._refresh_sidebar_file_list()
            return
        if len(hdr_files) == 1:
            self._load_dataset(hdr_files[0])
            self._refresh_sidebar_file_list()
            return
        self._load_dataset(hdr_files[0])
        self._refresh_sidebar_file_list()

    def _open_file_dialog(self) -> None:
        selected = filedialog.askopenfilename(
            title="Open ENVI header (.hdr)",
            initialdir=DATA_DIR if DATA_DIR.exists() else Path.home(),
            filetypes=[("ENVI Header", "*.hdr"), ("All files", "*.*")],
        )
        if selected:
            self._load_dataset(Path(selected))

    def _load_dataset(self, hdr_path: Path) -> None:
        self.status_var.set(f"Loading: {hdr_path.name} ...")
        self.root.update_idletasks()

        try:
            self.img = load_dataset(hdr_path)
            metadata = self.img.metadata

            self.current_path = hdr_path
            self.wavelengths = parse_wavelengths(metadata)
            self.ignore_value = parse_ignore_value(metadata)
            rgb_bands = parse_rgb_bands(metadata, self.img.nbands)
            self.rgb_bands = rgb_bands
            self.rgb_display = read_rgb_image(
                self.img,
                rgb_bands,
                self.ignore_value,
            )

            self.spectrum = None
            self.pixel_pos = None
            self._reset_zoom()
            self._refresh_plots()
            self._update_info_panel()
            self._refresh_sidebar_file_list()

            self.status_var.set(
                f"{hdr_path.name} | {self.img.nrows}x{self.img.ncols} | "
                f"{self.img.nbands} bands | RGB bands {rgb_bands} | Click image to inspect."
            )
        except Exception as exc:
            self.img = None
            self.rgb_display = None
            self.spectrum = None
            self.pixel_pos = None
            self.rgb_bands = None
            self._reset_zoom()
            self._refresh_plots()
            self._update_info_panel()
            self._refresh_sidebar_file_list()
            messagebox.showerror("Failed to load dataset", str(exc))
            self.status_var.set("Load failed.")

    def _update_info_panel(self) -> None:
        if self.current_path is None or self.img is None:
            self.file_var.set("File: -")
            self.shape_var.set("Shape: -")
            self.range_var.set("Range: -")
            self.pixel_var.set("Pixel: -")
            self.map_coords_var.set("Map coords: -")
            return
        self.file_var.set(f"File: {self.current_path.name}")
        self.shape_var.set(f"Shape: {self.img.nrows} x {self.img.ncols} x {self.img.nbands}")
        if self.rgb_bands is None:
            self.range_var.set("Range: RGB bands -")
        else:
            self.range_var.set(f"Range: RGB bands {self.rgb_bands}")
        if self.pixel_pos is None:
            self.pixel_var.set("Pixel: -")
            self.map_coords_var.set("Map coords: -")
        else:
            row, col = self.pixel_pos
            self.pixel_var.set(f"Pixel: row={row}, col={col}")
            coords_text = format_map_coordinates(self.img.metadata, row, col)
            self.map_coords_var.set(f"Map coords: {coords_text}")

    def _refresh_plots(self) -> None:
        self._draw_rgb_panel()
        self._draw_spectrum_panel()
        self.canvas.draw_idle()

    def _draw_rgb_panel(self) -> None:
        self.ax_rgb.clear()
        self.ax_rgb.set_title("RGB Preview", fontsize=13, pad=10, color=TEXT_FG)
        self.ax_rgb.set_facecolor("#171b20")
        self.ax_rgb.axis("off")

        if self.rgb_display is None:
            self.ax_rgb.text(
                0.5,
                0.5,
                "Open an ENVI .hdr file\nto display RGB preview",
                ha="center",
                va="center",
                transform=self.ax_rgb.transAxes,
                color=MUTED_FG,
                fontsize=11,
            )
            return

        self.ax_rgb.imshow(self.rgb_display, interpolation="bilinear", aspect="auto")
        self._apply_zoom_limits()
        if self.pixel_pos is not None:
            row, col = self.pixel_pos
            self.ax_rgb.plot(col, row, marker="+", color="red", markersize=14, markeredgewidth=2.2)
            self.ax_rgb.axhline(row, color="white", alpha=0.35, linewidth=0.8)
            self.ax_rgb.axvline(col, color="white", alpha=0.35, linewidth=0.8)

    def _draw_spectrum_panel(self) -> None:
        self.ax_spec.clear()
        self.ax_spec.set_title("Spectral Signature", fontsize=13, pad=10, color=TEXT_FG)
        self.ax_spec.set_facecolor("#171b20")

        if self.spectrum is None or self.pixel_pos is None:
            self.ax_spec.text(
                0.5,
                0.5,
                "Click a pixel in the RGB image",
                ha="center",
                va="center",
                transform=self.ax_spec.transAxes,
                color=MUTED_FG,
                fontsize=11,
            )
            self.ax_spec.set_xticks([])
            self.ax_spec.set_yticks([])
            return

        row, col = self.pixel_pos
        x_axis = self.wavelengths if self.wavelengths is not None else np.arange(self.spectrum.size)
        x_label = "Wavelength (nm)" if self.wavelengths is not None else "Band index"

        finite_vals = self.spectrum[np.isfinite(self.spectrum)]
        baseline = float(np.min(finite_vals)) if finite_vals.size else 0.0
        self.ax_spec.plot(x_axis, self.spectrum, color=ACCENT, linewidth=1.8)
        self.ax_spec.fill_between(x_axis, self.spectrum, baseline, color=ACCENT, alpha=0.10)
        self.ax_spec.set_xlabel(x_label, color=TEXT_FG)
        self.ax_spec.set_ylabel("Reflectance", color=TEXT_FG)
        self.ax_spec.set_title(f"Spectral Signature (row={row}, col={col})", fontsize=12, pad=8, color=TEXT_FG)
        self.ax_spec.grid(True, alpha=0.25, linestyle="--", linewidth=0.8, color="#8a939d")
        self.ax_spec.tick_params(colors=TEXT_FG)
        for spine in self.ax_spec.spines.values():
            spine.set_color("#3a434d")

    def _on_plot_click(self, event) -> None:
        if event.inaxes is not self.ax_rgb or self.img is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if not (0 <= row < self.img.nrows and 0 <= col < self.img.ncols):
            return

        self.pixel_pos = (row, col)
        self.zoom_center = (float(col), float(row))
        self.spectrum = read_pixel_spectrum(self.img, row, col, self.ignore_value)
        self._update_info_panel()
        self._refresh_plots()
        self.status_var.set(f"Pixel ({row}, {col}) selected. Use 'Export spectrum CSV' to save.")

    def _apply_zoom_limits(self) -> None:
        if self.rgb_display is None:
            return
        height, width = self.rgb_display.shape[:2]
        if width <= 0 or height <= 0:
            return

        zoom = max(1.0, min(self.zoom_factor, 20.0))
        self.zoom_factor = zoom
        if self.zoom_center is None:
            center_x = (width - 1) / 2.0
            center_y = (height - 1) / 2.0
        else:
            center_x = min(max(self.zoom_center[0], 0.0), float(width - 1))
            center_y = min(max(self.zoom_center[1], 0.0), float(height - 1))

        half_w = max(width / (2.0 * zoom), 1.0)
        half_h = max(height / (2.0 * zoom), 1.0)
        x_min = max(-0.5, center_x - half_w)
        x_max = min(width - 0.5, center_x + half_w)
        y_min = max(-0.5, center_y - half_h)
        y_max = min(height - 0.5, center_y + half_h)

        self.ax_rgb.set_xlim(x_min, x_max)
        self.ax_rgb.set_ylim(y_max, y_min)

    def _zoom_in(self) -> None:
        if self.rgb_display is None:
            return
        self.zoom_factor = min(self.zoom_factor * 1.25, 20.0)
        self._refresh_plots()

    def _zoom_out(self) -> None:
        if self.rgb_display is None:
            return
        self.zoom_factor = max(self.zoom_factor / 1.25, 1.0)
        self._refresh_plots()

    def _reset_zoom(self) -> None:
        self.zoom_factor = 1.0
        self.zoom_center = None
        if self.rgb_display is not None:
            self._refresh_plots()

    def _export_csv(self) -> None:
        if self.spectrum is None or self.pixel_pos is None:
            messagebox.showinfo("Nothing to export", "Select a pixel first.")
            return

        row, col = self.pixel_pos
        default_name = f"spectrum_r{row}_c{col}.csv"
        out_path = filedialog.asksaveasfilename(
            title="Export spectrum to CSV",
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not out_path:
            return

        x_axis = self.wavelengths if self.wavelengths is not None else np.arange(self.spectrum.size)
        x_col = "wavelength_nm" if self.wavelengths is not None else "band"

        with open(out_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([x_col, "value"])
            for x, y in zip(x_axis, self.spectrum):
                writer.writerow([float(x), "" if np.isnan(y) else float(y)])

        self.status_var.set(f"Spectrum exported: {out_path}")
        messagebox.showinfo("Export complete", f"Spectrum saved to:\n{out_path}")


def main() -> None:
    root = tk.Tk()
    HyperspectralViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
