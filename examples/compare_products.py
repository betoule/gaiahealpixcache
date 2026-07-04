#!/usr/bin/env python3
"""Compare the spatial coverage of the three main Gaia data products.

Picks a random sky position, queries a 1-degree cone for each product
(source, sampled_spectra, continuous_spectra), and plots the positions
of the returned sources so you can see which sources have which
spectroscopy product available.
"""

import os
import sys

import numpy as np

from gaiahealpixcache import query, query_spectra, match_catalogs, spectro_wavelengths


def main():
    # ---- sky position ------------------------------------------------------
    # Fixed position so repeated runs use the cached tiles.
    ra_deg = 76.377
    dec_deg = 52.831
    radius_arcmin = 20.0
    print(f"Query centre: RA = {ra_deg:.4f}°, Dec = {dec_deg:.4f}°")
    print(f"Radius: {radius_arcmin / 60:.2f}°")

    # ---- query source catalogue --------------------------------------------
    print("\nQuerying  source          ...", end=" ", flush=True)
    sources = query(ra_deg, dec_deg, radius_arcmin)
    print(f"{len(sources)} sources")

    src_ra = np.asarray(sources["ra"])
    src_dec = np.asarray(sources["dec"])

    # ---- query sampled spectra ---------------------------------------------
    print("Querying  sampled_spectra ...", end=" ", flush=True)
    try:
        meta_s, flux_s = query_spectra(
            ra_deg, dec_deg, radius_arcmin, product="sampled_spectra"
        )
        print(f"{len(meta_s)} sources")
    except Exception as exc:
        print(f"FAILED: {exc}")
        meta_s = None

    # ---- query continuous spectra ------------------------------------------
    print("Querying  continuous_spectra ...", end=" ", flush=True)
    try:
        meta_c, flux_c = query_spectra(
            ra_deg, dec_deg, radius_arcmin, product="continuous_spectra"
        )
        print(f"{len(meta_c)} sources")
    except ImportError:
        print(
            "SKIPPED — gaiaxpy not installed.\n"
            "  Install with:  pip install gaiahealpixcache[spectro]"
        )
        meta_c = None
    except Exception as exc:
        print(f"FAILED: {exc}")
        meta_c = None

    # ---- scatter plot of spatial coverage -----------------------------------
    wavelengths = spectro_wavelengths("sampled_spectra")
    _plot_spatial(ra_deg, dec_deg, src_ra, src_dec, meta_s, meta_c)
    _plot_ratio(wavelengths, meta_s, flux_s, meta_c, flux_c)


def _select_backend():
    """Return pyplot, falling back to Agg when interactive backends fail."""
    import matplotlib

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    # Some interactive backends (e.g. QtAgg) can be imported but fail
    # when a figure is created (missing Qt bindings).  Attempt a
    # throw-away figure to force backend initialisation, and fall back
    # to Agg on failure.
    try:
        fig = plt.figure()
        plt.close(fig)
    except ImportError:
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    return plt


def _plot_spatial(ra_deg, dec_deg, src_ra, src_dec, meta_s, meta_c):
    try:
        plt = _select_backend()
    except ImportError:
        print(
            "matplotlib is required for plotting.\n"
            "  Install with:  pip install matplotlib"
        )
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        src_ra, src_dec,
        facecolors="none", edgecolors="steelblue", s=12, linewidths=0.6,
        label=f"source ({len(src_ra)})",
    )

    if meta_s is not None and len(meta_s):
        ax.scatter(
            meta_s["ra"], meta_s["dec"],
            marker="+", color="coral", s=30, linewidths=0.8,
            label=f"sampled_spectra ({len(meta_s)})",
        )

    if meta_c is not None and len(meta_c):
        ax.scatter(
            meta_c["ra"], meta_c["dec"],
            marker="x", color="limegreen", s=30, linewidths=0.8,
            label=f"continuous_spectra ({len(meta_c)})",
        )

    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(
        f"Cone centred at RA = {ra_deg:.3f}°, Dec = {dec_deg:.3f}°  (1° radius)"
    )
    ax.legend(markerscale=1.5)
    ax.set_aspect(1.0 / np.cos(np.radians(dec_deg)))
    fig.tight_layout()

    out = "compare_products.png"
    fig.savefig(out, dpi=150)
    print(f"\nSpatial coverage plot saved to {out}")


def _plot_ratio(wavelengths, meta_s, flux_s, meta_c, flux_c):
    """Plot the ratio sampled/continuous for sources present in both sets.

    A tight scatter around 1.0 confirms the gaiaxpy calibration produces
    fluxes consistent with the pre-sampled reference archive.
    """
    if meta_s is None or meta_c is None:
        return
    if len(meta_s) == 0 or len(meta_c) == 0:
        return

    idx_s, idx_c = match_catalogs(meta_s, meta_c)
    n_common = len(idx_s)
    print(f"\nSources with both sampled and continuous spectra: {n_common}")

    if n_common == 0:
        return

    ratio = flux_s[idx_s] / flux_c[idx_c]
    # Clip extreme outliers that would squash the y-axis
    ratio = np.clip(ratio, 0.5, 1.5)

    fig, ax = _select_backend().subplots(figsize=(8, 5))

    for i in range(n_common):
        ax.plot(
            wavelengths, ratio[i],
            color="steelblue", alpha=0.08, linewidth=0.4,
        )

    # Median ratio curve (bold)
    med = np.median(ratio, axis=0)
    ax.plot(
        wavelengths, med,
        color="coral", linewidth=1.8, label=f"median (n={n_common})",
    )

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.6)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Flux ratio  sampled / continuous")
    ax.set_title("Consistency check: sampled vs continuous spectra")
    ax.legend()
    ax.set_ylim(0.5, 1.5)
    fig.tight_layout()

    out = "spectral_ratio.png"
    fig.savefig(out, dpi=150)
    print(f"Spectral ratio plot saved to {out}")


if __name__ == "__main__":
    main()
