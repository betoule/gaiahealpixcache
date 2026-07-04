#!/usr/bin/env python3
"""Compare the spatial coverage of the three main Gaia data products.

Picks a random sky position, queries a 1-degree cone for each product
(source, sampled_spectra, continuous_spectra), and plots the positions
of the returned sources so you can see which sources have which
spectroscopy product available.
"""

import sys

import numpy as np

from gaiahealpixcache import query, query_spectra


def main():
    # ---- sky position ------------------------------------------------------
    # Fixed position so repeated runs use the cached tiles.
    ra_deg = 76.377
    dec_deg = 52.831
    radius_arcmin = 60.0  # 1 degree
    print(f"Query centre: RA = {ra_deg:.4f}°, Dec = {dec_deg:.4f}°")
    print("Radius: 1.0°")

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

    # ---- scatter plot ------------------------------------------------------
    _plot(ra_deg, dec_deg, src_ra, src_dec, meta_s, meta_c)


def _plot(ra_deg, dec_deg, src_ra, src_dec, meta_s, meta_c):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is required for plotting.\n"
            "  Install with:  pip install matplotlib"
        )
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # All sources (empty circles)
    ax.scatter(
        src_ra, src_dec,
        facecolors="none", edgecolors="steelblue", s=12, linewidths=0.6,
        label=f"source ({len(src_ra)})",
    )

    # Sampled spectra (plus markers)
    if meta_s is not None and len(meta_s):
        ax.scatter(
            meta_s["ra"], meta_s["dec"],
            marker="+", color="coral", s=30, linewidths=0.8,
            label=f"sampled_spectra ({len(meta_s)})",
        )

    # Continuous spectra (cross markers)
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
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
