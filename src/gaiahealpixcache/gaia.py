import gzip
import os
from pathlib import Path

import healpy
import numpy as np
from tqdm import tqdm

from .cache import cached_download, get_cache_dir

GAIA_SOURCE_URL = "https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/"
MD5SUM_URL = f"{GAIA_SOURCE_URL}_MD5SUM.txt"

COLUMNS_OF_INTEREST = [
    "source_id",
    "ra",
    "ra_error",
    "dec",
    "dec_error",
    "parallax",
    "parallax_error",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "phot_g_mean_flux",
    "phot_g_mean_flux_error",
    "phot_bp_mean_flux",
    "phot_bp_mean_flux_error",
    "phot_rp_mean_flux",
    "phot_rp_mean_flux_error",
    "radial_velocity",
    "radial_velocity_error",
]

_md5sum_path = None


def _get_md5sum_path():
    global _md5sum_path
    if _md5sum_path is None:
        _md5sum_path = cached_download(MD5SUM_URL)
    return _md5sum_path


def parse_md5sum(fn):
    """Use md5sum file to get the full list of files.

    Parameters
    ----------
    fn : str
        Path to md5sum file from the Gaia source directory.

    Returns
    -------
    bins : list[int]
        First pixel of each file range.
    ranges : list[str]
        Pixel range strings used in filenames.
    """
    with open(fn) as fid:
        lines = fid.readlines()
    ranges = [
        l.split("GaiaSource_")[-1].split(".csv.gz")[0]
        for l in lines
        if l.endswith(".gz\n")
    ]
    bins = [int(b.split("-")[0]) for b in ranges]
    return bins, ranges


def get_pixlist(ras, decs, level=8):
    """Return list of healpix pixels matching ra, dec coordinates.

    Parameters
    ----------
    ras : array-like
        Right ascension coordinates in degrees.
    decs : array-like
        Declination coordinates in degrees.
    level : int, optional
        HEALPix level (default: 8, nside=256).

    Returns
    -------
    list[int]
        List of HEALPix pixel indices in [0, 786431].
    """
    nside = healpy.order2nside(level)
    pixlist = [
        healpy.ang2pix(nside, ra, dec, lonlat=True, nest=True)
        for ra, dec in zip(ras, decs)
    ]
    return list(np.unique(pixlist))


def get_pix_range(ra, dec):
    """Return a list of pixel ranges matching indexing of Gaia files.

    Parameters
    ----------
    ra : list[float]
        Right ascension coordinates.
    dec : list[float]
        Declination coordinates.

    Returns
    -------
    list[str]
        Pixel range strings covered by the given sky coordinates.
    """
    pixlist = get_pixlist(ra, dec)
    bins, ranges = parse_md5sum(_get_md5sum_path())
    range_index = np.digitize(pixlist, bins) - 1
    range_index = np.unique(range_index)
    return [ranges[i] for i in range_index]


def retrieve_gaia_data(pixel_range):
    """Load Gaia data for a given pixel range, downloading and caching if needed.

    Downloads the CSV.gz file, converts to compressed numpy format, and caches
    the result for fast subsequent access.

    Parameters
    ----------
    pixel_range : str
        Pixel range string (e.g., '0-63').

    Returns
    -------
    np.recarray
        Structured array with columns from COLUMNS_OF_INTEREST.
    """
    fname = os.path.join(get_cache_dir(), f"GaiaSource_{pixel_range}.npy")
    if os.path.exists(fname):
        return np.load(fname, allow_pickle=True)

    rawname = f"GaiaSource_{pixel_range}.csv.gz"
    sampled = cached_download(f"{GAIA_SOURCE_URL}{rawname}")
    sources = read_gaia(sampled)
    np.save(fname, sources)
    os.remove(sampled)
    return sources


def read_gaia(fname):
    """Parse a gzipped Gaia CSV file into a numpy recarray.

    Parameters
    ----------
    fname : str
        Path to the gzipped CSV file.

    Returns
    -------
    np.recarray
        Structured array with selected columns.
    """
    fid = gzip.GzipFile(fname)
    lines = tqdm(fid.readlines())
    flux = []
    keep_index = None
    keys = None

    def process(line):
        vals = line.replace(b"[", b"").replace(b"]", b"").replace(b'"', b"").split(b",")
        return [float(vals[i].replace(b"null", b"nan")) for i in keep_index]

    for line in lines:
        if line[0] == 35:
            continue
        if line[0] == 115:
            keys = [k.strip().decode() for k in line.split(b",")]
            keep_index = [keys.index(k) for k in keys if k in COLUMNS_OF_INTEREST]
            keys = [keys[i] for i in keep_index]
            assert len(keys) == len(COLUMNS_OF_INTEREST), "missing column"
            continue
        flux.append(process(line))

    return np.rec.fromrecords(flux, names=keys)


def haversine(ra1, dec1, ra2, dec2):
    """Calculate the great-circle distance between two points on a sphere.

    Parameters
    ----------
    ra1, dec1 : float or array-like
        First point coordinates in degrees.
    ra2, dec2 : float or array-like
        Second point coordinates in degrees.

    Returns
    -------
    float or np.ndarray
        Angular distance in degrees.
    """
    _ra1, _dec1, _ra2, _dec2 = (
        np.radians(ra1),
        np.radians(dec1),
        np.radians(ra2),
        np.radians(dec2),
    )
    dlambda = np.array(_ra1 - _ra2)
    return np.degrees(
        np.arccos(
            np.sin(_dec1) * np.sin(_dec2)
            + np.cos(_dec1) * np.cos(_dec2) * np.cos(dlambda)
        )
    )


def query(ra_deg, dec_deg, radius_arcmin=30):
    """Query Gaia sources within a circular region around given coordinates.

    Parameters
    ----------
    ra_deg : float
        Right ascension center in degrees.
    dec_deg : float
        Declination center in degrees.
    radius_arcmin : float, optional
        Search radius in arcminutes (default: 30).

    Returns
    -------
    np.recarray
        Structured array of Gaia sources within the search radius.
    """
    rad = radius_arcmin / 60
    dra, ddec = np.meshgrid(
        np.linspace(ra_deg - rad, ra_deg + rad, 5),
        np.linspace(dec_deg - rad, dec_deg + rad, 5),
    )

    pranges = get_pix_range(dra, ddec)
    all_sources = []
    for pixel in pranges:
        sources = retrieve_gaia_data(pixel)
        in_radius = haversine(sources["ra"], sources["dec"], ra_deg, dec_deg) < rad
        all_sources.append(sources[in_radius])
    return np.hstack(all_sources)
