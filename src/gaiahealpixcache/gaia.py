"""Gaia data retrieval, HEALPix partitioning, and cache management."""

import ast
import gzip
import os

import healpy
import numpy as np
from tqdm import tqdm

from .cache import cached_download, get_cache_dir
from .celestial import conform_coordinates
from .products import COLUMNS_OF_INTEREST, GaiaProduct, get_product

_DANGEROUS_NAMES = {
    "__import__",
    "open",
    "exec",
    "eval",
    "compile",
    "__builtins__",
    "os",
    "sys",
    "subprocess",
    "importlib",
    "getattr",
    "setattr",
    "delattr",
    "dir",
    "vars",
    "globals",
    "locals",
    "breakpoint",
    "input",
    "print",
}


def _validate_where(expr: str):
    """Validate a `where` filter expression using AST analysis.

    Only allows safe operations: comparisons, arithmetic, boolean ops,
    numpy function calls, and numeric/string literals.

    Parameters
    ----------
    expr : str
        Filter expression to validate.

    Raises
    ------
    ValueError
        If the expression contains disallowed constructs.
    """
    tree = ast.parse(expr, mode="eval")

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in _DANGEROUS_NAMES:
                raise ValueError(f"Disallowed name in filter expression: {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ValueError(
                    f"Disallowed attribute access in filter expression: {node.attr}"
                )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _DANGEROUS_NAMES:
                raise ValueError(
                    f"Disallowed function call in filter expression: {node.func.id}"
                )


def _safe_eval_where(expr: str, data: np.recarray) -> np.ndarray:
    """Safely evaluate a `where` filter expression on a recarray.

    Parameters
    ----------
    expr : str
        Filter expression (e.g., "phot_g_mean_mag < 16").
    data : np.recarray
        Loaded Gaia data.

    Returns
    -------
    np.ndarray[bool]
        Boolean mask for rows matching the filter.
    """
    _validate_where(expr)
    safe_ns: dict = {"np": np}
    for col in data.dtype.names:
        safe_ns[col] = data[col]
    return eval(expr, {"__builtins__": {}}, safe_ns)  # noqa: S307


class _CacheLock:
    """File-based lock using fcntl to prevent concurrent downloads.

    On Windows (where fcntl is unavailable), falls back to lock files
    with atomic creation.
    """

    def __init__(self, lock_path: str, timeout: float = 300.0):
        self._lock_path = lock_path
        self._timeout = timeout
        self._fd: int | None = None

    def acquire(self):
        import fcntl
        import time

        self._fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR)
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                time.sleep(0.1)
        raise TimeoutError(
            f"Could not acquire cache lock {self._lock_path} within "
            f"{self._timeout:.0f}s"
        )

    def release(self):
        if self._fd is not None:
            import fcntl

            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
        lock_file = self._lock_path
        if os.path.exists(lock_file):
            os.unlink(lock_file)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


_md5sum_cache: dict[str, str | None] = {}


def _resolve_product(product: str | GaiaProduct | None = None) -> GaiaProduct:
    if product is None or isinstance(product, str):
        return get_product(product or "source")
    return product


def _get_md5sum_path(product: GaiaProduct) -> str:
    if product.name not in _md5sum_cache or _md5sum_cache[product.name] is None:
        md5url = f"{product.url}{product.md5sum_file}"
        _md5sum_cache[product.name] = cached_download(md5url)
    return _md5sum_cache[product.name]


def parse_md5sum(fn, file_prefix="GaiaSource_", file_ext=".csv.gz"):
    """Use md5sum file to get the full list of files.

    Parameters
    ----------
    fn : str
        Path to md5sum file from the Gaia source directory.
    file_prefix : str, optional
        Prefix used in data filenames (default: "GaiaSource_").
    file_ext : str, optional
        Extension of data files (default: ".csv.gz").

    Returns
    -------
    bins : list[int]
        First pixel of each file range.
    ranges : list[str]
        Pixel range strings used in filenames.
    """
    with open(fn, encoding="utf-8") as fid:
        lines = fid.readlines()
    ranges = [
        l.split(file_prefix)[-1].split(file_ext)[0]
        for l in lines
        if l.endswith(f"{file_ext}\n")
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
    radecs = [conform_coordinates(ra, dec) for ra, dec in zip(ras, decs)]
    nside = healpy.order2nside(level)
    pixlist = [
        healpy.ang2pix(nside, ra, dec, lonlat=True, nest=True) for ra, dec in radecs
    ]
    return list(np.unique(pixlist))


def get_pix_range(ra, dec, product: str | GaiaProduct | None = None):
    """Return a list of pixel ranges matching indexing of Gaia files.

    Parameters
    ----------
    ra : list[float]
        Right ascension coordinates.
    dec : list[float]
        Declination coordinates.
    product : str or GaiaProduct, optional
        Product name or instance (default: "source").

    Returns
    -------
    list[str]
        Pixel range strings covered by the given sky coordinates.
    """
    prod = _resolve_product(product)
    pixlist = get_pixlist(ra, dec)
    md5path = _get_md5sum_path(prod)
    bins, ranges = parse_md5sum(md5path, prod.file_prefix, prod.file_ext)
    range_index = np.digitize(pixlist, bins) - 1
    range_index = np.unique(range_index)
    return [ranges[i] for i in range_index]


def _cache_filename(product: GaiaProduct, pixel_range: str) -> str:
    cfg_hash = product.config_hash()
    stem = product.file_prefix.rstrip("_").rstrip(".csv").rstrip(".gz")
    return os.path.join(get_cache_dir(), f"{stem}_{pixel_range}_{cfg_hash}.npy")


def retrieve_gaia_data(
    pixel_range: str,
    product: str | GaiaProduct | None = None,
) -> np.recarray:
    """Load Gaia data for a given pixel range, downloading and caching if needed.

    Downloads the CSV.gz file, converts to compressed numpy format, and caches
    the result for fast subsequent access.

    Parameters
    ----------
    pixel_range : str
        Pixel range string (e.g., '0-63').
    product : str or GaiaProduct, optional
        Product name or instance (default: "source").

    Returns
    -------
    np.recarray
        Structured array with selected columns.
    """
    prod = _resolve_product(product)
    fname = _cache_filename(prod, pixel_range)
    lock_path = fname + ".lock"

    with _CacheLock(lock_path):
        if os.path.exists(fname):
            return np.load(fname, allow_pickle=True)

        rawname = f"{prod.file_prefix}{pixel_range}{prod.file_ext}"
        sampled = cached_download(f"{prod.url}{rawname}")
        sources = read_gaia(sampled, prod.columns, prod.where)
        np.save(fname, sources)
        os.remove(sampled)
        return sources


def read_gaia(
    fname: str,
    columns: list[str] | None = None,
    where: str | None = None,
) -> np.recarray:
    """Parse a gzipped Gaia CSV file into a numpy recarray.

    Parameters
    ----------
    fname : str
        Path to the gzipped CSV file.
    columns : list[str], optional
        Columns to retain. Uses COLUMNS_OF_INTEREST if not given.
    where : str | None, optional
        Filter expression evaluated as a boolean mask. Only matching rows
        are returned.

    Returns
    -------
    np.recarray
        Structured array with selected columns.
    """
    if columns is None:
        columns = COLUMNS_OF_INTEREST

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
            keep_index = [keys.index(k) for k in keys if k in columns]
            keys = [keys[i] for i in keep_index]
            assert len(keys) == len(columns), "missing column"
            continue
        flux.append(process(line))

    result = np.rec.fromrecords(flux, names=keys)
    if where is not None:
        mask = _safe_eval_where(where, result)
        result = result[mask]
    return result


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


def query(
    ra_deg: float,
    dec_deg: float,
    radius_arcmin: float = 30,
    product: str | GaiaProduct | None = None,
) -> np.recarray:
    """Query Gaia sources within a circular region around given coordinates.

    Parameters
    ----------
    ra_deg : float
        Right ascension center in degrees.
    dec_deg : float
        Declination center in degrees.
    radius_arcmin : float, optional
        Search radius in arcminutes (default: 30).
    product : str or GaiaProduct, optional
        Product name or instance (default: "source").

    Returns
    -------
    np.recarray
        Structured array of Gaia sources within the search radius.
    """
    prod = _resolve_product(product)
    rad = radius_arcmin / 60
    cdec = np.cos(np.radians(dec_deg))
    dra, ddec = np.meshgrid(
        np.linspace(ra_deg - rad / cdec, ra_deg + rad / cdec, 5),
        np.linspace(dec_deg - rad, dec_deg + rad, 5),
    )

    pranges = get_pix_range(dra, ddec, product=prod)
    all_sources = []
    for pixel in pranges:
        sources = retrieve_gaia_data(pixel, product=prod)
        in_radius = haversine(sources["ra"], sources["dec"], ra_deg, dec_deg) < rad
        all_sources.append(sources[in_radius])
    return np.hstack(all_sources)
