"""Gaia data retrieval, HEALPix partitioning, and cache management."""

import ast
import gzip
import os

import healpy
import numpy as np
from tqdm import tqdm
from filelock import FileLock

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
    ext = ".npz" if product.spectro else ".npy"
    return os.path.join(get_cache_dir(), f"{stem}_{pixel_range}_{cfg_hash}{ext}")


def retrieve_gaia_data(
    pixel_range: str,
    product: str | GaiaProduct | None = None,
) -> np.recarray | tuple[np.recarray, np.ndarray]:
    """Load Gaia data for a given pixel range, downloading and caching if needed.

    Downloads the CSV.gz file, converts to compressed numpy format, and caches
    the result for fast subsequent access. For spectroscopy products, returns
    a tuple of (metadata recarray, flux 2D array).

    Parameters
    ----------
    pixel_range : str
        Pixel range string (e.g., '0-63').
    product : str or GaiaProduct, optional
        Product name or instance (default: "source").

    Returns
    -------
    np.recarray or tuple[np.recarray, np.ndarray]
        Structured array with selected columns, or (meta, flux) for spectro.
    """
    prod = _resolve_product(product)
    fname = _cache_filename(prod, pixel_range)
    lock_path = fname + ".lock"

    with FileLock(lock_path):
        if os.path.exists(fname):
            return _load_cached(fname, prod.spectro)

        rawname = f"{prod.file_prefix}{pixel_range}{prod.file_ext}"
        sampled = cached_download(f"{prod.url}{rawname}")
        if prod.spectro:
            meta, flux = read_gaia_spectra(
                sampled, prod.spectro_meta_cols, prod.spectro_flux_cols, prod.where
            )
            np.savez(fname, meta=meta, flux=flux)
        else:
            sources = read_gaia(sampled, prod.columns, prod.where)
            np.save(fname, sources)
            meta, flux = sources, None  # type: ignore
        os.remove(sampled)
        return (meta, flux) if prod.spectro else meta


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
    source_id_pos = None

    def process(line):
        vals = line.replace(b"[", b"").replace(b"]", b"").replace(b'"', b"").split(b",")
        result = []
        for pos, i in enumerate(keep_index):
            v = vals[i].replace(b"null", b"nan")
            result.append(int(v) if pos == source_id_pos else float(v))
        return result

    for line in lines:
        if line[0] == 35:
            continue
        if line[0] == 115:
            keys = [k.strip().decode() for k in line.split(b",")]
            keep_index = [keys.index(k) for k in keys if k in columns]
            keys = [keys[i] for i in keep_index]
            source_id_pos = keys.index("source_id") if "source_id" in keys else None
            assert len(keys) == len(columns), "missing column"
            continue
        flux.append(process(line))

    result = np.rec.fromrecords(flux, names=keys)
    if where is not None:
        mask = _safe_eval_where(where, result)
        result = result[mask]
    return result


def _load_cached(
    fname: str, spectro: bool
) -> np.recarray | tuple[np.recarray, np.ndarray]:
    """Load cached data from disk.

    Parameters
    ----------
    fname : str
        Path to the cached file (.npy or .npz).
    spectro : bool
        If True, load as .npz with meta and flux arrays.

    Returns
    -------
    np.recarray or tuple[np.recarray, np.ndarray]
        Loaded data, or (meta, flux) for spectro products.
    """
    if spectro:
        data = np.load(fname, allow_pickle=True)
        return data["meta"], data["flux"]
    return np.load(fname, allow_pickle=True)


def read_gaia_spectra(
    fname: str,
    meta_cols: list[str] | None = None,
    flux_range: tuple[int, int] | None = None,
    where: str | None = None,
) -> tuple[np.recarray, np.ndarray]:
    """Parse a gzipped Gaia spectroscopy CSV file.

    Extracts metadata columns and a 2D flux array from the sampled mean
    spectrum format.

    Parameters
    ----------
    fname : str
        Path to the gzipped CSV file.
    meta_cols : list[str], optional
        Metadata columns to extract (default: ["source_id", "ra", "dec"]).
    flux_range : tuple[int, int], optional
        Column index range [start, stop) for flux values
        (default: (4, 347) for 343 flux points).
    where : str | None, optional
        Filter expression evaluated on metadata. Only matching rows
        are returned.

    Returns
    -------
    tuple[np.recarray, np.ndarray]
        (metadata recarray, flux 2D array of shape (n_sources, n_flux_points))
    """
    if meta_cols is None:
        meta_cols = ["source_id", "ra", "dec"]
    if flux_range is None:
        flux_range = (4, 347)

    fid = gzip.GzipFile(fname)
    lines = tqdm(fid.readlines())
    meta_records: list[tuple] = []
    flux_records: list[list[float]] = []
    meta_index: list[int] = []
    keys: list[str] = []

    for line in lines:
        if line[0] == 35:
            continue
        if line[0] == 115:
            keys = [k.strip().decode() for k in line.split(b",")]
            meta_index = [keys.index(k) for k in meta_cols if k in keys]
            assert len(meta_index) == len(
                meta_cols
            ), f"Missing metadata columns: {meta_cols}"
            continue

        vals = line.replace(b"[", b"").replace(b"]", b"").replace(b'"', b"").split(b",")
        meta_vals = []
        for i in meta_index:
            v = vals[i].replace(b"null", b"nan")
            if i == 0:
                meta_vals.append(int(v))
            else:
                meta_vals.append(float(v))
        meta_records.append(tuple(meta_vals))

        flux_vals = []
        for j in range(flux_range[0], flux_range[1]):
            v = vals[j].replace(b"null", b"nan")
            flux_vals.append(float(v))
        flux_records.append(flux_vals)

    meta = np.rec.fromrecords(meta_records, names=meta_cols)
    flux = np.array(flux_records, dtype=np.float32)

    if where is not None:
        mask = _safe_eval_where(where, meta)
        meta = meta[mask]
        flux = flux[mask]

    return meta, flux


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


def query_spectra(
    ra_deg: float,
    dec_deg: float,
    radius_arcmin: float = 30,
    product: str | GaiaProduct | None = None,
) -> tuple[np.recarray, np.ndarray]:
    """Query Gaia spectra within a circular region around given coordinates.

    Parameters
    ----------
    ra_deg : float
        Right ascension center in degrees.
    dec_deg : float
        Declination center in degrees.
    radius_arcmin : float, optional
        Search radius in arcminutes (default: 30).
    product : str or GaiaProduct, optional
        Product name or instance (default: "sampled_spectra").

    Returns
    -------
    tuple[np.recarray, np.ndarray]
        (metadata recarray, flux 2D array) for sources within the search
        radius. Flux has shape (n_sources, n_flux_points).
    """
    prod = _resolve_product(product or "sampled_spectra")
    rad = radius_arcmin / 60
    cdec = np.cos(np.radians(dec_deg))
    dra, ddec = np.meshgrid(
        np.linspace(ra_deg - rad / cdec, ra_deg + rad / cdec, 5),
        np.linspace(dec_deg - rad, dec_deg + rad, 5),
    )

    pranges = get_pix_range(dra, ddec, product=prod)
    all_meta: list[np.recarray] = []
    all_flux: list[np.ndarray] = []
    for pixel in pranges:
        meta, flux = retrieve_gaia_data(pixel, product=prod)
        in_radius = haversine(meta["ra"], meta["dec"], ra_deg, dec_deg) < rad
        all_meta.append(meta[in_radius])
        all_flux.append(flux[in_radius])
    return np.hstack(all_meta), np.vstack(all_flux)


def _in_rectangle(ra, dec, ra_min, ra_max, dec_min, dec_max):
    """Boolean mask for sources inside a rectangular region.

    Handles RA wrapping across the 0/360 boundary.
    """
    if ra_max < ra_min:
        ra_mask = (ra >= ra_min) | (ra <= ra_max)
    else:
        ra_mask = (ra >= ra_min) & (ra <= ra_max)
    return ra_mask & (dec >= dec_min) & (dec <= dec_max)


def _cover_grid(ra_min, ra_max, dec_min, dec_max):
    """Return meshgrid coords that cover the rectangular region for tile discovery."""
    if ra_max < ra_min:
        ra_max += 360
    dra, ddec = np.meshgrid(
        np.linspace(ra_min, ra_max, 5),
        np.linspace(dec_min, dec_max, 5),
    )
    dra = dra % 360
    return dra, ddec


def query_rectangular(
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    product: str | GaiaProduct | None = None,
) -> np.recarray:
    """Query Gaia sources within a rectangular region.

    Parameters
    ----------
    ra_min : float
        Minimum right ascension in degrees.
    ra_max : float
        Maximum right ascension in degrees.
    dec_min : float
        Minimum declination in degrees.
    dec_max : float
        Maximum declination in degrees.
    product : str or GaiaProduct, optional
        Product name or instance (default: "source").

    Returns
    -------
    np.recarray
        Structured array of Gaia sources within the rectangular region.
    """
    prod = _resolve_product(product)
    dra, ddec = _cover_grid(ra_min, ra_max, dec_min, dec_max)

    pranges = get_pix_range(dra, ddec, product=prod)
    all_sources = []
    for pixel in set(pranges):
        sources = retrieve_gaia_data(pixel, product=prod)
        mask = _in_rectangle(
            sources["ra"], sources["dec"], ra_min, ra_max, dec_min, dec_max
        )
        all_sources.append(sources[mask])
    return np.hstack(all_sources)


def query_spectra_rectangular(
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    product: str | GaiaProduct | None = None,
) -> tuple[np.recarray, np.ndarray]:
    """Query Gaia spectra within a rectangular region.

    Parameters
    ----------
    ra_min : float
        Minimum right ascension in degrees.
    ra_max : float
        Maximum right ascension in degrees.
    dec_min : float
        Minimum declination in degrees.
    dec_max : float
        Maximum declination in degrees.
    product : str or GaiaProduct, optional
        Product name or instance (default: "sampled_spectra").

    Returns
    -------
    tuple[np.recarray, np.ndarray]
        (metadata recarray, flux 2D array) for sources within the rectangular
        region. Flux has shape (n_sources, n_flux_points).
    """
    prod = _resolve_product(product or "sampled_spectra")
    dra, ddec = _cover_grid(ra_min, ra_max, dec_min, dec_max)

    pranges = get_pix_range(dra, ddec, product=prod)
    all_meta: list[np.recarray] = []
    all_flux: list[np.ndarray] = []
    for pixel in set(pranges):
        meta, flux = retrieve_gaia_data(pixel, product=prod)
        mask = _in_rectangle(meta["ra"], meta["dec"], ra_min, ra_max, dec_min, dec_max)
        all_meta.append(meta[mask])
        all_flux.append(flux[mask])
    return np.hstack(all_meta), np.vstack(all_flux)


_SPECTRO_WAVELENGTHS = np.arange(336, 1022, 2, dtype=np.float64)


def spectro_wavelengths(product: str | GaiaProduct | None = None) -> np.ndarray:
    """Return the wavelengths corresponding to the sampled spectrum flux points.

    Gaia sampled mean spectra use 343 wavelength positions from 336 to 1020 nm
    with a step of 2 nm. If a custom product specifies a narrower flux range,
    only the corresponding wavelengths are returned.

    Parameters
    ----------
    product : str or GaiaProduct, optional
        Product name or instance (default: "sampled_spectra").

    Returns
    -------
    np.ndarray
        Array of wavelength values in nanometers.
    """
    prod = _resolve_product(product or "sampled_spectra")
    start, stop = prod.spectro_flux_cols
    return _SPECTRO_WAVELENGTHS[start - 4 : stop - 4].copy()


def match_catalogs(
    cat_a: np.recarray,
    cat_b: np.recarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match two catalogs by source_id using an inner join.

    Returns index arrays so that ``cat_a[idx_a]`` and ``cat_b[idx_b]``
    contain the same sources. Uses a hash-based algorithm with O(n+m)
    time complexity.

    Parameters
    ----------
    cat_a : np.recarray
        First catalog (must have a ``source_id`` column).
    cat_b : np.recarray
        Second catalog (must have a ``source_id`` column).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(idx_a, idx_b)`` index arrays where each pair ``(idx_a[k], idx_b[k])``
        corresponds to the same source_id.
    """
    ids_a = cat_a["source_id"]
    ids_b = cat_b["source_id"]

    if len(ids_a) <= len(ids_b):
        id_map: dict[int, int] = {int(sid): i for i, sid in enumerate(ids_a)}
        match_b: list[int] = []
        match_a: list[int] = []
        for j, sid in enumerate(ids_b):
            if int(sid) in id_map:
                match_a.append(id_map[int(sid)])
                match_b.append(j)
    else:
        id_map = {int(sid): i for i, sid in enumerate(ids_b)}
        match_a = []
        match_b = []
        for i, sid in enumerate(ids_a):
            if int(sid) in id_map:
                match_a.append(i)
                match_b.append(id_map[int(sid)])

    return np.array(match_a), np.array(match_b)
