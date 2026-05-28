"""Gaia HEALPix Cache -- download and cache Gaia catalog data for offline work."""

from .cache import clear_cache, get_cache_dir
from .celestial import center_at_date, conform_coordinates, gaia_to_topocentric
from .gaia import (
    get_pix_range,
    get_pixlist,
    haversine,
    match_catalogs,
    parse_md5sum,
    query,
    query_rectangular,
    query_spectra,
    query_spectra_rectangular,
    retrieve_gaia_data,
    spectro_wavelengths,
)
from .products import (
    COLUMNS_OF_INTEREST,
    GaiaProduct,
    get_product,
    list_products,
    register_product,
    unregister_product,
)

__all__ = [
    "COLUMNS_OF_INTEREST",
    "GaiaProduct",
    "center_at_date",
    "clear_cache",
    "conform_coordinates",
    "get_cache_dir",
    "get_pix_range",
    "get_pixlist",
    "get_product",
    "gaia_to_topocentric",
    "haversine",
    "list_products",
    "match_catalogs",
    "parse_md5sum",
    "query",
    "query_rectangular",
    "query_spectra",
    "query_spectra_rectangular",
    "register_product",
    "retrieve_gaia_data",
    "spectro_wavelengths",
    "unregister_product",
]
