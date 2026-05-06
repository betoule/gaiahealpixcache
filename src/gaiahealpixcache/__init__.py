"""Gaia HEALPix Cache -- download and cache Gaia catalog data for offline work."""

from .cache import clear_cache, get_cache_dir
from .celestial import center_at_date, gaia_to_topocentric
from .gaia import (
    get_pix_range,
    get_pixlist,
    haversine,
    parse_md5sum,
    query,
    retrieve_gaia_data,
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
    "get_cache_dir",
    "get_pix_range",
    "get_pixlist",
    "get_product",
    "gaia_to_topocentric",
    "haversine",
    "list_products",
    "parse_md5sum",
    "query",
    "register_product",
    "retrieve_gaia_data",
    "unregister_product",
]
