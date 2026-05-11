"""Configurable Gaia product definitions and persistent registry."""

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass(frozen=True)
class GaiaProduct:
    """Configuration for a Gaia data product.

    Attributes
    ----------
    name : str
        Unique identifier for the product.
    url : str
        Base download URL for the product files.
    md5sum_file : str
        Name of the MD5SUM index file.
    file_prefix : str
        Prefix used in data filenames (e.g., "GaiaSource_").
    file_ext : str
        Extension of data files (e.g., ".csv.gz").
    columns : list[str]
        Columns to retain when parsing CSV files.
    where : str | None
        Optional filter expression evaluated as a boolean mask on the loaded
        recarray. Only rows where the expression evaluates to True are kept
        in the cached .npy file. Column names refer to Gaia column names
        (e.g., "phot_g_mean_mag < 16").
    """

    name: str
    url: str
    md5sum_file: str
    file_prefix: str
    file_ext: str
    columns: list[str] = field(default_factory=list)
    where: str | None = field(default=None)

    def config_hash(self):
        """Return a short hash of the product configuration.

        Used to make cache filenames unique per product+columns+filter combo.
        """
        config_str = f"{self.name}:{','.join(sorted(self.columns))}:{self.where or ''}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


# ---------------------------------------------------------------------------
# Default products shipped with the package
# ---------------------------------------------------------------------------

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
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "radial_velocity",
    "radial_velocity_error",
]

DEFAULT_PRODUCTS: dict[str, "GaiaProduct"] = {
    "source": GaiaProduct(
        name="source",
        url="https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/",
        md5sum_file="_MD5SUM.txt",
        file_prefix="GaiaSource_",
        file_ext=".csv.gz",
        columns=COLUMNS_OF_INTEREST,
    ),
    "bright_sources": GaiaProduct(
        name="bright_sources",
        url="https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/",
        md5sum_file="_MD5SUM.txt",
        file_prefix="GaiaSource_",
        file_ext=".csv.gz",
        columns=COLUMNS_OF_INTEREST,
        where="phot_g_mean_mag < 16",
    ),
}


# ---------------------------------------------------------------------------
# Product registry
# ---------------------------------------------------------------------------

_product_registry: dict[str, "GaiaProduct"] = {}


def _get_config_dir() -> Path:
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA")) / "gaiahealpixcache"
    elif os.name == "posix":
        if "XDG_CONFIG_HOME" in os.environ:
            base = Path(os.environ["XDG_CONFIG_HOME"]) / "gaiahealpixcache"
        else:
            base = Path.home() / ".config" / "gaiahealpixcache"
    else:
        raise OSError("Unsupported operating system")
    return base


def _load_user_products() -> dict[str, "GaiaProduct"]:
    """Load custom products from user config directory."""
    products_dir = _get_config_dir() / "products"
    if not products_dir.exists():
        return {}

    result: dict[str, "GaiaProduct"] = {}
    for fn in sorted(products_dir.glob("*.json")):
        try:
            with open(fn) as fid:
                data = json.load(fid)
            prod = GaiaProduct.from_dict(data)
            result[prod.name] = prod
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: failed to load product from {fn}: {e}")
    return result


def _init_registry():
    """Initialize the product registry with defaults + user products."""
    global _product_registry
    if not _product_registry:
        _product_registry = dict(DEFAULT_PRODUCTS)
        _product_registry.update(_load_user_products())


def get_product(name: str) -> "GaiaProduct":
    """Look up a product by name.

    Parameters
    ----------
    name : str
        Product name (e.g., "source").

    Returns
    -------
    GaiaProduct
        The product configuration.

    Raises
    ------
    KeyError
        If the product is not found.
    """
    _init_registry()
    if name not in _product_registry:
        raise KeyError(
            f"Product '{name}' not found. "
            f"Available: {', '.join(sorted(_product_registry))}"
        )
    return _product_registry[name]


def register_product(product: "GaiaProduct"):
    """Register a product in the in-memory registry and persist to disk.

    Parameters
    ----------
    product : GaiaProduct
        The product to register.
    """
    _init_registry()
    _product_registry[product.name] = product

    products_dir = _get_config_dir() / "products"
    products_dir.mkdir(parents=True, exist_ok=True)
    fn = products_dir / f"{product.name}.json"
    with open(fn, "w") as fid:
        json.dump(product.to_dict(), fid, indent=2)


def unregister_product(name: str):
    """Remove a custom product from the registry, disk, and cache.

    Parameters
    ----------
    name : str
        Product name to remove.

    Raises
    ------
    KeyError
        If the product is not found or is a default product.
    """
    _init_registry()
    if name in DEFAULT_PRODUCTS:
        raise KeyError(f"Cannot unregister default product '{name}'")
    if name not in _product_registry:
        raise KeyError(f"Product '{name}' not found")

    product = _product_registry.pop(name)

    fn = _get_config_dir() / "products" / f"{name}.json"
    if fn.exists():
        fn.unlink()

    _cleanup_cache(product)


def list_products() -> list[str]:
    """Return sorted list of available product names."""
    _init_registry()
    return sorted(_product_registry.keys())


def _cleanup_cache(product: "GaiaProduct"):
    """Remove cached .npy files matching the product's config hash."""
    from .cache import get_cache_dir

    cfg_hash = product.config_hash()
    cache_dir = get_cache_dir()
    if not cache_dir or not os.path.isdir(cache_dir):
        return

    for fn in os.listdir(cache_dir):
        if fn.endswith(".npy") and f"_{cfg_hash}.npy" in fn:
            os.remove(os.path.join(cache_dir, fn))
