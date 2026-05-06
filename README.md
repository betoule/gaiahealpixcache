# gaiahealpixcache

Download and cache a lightweight version of the Gaia DR3 catalog for offline work.

This tool follows the official HEALPix level 8 partitioning of the Gaia archive, enabling on-demand partial download with a pure NumPy backend. Much faster than querying the online archive for large amounts of data.

## Installation

```bash
pip install gaiahealpixcache
```

Or from source:

```bash
git clone https://github.com/betoule/gaiahealpixcache.git
cd gaiahealpixcache
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

### Query sources around sky coordinates

```python
import gaiahealpixcache

sources = gaiahealpixcache.query(ra_deg=76.377, dec_deg=52.831, radius_arcmin=30)
print(len(sources))
print(sources["source_id"][:5])
print(sources["phot_g_mean_mag"][:5])
```

By default, the `source` product returns these columns:
`source_id`, `ra`, `ra_error`, `dec`, `dec_error`, `parallax`, `parallax_error`,
`pmra`, `pmra_error`, `pmdec`, `pmdec_error`, `phot_g_mean_mag`, `phot_bp_mean_mag`,
`phot_rp_mean_mag`, `radial_velocity`, `radial_velocity_error`.

### Convert to topocentric coordinates

```python
import gaiahealpixcache
from astropy.time import Time

now = Time.now()
mjd = now.mjd

sources = gaiahealpixcache.query(ra_deg=76.377, dec_deg=52.831)

topo = gaiahealpixcache.gaia_to_topocentric(
    sources,
    mjd=mjd,
    lon_deg=5.71,
    lat_deg=43.93,
    height_m=640.0,
)
print(topo["ra_apparent_deg"][:5])
print(topo["alt_deg"][:5])
```

### Coordinate normalization

Coordinates outside the standard convention (RA in [0, 360), Dec in [-90, 90]) are
automatically normalized. You can also call the helper directly:

```python
ra, dec = gaiahealpixcache.conform_coordinates(-10.0, 95.0)
# ra=350.0, dec=85.0
```

### Custom products

Define and register a product with a custom column selection:

```python
from gaiahealpixcache import GaiaProduct, register_product, query, list_products

my_product = GaiaProduct(
    name="bright_stars",
    url="https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/",
    md5sum_file="_MD5SUM.txt",
    file_prefix="GaiaSource_",
    file_ext=".csv.gz",
    columns=["source_id", "ra", "dec", "phot_g_mean_mag", "parallax"],
)
register_product(my_product)

sources = query(76.377, 52.831, product="bright_stars")
```

Products are persisted to `~/.config/gaiahealpixcache/products/` and survive sessions.

```python
# List all available products
print(gaiahealpixcache.list_products())

# Remove a custom product (also cleans its cached data)
gaiahealpixcache.unregister_product("bright_stars")
```

### Manage cache

```python
cache_dir = gaiahealpixcache.get_cache_dir()
print(f"Cache location: {cache_dir}")

gaiahealpixcache.clear_cache()
```

## Concurrency

Each cached tile is protected by a file-based lock (`fcntl.flock` on Unix). When two
processes or threads request the same tile simultaneously, only one downloads it; the
other waits and loads the result once it's ready. This prevents redundant downloads and
corrupted cache files.

## API

| Function | Description |
|---|---|
| `query(ra_deg, dec_deg, radius_arcmin, product)` | Query Gaia sources within a circular region |
| `gaia_to_topocentric(catalog, mjd, ...)` | Convert ICRS catalog to topocentric coordinates |
| `center_at_date(ra, dec, mjd)` | Get apparent RA/Dec at a given date |
| `conform_coordinates(ra, dec)` | Normalize coordinates to standard convention |
| `get_pixlist(ras, decs, level)` | Get HEALPix pixels for coordinates |
| `get_pix_range(ra, dec, product)` | Get Gaia file pixel ranges for coordinates |
| `retrieve_gaia_data(pixel_range, product)` | Download/cache a single Gaia tile |
| `haversine(ra1, dec1, ra2, dec2)` | Great-circle distance in degrees |
| `get_cache_dir()` | Get cache directory path |
| `clear_cache()` | Remove all cached data |
| `get_product(name)` | Look up a product by name |
| `list_products()` | List all available product names |
| `register_product(product)` | Register and persist a custom product |
| `unregister_product(name)` | Remove a custom product and its cache |

## License

MIT
