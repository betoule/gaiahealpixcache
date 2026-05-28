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

## Quick Start

```python
import gaiahealpixcache

# Query sources around a sky position
sources = gaiahealpixcache.query(ra_deg=76.377, dec_deg=52.831, radius_arcmin=30)
print(f"{len(sources)} sources found")
print(sources["source_id"][:5])
print(sources["phot_g_mean_mag"][:5])
```

Tiles are downloaded on first access and cached as compressed NumPy arrays for fast subsequent queries.

### Rectangular Region Query

For rectangular sky regions, use `query_rectangular` with `ra_min`, `ra_max`, `dec_min`, `dec_max`:

```python
sources = gaiahealpixcache.query_rectangular(
    ra_min=76.0, ra_max=78.0,
    dec_min=52.0, dec_max=54.0,
)
```

RA wrapping across the 0/360 boundary is handled automatically (e.g., `ra_min=358, ra_max=2`).

The spectroscopy equivalent is `query_spectra_rectangular`:

```python
meta, flux = gaiahealpixcache.query_spectra_rectangular(
    ra_min=76.0, ra_max=78.0,
    dec_min=52.0, dec_max=54.0,
)
```

## Products

A *product* defines which columns are loaded from the Gaia archive and which rows are kept. Multiple products can coexist, each with its own cache namespace.

### Default Products

| Product | Description | Filter |
|---|---|---|
| `source` | Full Gaia source catalog (selected columns) | none |
| `bright_sources` | Sources with G < 16 | `phot_g_mean_mag < 16` |
| `sampled_spectra` | Sampled mean spectra (343 flux points) | none |

```python
# Use the bright_sources product to save disk space
sources = gaiahealpixcache.query(76.377, 52.831, product="bright_sources")
```

The default column set is:
`source_id`, `ra`, `ra_error`, `dec`, `dec_error`, `parallax`, `parallax_error`,
`pmra`, `pmra_error`, `pmdec`, `pmdec_error`, `phot_g_mean_mag`, `phot_bp_mean_mag`,
`phot_rp_mean_mag`, `radial_velocity`, `radial_velocity_error`.

### Custom Products

Define a product with a custom column selection and optional row filter:

```python
from gaiahealpixcache import GaiaProduct, register_product, query

my_product = GaiaProduct(
    name="astrometry_lite",
    url="https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/",
    md5sum_file="_MD5SUM.txt",
    file_prefix="GaiaSource_",
    file_ext=".csv.gz",
    columns=["source_id", "ra", "dec", "parallax", "pmra", "pmdec"],
    where="(parallax > 0) & (phot_g_mean_mag < 18)",
)
register_product(my_product)

sources = query(76.377, 52.831, product="astrometry_lite")
```

Products are persisted to `~/.config/gaiahealpixcache/products/` and survive sessions.

```python
# List all available products
print(gaiahealpixcache.list_products())

# Remove a custom product (also cleans its cached data)
gaiahealpixcache.unregister_product("astrometry_lite")
```

### Filter Expressions (`where`)

The `where` field accepts a Python boolean expression evaluated over the loaded data. Column names refer to Gaia column names:

```python
# Good proper-motion candidates
where="abs(pmra) > 10 and abs(pmdec) > 10"

# Bright sources with parallax
where="(phot_g_mean_mag < 14) & (parallax > 0)"

# Use numpy functions
where="np.isfinite(phot_g_mean_mag) & np.isfinite(parallax)"
```

Filter expressions are validated by AST analysis and evaluated in a restricted namespace (column arrays + `np` only). No filesystem access or arbitrary imports are possible.

## Spectroscopy

The `sampled_spectra` product provides Gaia sampled mean spectra — 343 flux points per source spanning 330–1050 nm.

```python
import gaiahealpixcache

# Query spectra around a sky position
meta, flux = gaiahealpixcache.query_spectra(
    ra_deg=76.377,
    dec_deg=52.831,
    radius_arcmin=30,
)
print(f"{len(meta)} sources with spectra")
print(meta["source_id"][:5])
print(flux.shape)  # (num_sources, 343)
```

`query_spectra` returns a tuple of `(meta, flux)`:
- `meta`: structured array with `source_id`, `ra`, `dec`
- `flux`: 2D float array with shape `(num_sources, 343)`

Spectro tiles are cached separately as `.npz` files. The same `where` filter and product system applies:

```python
# Custom spectro product with narrower flux range
from gaiahealpixcache import GaiaProduct, register_product, query_spectra

spec_product = GaiaProduct(
    name="spectra_uv_only",
    url="https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_spectro/",
    md5sum_file="_MD5SUM.txt",
    file_prefix="GaiaSpectro_",
    file_ext="_sampledSpectrum.csv",
    columns=["source_id", "ref_epoch", "ra", "dec"],
    spectro=True,
    spectro_meta_cols=["source_id", "ra", "dec"],
    spectro_flux_cols=(4, 100),  # first 96 flux points only
    where="phot_g_mean_mag < 15",
)
register_product(spec_product)

meta, flux = query_spectra(76.377, 52.831, product="spectra_uv_only")
```

### Wavelengths

Retrieve the wavelength array corresponding to the flux points:

```python
import gaiahealpixcache

wavelengths = gaiahealpixcache.spectro_wavelengths()
print(wavelengths)  # [336., 338., 340., ..., 1018., 1020.]
```

Wavelengths are 343 values from 336 to 1020 nm with a step of 2 nm. For custom products with a narrower `spectro_flux_cols` range, only the corresponding wavelengths are returned.

### Catalog Matching

Match two catalogs by `source_id` using an efficient inner join:

```python
import gaiahealpixcache

# Query photometry and spectra separately
sources = gaiahealpixcache.query(76.377, 52.831)
meta, flux = gaiahealpixcache.query_spectra(76.377, 52.831)

# Find common sources
idx_a, idx_b = gaiahealpixcache.match_catalogs(sources, meta)
print(f"{len(idx_a)} sources have both photometry and spectra")
print(sources["source_id"][idx_a][:5])
print(flux[idx_b].shape)
```

`match_catalogs` returns index arrays so that `cat_a[idx_a][k]` and `cat_b[idx_b][k]` correspond to the same source. Uses a hash-based algorithm with O(n+m) time complexity.

## Coordinate Transforms

### Topocentric Conversion

Convert ICRS catalog coordinates to apparent topocentric positions:

```python
import gaiahealpixcache
from astropy.time import Time

now = Time.now()

sources = gaiahealpixcache.query(ra_deg=76.377, dec_deg=52.831)

topo = gaiahealpixcache.gaia_to_topocentric(
    sources,
    mjd=now.mjd,
    lon_deg=5.71,
    lat_deg=43.93,
    height_m=640.0,
)
print(topo["ra_apparent_deg"][:5])
print(topo["alt_deg"][:5])
```

### Coordinate Normalization

Coordinates outside the standard convention (RA in [0, 360), Dec in [-90, 90]) are
automatically normalized. You can also call the helper directly:

```python
ra, dec = gaiahealpixcache.conform_coordinates(-10.0, 95.0)
# ra=350.0, dec=85.0
```

## Cache Management

```python
cache_dir = gaiahealpixcache.get_cache_dir()
print(f"Cache location: {cache_dir}")

gaiahealpixcache.clear_cache()
```

Cached tiles are stored as `.npy` files named with the product's configuration hash, so different products and filters maintain separate cache entries.

### Custom Cache Directory

By default, the cache follows the XDG Base Directory specification (`~/.cache/gaiahealpixcache` on Linux). Override it with the `GAIAXCACHE` environment variable:

```bash
export GAIAXCACHE=/path/to/my/cache
```

This is useful when the default cache location has limited disk space or when you want to share the cache across projects.

## Concurrency

Each cached tile is protected by a file-based lock (`fcntl.flock` on Unix). When two
processes or threads request the same tile simultaneously, only one downloads it; the
other waits and loads the result once it's ready. Downloads use atomic rename — if a
download is interrupted, the partial file is discarded, preventing corrupted cache entries.

## API Reference

| Function | Description |
|---|---|
| `query(ra_deg, dec_deg, radius_arcmin, product)` | Query Gaia sources within a circular region |
| `query_rectangular(ra_min, ra_max, dec_min, dec_max, product)` | Query Gaia sources within a rectangular region |
| `query_spectra(ra_deg, dec_deg, radius_arcmin, product)` | Query Gaia spectra within a circular region |
| `query_spectra_rectangular(ra_min, ra_max, dec_min, dec_max, product)` | Query Gaia spectra within a rectangular region |
| `gaia_to_topocentric(catalog, mjd, ...)` | Convert ICRS catalog to topocentric coordinates |
| `center_at_date(ra, dec, mjd)` | Get apparent RA/Dec at a given date |
| `conform_coordinates(ra, dec)` | Normalize coordinates to standard convention |
| `get_pixlist(ras, decs, level)` | Get HEALPix pixels for coordinates |
| `get_pix_range(ra, dec, product)` | Get Gaia file pixel ranges for coordinates |
| `retrieve_gaia_data(pixel_range, product)` | Download/cache a single Gaia tile |
| `haversine(ra1, dec1, ra2, dec2)` | Great-circle distance in degrees |
| `spectro_wavelengths(product)` | Get wavelength array for spectro flux points |
| `match_catalogs(cat_a, cat_b)` | Inner join two catalogs by source_id |
| `get_cache_dir()` | Get cache directory path |
| `clear_cache()` | Remove all cached data |
| `get_product(name)` | Look up a product by name |
| `list_products()` | List all available product names |
| `register_product(product)` | Register and persist a custom product |
| `unregister_product(name)` | Remove a custom product and its cache |

## License

MIT
