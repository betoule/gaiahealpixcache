# gaiahealpixcache

Download and cache a lightweight version of the Gaia DR3 catalog for offline work.

This tool follows the official HEALPix level 8 partitioning of the Gaia archive, enabling on-demand partial download with a pure NumPy backend. Much faster than querying the online archive for large amounts of data.

## Installation

```bash
uv pip install gaiahealpixcache
```

Or from source:

```bash
git clone https://github.com/betoule/gaiahealpixcache.git
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
print(sources["phot_g_mean_flux"][:5])
```

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

### Manage cache

```python
cache_dir = gaiahealpixcache.get_cache_dir()
print(f"Cache location: {cache_dir}")

gaiahealpixcache.clear_cache()
```

## API

| Function | Description |
|---|---|
| `query(ra_deg, dec_deg, radius_arcmin)` | Query Gaia sources within a circular region |
| `gaia_to_topocentric(catalog, mjd, ...)` | Convert ICRS catalog to topocentric coordinates |
| `center_at_date(ra, dec, mjd)` | Get apparent RA/Dec at a given date |
| `get_pixlist(ras, decs, level)` | Get HEALPix pixels for coordinates |
| `get_pix_range(ra, dec)` | Get Gaia file pixel ranges for coordinates |
| `retrieve_gaia_data(pixel_range)` | Download/cache a single Gaia tile |
| `haversine(ra1, dec1, ra2, dec2)` | Great-circle distance in degrees |
| `get_cache_dir()` | Get cache directory path |
| `clear_cache()` | Remove all cached data |

## License

MIT
