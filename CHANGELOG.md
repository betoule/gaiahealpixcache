# Changelog

All notable changes to this project will be documented in this file.

## [0.3.1] — 2025-05-31

### Fixed
- `source_id` precision loss for large values — converted bytes directly with `int(v)` instead of `int(float(v))` which loses precision for 19-digit Gaia source IDs

## [0.3.0] — 2025-05-28

### Added
- **Spectroscopy support**: `query_spectra()`, `spectro_wavelengths()`, `match_catalogs()` — query Gaia sampled mean spectra (343 flux points, 336–1020 nm)
- **Rectangular region queries**: `query_rectangular()`, `query_spectra_rectangular()` — query by RA/DEC bounds with RA wrapping support
- **Configurable cache directory**: `GAIAXCACHE` env var to override default cache location
- Added module and class docstrings

### Changed
- `source_id` is now an integer (was string)
- Improved airmass computation

### Fixed
- README: corrected wavelength range for sampled spectra (336–1020 nm)

## [0.2.0] — 2025-05-27

### Added
- Product system with configurable column selection, filters, and custom products
- Coordinate transforms: `gaia_to_topocentric()`, `compute_airmass()`
- `list_products()`, `register_product()`, `unregister_product()` API

## [0.1.0] — Initial release

- Basic Gaia catalog download and caching
- Circular cone search queries
