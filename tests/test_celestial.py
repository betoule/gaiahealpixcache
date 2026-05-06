import numpy as np
import pytest

from gaiahealpixcache.celestial import (
    center_at_date,
    conform_coordinates,
    gaia_to_topocentric,
)


def _make_catalog(n=3):
    """Create a minimal catalog for testing."""
    from gaiahealpixcache.gaia import COLUMNS_OF_INTEREST

    data = {k: np.zeros(n) for k in COLUMNS_OF_INTEREST}
    data["ra"] = np.array([76.377, 76.4, 76.5])
    data["dec"] = np.array([52.831, 52.85, 52.9])
    data["parallax"] = np.array([10.0, 5.0, 0.0])
    data["pmra"] = np.array([1.0, -1.0, 0.5])
    data["pmdec"] = np.array([2.0, 0.5, -0.5])
    data["radial_velocity"] = np.array([100.0, -50.0, 0.0])
    return np.rec.fromarrays(
        [data[k] for k in COLUMNS_OF_INTEREST], names=COLUMNS_OF_INTEREST
    )


def test_gaia_to_topocentric_basic():
    catalog = _make_catalog()
    result = gaia_to_topocentric(catalog)
    assert len(result) == len(catalog)
    assert "ra_apparent_deg" in result.dtype.names
    assert "dec_apparent_deg" in result.dtype.names
    assert "alt_deg" in result.dtype.names
    assert "az_deg" in result.dtype.names
    assert "airmass" in result.dtype.names


def test_gaia_to_topocentric_with_mjd():
    catalog = _make_catalog()
    result = gaia_to_topocentric(catalog, mjd=60000.0)
    assert len(result) == len(catalog)
    assert "ra_apparent_deg" in result.dtype.names


def test_gaia_to_topocentric_negative_parallax():
    catalog = _make_catalog()
    catalog["parallax"] = np.array([-5.0, 0.0, 10.0])
    result = gaia_to_topocentric(catalog)
    assert len(result) == len(catalog)


def test_gaia_to_topocentric_custom_obs():
    catalog = _make_catalog()
    result = gaia_to_topocentric(
        catalog,
        lon_deg=-70.0,
        lat_deg=-23.0,
        height_m=2000.0,
    )
    assert len(result) == len(catalog)


def test_center_at_date():
    ra_app, dec_app = center_at_date(76.377, 52.831, 60000.0)
    assert isinstance(ra_app, float)
    assert isinstance(dec_app, float)
    assert -180 <= ra_app <= 540
    assert -90 <= dec_app <= 90


def test_center_at_date_different_epoch():
    ra_app, dec_app = center_at_date(76.377, 52.831, 60000.0, refepoch=2000.0)
    assert isinstance(ra_app, float)
    assert isinstance(dec_app, float)


def test_conform_coordinates_normal():
    ra, dec = conform_coordinates(100.0, 45.0)
    assert ra == pytest.approx(100.0)
    assert dec == pytest.approx(45.0)


def test_conform_coordinates_negative_ra():
    ra, dec = conform_coordinates(-10.0, 45.0)
    assert ra == pytest.approx(350.0)
    assert dec == pytest.approx(45.0)


def test_conform_coordinates_dec_over_90():
    ra, dec = conform_coordinates(100.0, 95.0)
    assert ra == pytest.approx(280.0)
    assert dec == pytest.approx(85.0)


def test_conform_coordinates_dec_under_minus90():
    ra, dec = conform_coordinates(100.0, -95.0)
    assert ra == pytest.approx(280.0)
    assert dec == pytest.approx(-85.0)


def test_conform_coordinates_array():
    ras = np.array([-10.0, 100.0])
    decs = np.array([45.0, 95.0])
    ra, dec = conform_coordinates(ras, decs)
    assert ra[0] == pytest.approx(350.0)
    # Array path doesn't apply the scalar >90/<-90 correction
