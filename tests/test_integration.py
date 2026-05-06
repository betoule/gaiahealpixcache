import pytest

import gaiahealpixcache


@pytest.mark.slow
def test_query_g191b2b():
    sources = gaiahealpixcache.query(76.37757540733, 52.83108869489)
    assert len(sources) > 0
    assert "source_id" in sources.dtype.names
    assert "ra" in sources.dtype.names
    assert "dec" in sources.dtype.names


@pytest.mark.slow
def test_query_small_radius():
    sources = gaiahealpixcache.query(76.377, 52.831, radius_arcmin=1)
    assert len(sources) > 0


@pytest.mark.slow
def test_retrieve_gaia_data():
    md5sum_path = gaiahealpixcache.gaia._get_md5sum_path()
    bins, ranges = gaiahealpixcache.parse_md5sum(md5sum_path)
    first_range = ranges[0]
    data = gaiahealpixcache.retrieve_gaia_data(first_range)
    assert len(data) > 0
    assert "source_id" in data.dtype.names


@pytest.mark.slow
def test_gaia_to_topocentric():
    sources = gaiahealpixcache.query(76.377, 52.831, radius_arcmin=5)
    topo = gaiahealpixcache.gaia_to_topocentric(sources)
    assert len(topo) == len(sources)
    assert "ra_apparent_deg" in topo.dtype.names
    assert "alt_deg" in topo.dtype.names
    assert "airmass" in topo.dtype.names


@pytest.mark.slow
def test_center_at_date():
    from astropy.time import Time
    mjd = Time.now().mjd
    ra_app, dec_app = gaiahealpixcache.center_at_date(76.377, 52.831, mjd)
    assert isinstance(ra_app, float)
    assert isinstance(dec_app, float)
