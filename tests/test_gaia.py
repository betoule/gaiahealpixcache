import os
import gzip
import tempfile

import numpy as np
import pytest

from gaiahealpixcache.gaia import (
    COLUMNS_OF_INTEREST,
    get_pixlist,
    get_pix_range,
    haversine,
    parse_md5sum,
    read_gaia,
    query,
)


def test_haversine_same_point():
    dist = haversine(0.0, 0.0, 0.0, 0.0)
    assert dist == pytest.approx(0.0, abs=1e-10)


def test_haversine_poles():
    dist = haversine(0.0, 90.0, 0.0, -90.0)
    assert dist == pytest.approx(180.0, abs=1e-10)


def test_haversine_equator():
    dist = haversine(0.0, 0.0, 90.0, 0.0)
    assert dist == pytest.approx(90.0, abs=1e-10)


def test_haversine_array():
    ra1 = np.array([0.0, 0.0])
    dec1 = np.array([0.0, 90.0])
    ra2 = np.array([90.0, 0.0])
    dec2 = np.array([0.0, -90.0])
    dists = haversine(ra1, dec1, ra2, dec2)
    np.testing.assert_allclose(dists, [90.0, 180.0], atol=1e-10)


def test_get_pixlist():
    ras = [0.0, 10.0]
    decs = [0.0, 10.0]
    pixlist = get_pixlist(ras, decs)
    assert isinstance(pixlist, list)
    assert len(pixlist) >= 1
    assert all(0 <= p < 786432 for p in pixlist)


def test_get_pixlist_unique():
    ras = [0.0, 0.0]
    decs = [0.0, 0.0]
    pixlist = get_pixlist(ras, decs)
    assert len(pixlist) == 1


def test_parse_md5sum():
    lines = [
        "abc123  GaiaSource_0-63.csv.gz\n",
        "def456  GaiaSource_64-127.csv.gz\n",
        "ghi789  GaiaSource_128-191.csv.gz\n",
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.writelines(lines)
        tmpfile = f.name

    try:
        bins, ranges = parse_md5sum(tmpfile)
        assert bins == [0, 64, 128]
        assert ranges == ["0-63", "64-127", "128-191"]
    finally:
        os.unlink(tmpfile)


def test_read_gaia():
    from gaiahealpixcache.gaia import COLUMNS_OF_INTEREST
    header = b",".join(c.encode() for c in COLUMNS_OF_INTEREST) + b"\n"
    data_values = b",".join(b"1.0" for _ in COLUMNS_OF_INTEREST) + b"\n"
    null_values = b",".join(b"null" for _ in COLUMNS_OF_INTEREST) + b"\n"
    data_lines = [data_values, null_values]
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv.gz", delete=False) as f:
        gz = gzip.GzipFile(fileobj=f, mode="wb")
        gz.write(header)
        for line in data_lines:
            gz.write(line)
        gz.close()
        tmpfile = f.name

    try:
        result = read_gaia(tmpfile)
        assert len(result) == 2
        assert result["source_id"][0] == 1.0
        assert result["ra"][0] == 1.0
        assert np.isnan(result["parallax"][1])
    finally:
        os.unlink(tmpfile)


def test_columns_of_interest():
    assert "source_id" in COLUMNS_OF_INTEREST
    assert "ra" in COLUMNS_OF_INTEREST
    assert "dec" in COLUMNS_OF_INTEREST
    assert len(COLUMNS_OF_INTEREST) == 19
