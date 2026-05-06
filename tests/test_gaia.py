import os
import gzip
import tempfile

import numpy as np
import pytest

from gaiahealpixcache.gaia import (
    COLUMNS_OF_INTEREST,
    get_pix_range,
    get_pixlist,
    haversine,
    parse_md5sum,
    query,
    read_gaia,
    retrieve_gaia_data,
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


def test_parse_md5sum_custom_prefix():
    lines = [
        "abc123  XPContMeanSpec_0-63.ecsv.gz\n",
        "def456  XPContMeanSpec_64-127.ecsv.gz\n",
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.writelines(lines)
        tmpfile = f.name

    try:
        bins, ranges = parse_md5sum(
            tmpfile, file_prefix="XPContMeanSpec_", file_ext=".ecsv.gz"
        )
        assert bins == [0, 64]
        assert ranges == ["0-63", "64-127"]
    finally:
        os.unlink(tmpfile)


def test_read_gaia():
    from gaiahealpixcache.products import COLUMNS_OF_INTEREST

    comment = b"# %ECSV 1.0\n"
    header = b",".join(c.encode() for c in COLUMNS_OF_INTEREST) + b"\n"
    data_values = b",".join(b"1.0" for _ in COLUMNS_OF_INTEREST) + b"\n"
    null_values = b",".join(b"null" for _ in COLUMNS_OF_INTEREST) + b"\n"
    data_lines = [data_values, null_values]
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv.gz", delete=False) as f:
        gz = gzip.GzipFile(fileobj=f, mode="wb")
        gz.write(comment)
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
    assert len(COLUMNS_OF_INTEREST) == 16


def test_get_pix_range():
    ranges = get_pix_range([76.377, 76.5], [52.831, 52.9])
    assert isinstance(ranges, list)
    assert len(ranges) >= 1


def test_haversine_negative_coords():
    dist = haversine(-180.0, -90.0, 180.0, 90.0)
    assert dist == pytest.approx(180.0, abs=1e-10)


def test_haversine_small_distance():
    dist = haversine(0.0, 0.0, 0.01, 0.01)
    assert 0 < dist < 1


def test_get_pixlist_level():
    ras = [0.0]
    decs = [0.0]
    pixlist_l8 = get_pixlist(ras, decs, level=8)
    pixlist_l9 = get_pixlist(ras, decs, level=9)
    assert len(pixlist_l9) >= len(pixlist_l8)


def test_query(mocker):
    mock_retrieve = mocker.patch(
        "gaiahealpixcache.gaia.retrieve_gaia_data",
        return_value=np.rec.fromarrays(
            [np.array([76.377, 76.5]), np.array([52.831, 53.0])],
            names=["ra", "dec"],
        ),
    )
    result = query(76.377, 52.831, radius_arcmin=30)
    assert len(result) >= 0
    mock_retrieve.assert_called()


def test_query_with_product(mocker):
    mock_retrieve = mocker.patch(
        "gaiahealpixcache.gaia.retrieve_gaia_data",
        return_value=np.rec.fromarrays(
            [np.array([76.377, 76.5]), np.array([52.831, 53.0])],
            names=["ra", "dec"],
        ),
    )
    result = query(76.377, 52.831, radius_arcmin=30, product="source")
    assert len(result) >= 0
    mock_retrieve.assert_called()


def test_retrieve_gaia_data_cached(mocker):
    mock_np_load = mocker.patch(
        "numpy.load",
        return_value=np.rec.fromarrays(
            [np.array([1.0, 2.0])],
            names=["source_id"],
        ),
    )
    mocker.patch("os.path.exists", return_value=True)
    result = retrieve_gaia_data("0-63")
    mock_np_load.assert_called_once()
    assert len(result) == 2


def test_retrieve_gaia_data_download(mocker):
    mock_cached = mocker.patch(
        "gaiahealpixcache.gaia.cached_download",
        return_value="/tmp/test.csv.gz",
    )
    mock_read = mocker.patch(
        "gaiahealpixcache.gaia.read_gaia",
        return_value=np.rec.fromarrays(
            [np.array([1.0])],
            names=["source_id"],
        ),
    )
    mock_save = mocker.patch("numpy.save")
    mock_remove = mocker.patch("os.remove")
    mock_exists = mocker.patch("os.path.exists", return_value=False)

    result = retrieve_gaia_data("0-63")
    mock_cached.assert_called_once()
    mock_read.assert_called_once()
    mock_save.assert_called_once()
    mock_remove.assert_called_once()
    assert len(result) == 1


def test_cache_lock_context_manager(tmp_path):
    from gaiahealpixcache.gaia import _CacheLock

    lock_path = str(tmp_path / "test.lock")
    lock = _CacheLock(lock_path)
    with lock:
        assert os.path.exists(lock_path)
    assert not os.path.exists(lock_path)


def test_cache_lock_double_acquire_timeout(tmp_path):
    from gaiahealpixcache.gaia import _CacheLock
    import threading

    lock_path = str(tmp_path / "test.lock")
    lock1 = _CacheLock(lock_path, timeout=5.0)
    lock2 = _CacheLock(lock_path, timeout=0.3)

    lock1.acquire()
    try:
        with pytest.raises(TimeoutError):
            lock2.acquire()
    finally:
        lock1.release()


def test_cache_lock_concurrent_access(tmp_path):
    from gaiahealpixcache.gaia import _CacheLock
    import threading, time

    lock_path = str(tmp_path / "test.lock")
    max_active: list[int] = [0]
    active_count: list[int] = [0]
    count_lock = threading.Lock()

    def worker():
        with _CacheLock(lock_path, timeout=5.0):
            with count_lock:
                active_count[0] += 1
                if active_count[0] > max_active[0]:
                    max_active[0] = active_count[0]
            time.sleep(0.05)
            with count_lock:
                active_count[0] -= 1

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_active[0] == 1
