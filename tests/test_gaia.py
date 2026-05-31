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
    match_catalogs,
    parse_md5sum,
    query,
    query_rectangular,
    query_spectra,
    query_spectra_rectangular,
    read_gaia,
    read_gaia_spectra,
    retrieve_gaia_data,
    spectro_wavelengths,
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
    data_values = (
        b",".join(b"10" if c == "source_id" else b"1.0" for c in COLUMNS_OF_INTEREST)
        + b"\n"
    )
    null_values = (
        b",".join(b"0" if c == "source_id" else b"null" for c in COLUMNS_OF_INTEREST)
        + b"\n"
    )
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
        assert result["source_id"][0] == 10
        assert result["ra"][0] == 1.0
        assert np.isnan(result["parallax"][1])
    finally:
        os.unlink(tmpfile)


def test_read_gaia_where_filter():
    from gaiahealpixcache.products import COLUMNS_OF_INTEREST

    header = b",".join(c.encode() for c in COLUMNS_OF_INTEREST) + b"\n"

    def get_v(i, v):
        if i == COLUMNS_OF_INTEREST.index("phot_g_mean_mag"):
            return v
        if i == COLUMNS_OF_INTEREST.index("source_id"):
            return b"1111"
        return b"1.0"

    bright = (
        b",".join(get_v(i, b"10.0") for i, c in enumerate(COLUMNS_OF_INTEREST)) + b"\n"
    )
    faint = (
        b",".join(get_v(i, b"18.0") for i, c in enumerate(COLUMNS_OF_INTEREST)) + b"\n"
    )
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv.gz", delete=False) as f:
        gz = gzip.GzipFile(fileobj=f, mode="wb")
        gz.write(header)
        gz.write(bright)
        gz.write(faint)
        gz.close()
        tmpfile = f.name

    try:
        result = read_gaia(tmpfile, where="phot_g_mean_mag < 16")
        assert len(result) == 1
        assert result["phot_g_mean_mag"][0] == 10.0
    finally:
        os.unlink(tmpfile)


def test_read_gaia_where_complex():
    from gaiahealpixcache.products import COLUMNS_OF_INTEREST

    header = b",".join(c.encode() for c in COLUMNS_OF_INTEREST) + b"\n"

    def get_v(i, v1, v2):
        if i == COLUMNS_OF_INTEREST.index("phot_g_mean_mag"):
            return v1
        if i == COLUMNS_OF_INTEREST.index("source_id"):
            return b"1111"
        return v2

    row1 = (
        b",".join(get_v(i, b"10.0", b"5.0") for i, c in enumerate(COLUMNS_OF_INTEREST))
        + b"\n"
    )
    row2 = (
        b",".join(get_v(i, b"18.0", b"30.0") for i, c in enumerate(COLUMNS_OF_INTEREST))
        + b"\n"
    )
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv.gz", delete=False) as f:
        gz = gzip.GzipFile(fileobj=f, mode="wb")
        gz.write(header)
        gz.write(row1)
        gz.write(row2)
        gz.close()
        tmpfile = f.name

    try:
        result = read_gaia(tmpfile, where="(phot_g_mean_mag < 16) & (parallax > 0)")
        assert len(result) == 1
        assert result["phot_g_mean_mag"][0] == 10.0
    finally:
        os.unlink(tmpfile)


def test_validate_where_safe():
    from gaiahealpixcache.gaia import _validate_where

    _validate_where("phot_g_mean_mag < 16")
    _validate_where("(phot_g_mean_mag < 16) & (parallax > 0)")
    _validate_where("np.isnan(phot_g_mean_mag)")
    _validate_where("(ra > 0) or (dec < 45)")


def test_validate_where_dangerous():
    from gaiahealpixcache.gaia import _validate_where

    with pytest.raises(ValueError):
        _validate_where("__import__('os')")
    with pytest.raises(ValueError):
        _validate_where("open('/etc/passwd').read()")
    with pytest.raises(ValueError):
        _validate_where("exec('print(1)')")
    with pytest.raises(ValueError):
        _validate_where("os.system('ls')")


def test_safe_eval_where():
    import numpy as np
    from gaiahealpixcache.gaia import _safe_eval_where

    data = np.rec.fromarrays(
        [np.array([10.0, 18.0, 14.0]), np.array([5.0, -1.0, 3.0])],
        names=["phot_g_mean_mag", "parallax"],
    )
    mask = _safe_eval_where("phot_g_mean_mag < 16", data)
    assert mask.sum() == 2

    mask = _safe_eval_where("(phot_g_mean_mag < 16) & (parallax > 0)", data)
    assert mask.sum() == 2


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


def test_read_gaia_spectra():
    meta_cols = ["source_id", "ra", "dec"]
    header = b",".join(
        [b"source_id", b"ref_epoch", b"ra", b"dec"]
        + [b"flux_" + str(i).encode() for i in range(343)]
    )
    header += b"\n"
    flux_vals = b",".join(b"1.0" for _ in range(343))
    data_line = (
        b",".join([b"12345", b"2016.0", b"76.377", b"52.831", flux_vals]) + b"\n"
    )
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv.gz", delete=False) as f:
        gz = gzip.GzipFile(fileobj=f, mode="wb")
        gz.write(header)
        gz.write(data_line)
        gz.close()
        tmpfile = f.name

    try:
        meta, flux = read_gaia_spectra(tmpfile)
        assert len(meta) == 1
        assert meta["source_id"][0] == 12345
        assert meta["ra"][0] == pytest.approx(76.377)
        assert flux.shape == (1, 343)
        assert flux.dtype == np.float32
    finally:
        os.unlink(tmpfile)


def test_read_gaia_spectra_multiple_sources():
    meta_cols = ["source_id", "ra", "dec"]
    header = b",".join(
        [b"source_id", b"ref_epoch", b"ra", b"dec"]
        + [b"flux_" + str(i).encode() for i in range(343)]
    )
    header += b"\n"
    flux_vals = b",".join(b"1.0" for _ in range(343))
    line1 = b",".join([b"100", b"2016.0", b"76.0", b"52.0", flux_vals]) + b"\n"
    flux_vals2 = b",".join(b"2.0" for _ in range(343))
    line2 = b",".join([b"200", b"2016.0", b"77.0", b"53.0", flux_vals2]) + b"\n"
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv.gz", delete=False) as f:
        gz = gzip.GzipFile(fileobj=f, mode="wb")
        gz.write(header)
        gz.write(line1)
        gz.write(line2)
        gz.close()
        tmpfile = f.name

    try:
        meta, flux = read_gaia_spectra(tmpfile)
        assert len(meta) == 2
        assert meta["source_id"][0] == 100
        assert meta["source_id"][1] == 200
        assert flux.shape == (2, 343)
        np.testing.assert_allclose(flux[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(flux[1], 2.0, atol=1e-6)
    finally:
        os.unlink(tmpfile)


def test_read_gaia_spectra_where_filter():
    meta_cols = ["source_id", "ra", "dec"]
    header = b",".join(
        [b"source_id", b"ref_epoch", b"ra", b"dec"]
        + [b"flux_" + str(i).encode() for i in range(343)]
    )
    header += b"\n"
    flux_vals = b",".join(b"1.0" for _ in range(343))
    line1 = b",".join([b"100", b"2016.0", b"76.0", b"52.0", flux_vals]) + b"\n"
    line2 = b",".join([b"200", b"2016.0", b"77.0", b"53.0", flux_vals]) + b"\n"
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv.gz", delete=False) as f:
        gz = gzip.GzipFile(fileobj=f, mode="wb")
        gz.write(header)
        gz.write(line1)
        gz.write(line2)
        gz.close()
        tmpfile = f.name

    try:
        meta, flux = read_gaia_spectra(tmpfile, where="ra < 76.5")
        assert len(meta) == 1
        assert meta["ra"][0] == pytest.approx(76.0)
        assert flux.shape == (1, 343)
    finally:
        os.unlink(tmpfile)


def test_read_gaia_spectra_custom_flux_range():
    header = b"source_id,ref_epoch,ra,dec,f0,f1,f2,f3,f4,f5\n"
    data_line = b"12345,2016.0,76.377,52.831,1.0,2.0,3.0,4.0,5.0,6.0\n"
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv.gz", delete=False) as f:
        gz = gzip.GzipFile(fileobj=f, mode="wb")
        gz.write(header)
        gz.write(data_line)
        gz.close()
        tmpfile = f.name

    try:
        meta, flux = read_gaia_spectra(tmpfile, flux_range=(4, 9))
        assert len(meta) == 1
        assert flux.shape == (1, 5)
        np.testing.assert_allclose(flux[0], [1.0, 2.0, 3.0, 4.0, 5.0])
    finally:
        os.unlink(tmpfile)


def test_query_spectra(mocker):
    mock_meta = np.rec.fromarrays(
        [np.array([76.377, 76.5]), np.array([52.831, 53.0])],
        names=["ra", "dec"],
    )
    mock_flux = np.array([[1.0] * 343, [2.0] * 343], dtype=np.float32)
    mock_retrieve = mocker.patch(
        "gaiahealpixcache.gaia.retrieve_gaia_data",
        return_value=(mock_meta, mock_flux),
    )
    meta, flux = query_spectra(76.377, 52.831, radius_arcmin=30)
    mock_retrieve.assert_called()
    assert len(meta) >= 0
    assert flux.shape[1] == 343


def test_query_spectra_with_product(mocker):
    mock_meta = np.rec.fromarrays(
        [np.array([76.377]), np.array([52.831])],
        names=["ra", "dec"],
    )
    mock_flux = np.array([[1.0] * 343], dtype=np.float32)
    mock_retrieve = mocker.patch(
        "gaiahealpixcache.gaia.retrieve_gaia_data",
        return_value=(mock_meta, mock_flux),
    )
    meta, flux = query_spectra(
        76.377, 52.831, radius_arcmin=30, product="sampled_spectra"
    )
    mock_retrieve.assert_called()
    assert len(meta) >= 0


def test_retrieve_gaia_data_spectro_cached(mocker):
    mock_np_load = mocker.patch(
        "numpy.load",
        return_value={
            "meta": np.rec.fromarrays(
                [np.array([1.0, 2.0])],
                names=["source_id"],
            ),
            "flux": np.array([[1.0] * 343, [2.0] * 343], dtype=np.float32),
        },
    )
    mocker.patch("os.path.exists", return_value=True)
    meta, flux = retrieve_gaia_data("0-63", product="sampled_spectra")
    mock_np_load.assert_called_once()
    assert len(meta) == 2
    assert flux.shape == (2, 343)


def test_retrieve_gaia_data_spectro_download(mocker):
    mock_cached = mocker.patch(
        "gaiahealpixcache.gaia.cached_download",
        return_value="/tmp/test_spectro.csv.gz",
    )
    mock_read = mocker.patch(
        "gaiahealpixcache.gaia.read_gaia_spectra",
        return_value=(
            np.rec.fromarrays([np.array([1.0])], names=["source_id"]),
            np.array([[1.0] * 343], dtype=np.float32),
        ),
    )
    mock_savez = mocker.patch("numpy.savez")
    mock_remove = mocker.patch("os.remove")
    mock_exists = mocker.patch("os.path.exists", return_value=False)

    meta, flux = retrieve_gaia_data("0-63", product="sampled_spectra")
    mock_cached.assert_called_once()
    mock_read.assert_called_once()
    mock_savez.assert_called_once()
    mock_remove.assert_called_once()
    assert len(meta) == 1
    assert flux.shape == (1, 343)


def test_spectro_product_config():
    from gaiahealpixcache.products import get_product

    prod = get_product("sampled_spectra")
    assert prod.spectro is True
    assert "source_id" in prod.spectro_meta_cols
    assert "ra" in prod.spectro_meta_cols
    assert "dec" in prod.spectro_meta_cols
    assert prod.spectro_flux_cols == (4, 347)


def test_spectro_wavelengths_default():
    wavelengths = spectro_wavelengths()
    assert len(wavelengths) == 343
    assert wavelengths[0] == pytest.approx(336.0)
    assert wavelengths[-1] == pytest.approx(1020.0)
    assert wavelengths.dtype == np.float64
    for i in range(len(wavelengths) - 1):
        assert wavelengths[i + 1] - wavelengths[i] == pytest.approx(2.0)


def test_spectro_wavelengths_with_product_string():
    wavelengths = spectro_wavelengths("sampled_spectra")
    assert len(wavelengths) == 343


def test_spectro_wavelengths_custom_flux_range():
    from gaiahealpixcache.products import GaiaProduct

    prod = GaiaProduct(
        name="custom_spectro",
        url="https://example.com/",
        md5sum_file="MD5SUM",
        file_prefix="Custom_",
        file_ext=".csv.gz",
        columns=["source_id"],
        spectro=True,
        spectro_meta_cols=["source_id", "ra", "dec"],
        spectro_flux_cols=(4, 9),
    )
    wavelengths = spectro_wavelengths(prod)
    assert len(wavelengths) == 5
    np.testing.assert_array_equal(wavelengths, [336.0, 338.0, 340.0, 342.0, 344.0])


def test_spectro_wavelengths_returns_copy():
    w1 = spectro_wavelengths()
    w2 = spectro_wavelengths()
    w1[0] = 999.0
    assert w2[0] == pytest.approx(336.0)


def test_match_catalogs_full_overlap():
    cat_a = np.rec.fromarrays(
        [np.array([100, 200, 300])],
        names=["source_id"],
    )
    cat_b = np.rec.fromarrays(
        [np.array([100, 200, 300])],
        names=["source_id"],
    )
    idx_a, idx_b = match_catalogs(cat_a, cat_b)
    assert len(idx_a) == 3
    assert len(idx_b) == 3
    for k in range(3):
        assert cat_a["source_id"][idx_a[k]] == cat_b["source_id"][idx_b[k]]


def test_match_catalogs_partial_overlap():
    cat_a = np.rec.fromarrays(
        [np.array([100, 200, 300, 400])],
        names=["source_id"],
    )
    cat_b = np.rec.fromarrays(
        [np.array([200, 300, 500])],
        names=["source_id"],
    )
    idx_a, idx_b = match_catalogs(cat_a, cat_b)
    assert len(idx_a) == 2
    assert len(idx_b) == 2
    for k in range(2):
        assert cat_a["source_id"][idx_a[k]] == cat_b["source_id"][idx_b[k]]


def test_match_catalogs_no_overlap():
    cat_a = np.rec.fromarrays(
        [np.array([100, 200])],
        names=["source_id"],
    )
    cat_b = np.rec.fromarrays(
        [np.array([300, 400])],
        names=["source_id"],
    )
    idx_a, idx_b = match_catalogs(cat_a, cat_b)
    assert len(idx_a) == 0
    assert len(idx_b) == 0


def test_match_catalogs_empty_catalog_a():
    cat_a = np.rec.fromarrays(
        [np.array([], dtype=np.int64)],
        names=["source_id"],
    )
    cat_b = np.rec.fromarrays(
        [np.array([100, 200])],
        names=["source_id"],
    )
    idx_a, idx_b = match_catalogs(cat_a, cat_b)
    assert len(idx_a) == 0
    assert len(idx_b) == 0


def test_match_catalogs_empty_catalog_b():
    cat_a = np.rec.fromarrays(
        [np.array([100, 200])],
        names=["source_id"],
    )
    cat_b = np.rec.fromarrays(
        [np.array([], dtype=np.int64)],
        names=["source_id"],
    )
    idx_a, idx_b = match_catalogs(cat_a, cat_b)
    assert len(idx_a) == 0
    assert len(idx_b) == 0


def test_match_catalogs_different_order():
    cat_a = np.rec.fromarrays(
        [np.array([100, 200, 300])],
        names=["source_id"],
    )
    cat_b = np.rec.fromarrays(
        [np.array([300, 100, 200])],
        names=["source_id"],
    )
    idx_a, idx_b = match_catalogs(cat_a, cat_b)
    assert len(idx_a) == 3
    assert cat_a["source_id"][idx_a[0]] == cat_b["source_id"][idx_b[0]]
    assert cat_a["source_id"][idx_a[1]] == cat_b["source_id"][idx_b[1]]
    assert cat_a["source_id"][idx_a[2]] == cat_b["source_id"][idx_b[2]]


def test_match_catalogs_returns_arrays():
    cat_a = np.rec.fromarrays(
        [np.array([100])],
        names=["source_id"],
    )
    cat_b = np.rec.fromarrays(
        [np.array([100])],
        names=["source_id"],
    )
    idx_a, idx_b = match_catalogs(cat_a, cat_b)
    assert isinstance(idx_a, np.ndarray)
    assert isinstance(idx_b, np.ndarray)


def test_match_catalogs_larger_a_than_b():
    cat_a = np.rec.fromarrays(
        [np.array([100, 200, 300, 400, 500])],
        names=["source_id"],
    )
    cat_b = np.rec.fromarrays(
        [np.array([300, 400])],
        names=["source_id"],
    )
    idx_a, idx_b = match_catalogs(cat_a, cat_b)
    assert len(idx_a) == 2
    assert len(idx_b) == 2
    for k in range(2):
        assert cat_a["source_id"][idx_a[k]] == cat_b["source_id"][idx_b[k]]


def test_query_rectangular(mocker):
    mock_retrieve = mocker.patch(
        "gaiahealpixcache.gaia.retrieve_gaia_data",
        return_value=np.rec.fromarrays(
            [np.array([76.0, 77.0, 78.0]), np.array([52.0, 53.0, 54.0])],
            names=["ra", "dec"],
        ),
    )
    result = query_rectangular(76.0, 77.5, 52.5, 53.5)
    assert len(result) == 1
    assert result["ra"][0] == 77.0
    assert result["dec"][0] == 53.0
    mock_retrieve.assert_called()


def test_query_rectangular_empty(mocker):
    mock_retrieve = mocker.patch(
        "gaiahealpixcache.gaia.retrieve_gaia_data",
        return_value=np.rec.fromarrays(
            [np.array([76.0, 77.0]), np.array([52.0, 53.0])],
            names=["ra", "dec"],
        ),
    )
    result = query_rectangular(80.0, 81.0, 55.0, 56.0)
    assert len(result) == 0


def test_query_rectangular_ra_wrap(mocker):
    mocker.patch(
        "gaiahealpixcache.gaia.get_pix_range",
        return_value=["0-63"],
    )
    mock_retrieve = mocker.patch(
        "gaiahealpixcache.gaia.retrieve_gaia_data",
        return_value=np.rec.fromarrays(
            [np.array([359.0, 0.5, 10.0]), np.array([10.0, 11.0, 12.0])],
            names=["ra", "dec"],
        ),
    )
    result = query_rectangular(358.0, 2.0, 9.0, 12.0)
    assert len(result) == 2
    assert 359.0 in result["ra"]
    assert 0.5 in result["ra"]


def test_query_rectangular_with_product(mocker):
    mock_retrieve = mocker.patch(
        "gaiahealpixcache.gaia.retrieve_gaia_data",
        return_value=np.rec.fromarrays(
            [np.array([76.0, 77.0]), np.array([52.0, 53.0])],
            names=["ra", "dec"],
        ),
    )
    result = query_rectangular(76.0, 77.5, 52.5, 53.5, product="source")
    assert len(result) >= 0
    mock_retrieve.assert_called()


def test_query_spectra_rectangular(mocker):
    mock_retrieve = mocker.patch(
        "gaiahealpixcache.gaia.retrieve_gaia_data",
        return_value=(
            np.rec.fromarrays(
                [np.array([76.0, 77.0, 78.0]), np.array([52.0, 53.0, 54.0])],
                names=["ra", "dec"],
            ),
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        ),
    )
    meta, flux = query_spectra_rectangular(76.0, 77.5, 52.5, 53.5)
    assert len(meta) == 1
    assert flux.shape == (1, 2)
    mock_retrieve.assert_called()


def test_query_spectra_rectangular_empty(mocker):
    mock_retrieve = mocker.patch(
        "gaiahealpixcache.gaia.retrieve_gaia_data",
        return_value=(
            np.rec.fromarrays(
                [np.array([76.0]), np.array([52.0])],
                names=["ra", "dec"],
            ),
            np.array([[1.0, 2.0]], dtype=np.float32),
        ),
    )
    meta, flux = query_spectra_rectangular(80.0, 81.0, 55.0, 56.0)
    assert len(meta) == 0
    assert flux.shape == (0, 2)


def test_query_spectra_rectangular_ra_wrap(mocker):
    mocker.patch(
        "gaiahealpixcache.gaia.get_pix_range",
        return_value=["0-63"],
    )
    mock_retrieve = mocker.patch(
        "gaiahealpixcache.gaia.retrieve_gaia_data",
        return_value=(
            np.rec.fromarrays(
                [np.array([359.0, 0.5, 10.0]), np.array([10.0, 11.0, 12.0])],
                names=["ra", "dec"],
            ),
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        ),
    )
    meta, flux = query_spectra_rectangular(358.0, 2.0, 9.0, 12.0)
    assert len(meta) == 2
    assert flux.shape == (2, 2)
