import os
import tempfile
from pathlib import Path

import pytest

from gaiahealpixcache.cache import cached_download, clear_cache, get_cache_dir, join
import numpy as np


def test_get_cache_dir_string():
    result = get_cache_dir()
    assert isinstance(result, str)
    assert "gaiahealpixcache" in result


def test_get_cache_dir_path():
    result = get_cache_dir(path=True)
    assert isinstance(result, Path)
    assert "gaiahealpixcache" in str(result)


def test_cached_download_cached(mocker):
    mock_get = mocker.patch("requests.get")
    mock_response = mocker.MagicMock()
    mock_response.headers = {"content-length": "100"}
    mock_response.iter_content.return_value = [b"test data"]
    mock_response.raise_for_status = mocker.MagicMock()
    mock_get.return_value = mock_response

    url = "https://example.com/testfile.txt"
    path1 = cached_download(url)
    mock_get.assert_called_once()

    path2 = cached_download(url)
    assert path1 == path2
    assert mock_get.call_count == 1


def test_cached_download_new(mocker):
    mock_get = mocker.patch("requests.get")
    mock_response = mocker.MagicMock()
    mock_response.headers = {"content-length": "100"}
    mock_response.iter_content.return_value = [b"test data"]
    mock_response.raise_for_status = mocker.MagicMock()
    mock_get.return_value = mock_response

    url = "https://example.com/newfile.txt"
    path = cached_download(url)
    assert os.path.exists(path)
    with open(path, "rb") as f:
        assert f.read() == b"test data"


def test_cached_download_cleanup_on_failure(mocker):
    mock_get = mocker.patch("requests.get")
    mock_response = mocker.MagicMock()
    mock_response.headers = {"content-length": "200"}
    mock_response.iter_content.side_effect = OSError("network interrupted")
    mock_response.raise_for_status = mocker.MagicMock()
    mock_get.return_value = mock_response

    url = "https://example.com/failfile.txt"
    cache_dir = get_cache_dir()
    with pytest.raises(OSError):
        cached_download(url)

    tmp_files = [f for f in os.listdir(cache_dir) if f.endswith(".tmp")]
    assert len(tmp_files) == 0, "Temp file not cleaned up after failed download"


def test_cached_download_atomic_rename(mocker):
    mock_get = mocker.patch("requests.get")
    mock_response = mocker.MagicMock()
    mock_response.headers = {"content-length": "50"}
    mock_response.iter_content.return_value = [b"atomic test data"]
    mock_response.raise_for_status = mocker.MagicMock()
    mock_get.return_value = mock_response

    url = "https://example.com/atomicfile.txt"
    path = cached_download(url)

    with open(path, "rb") as f:
        assert f.read() == b"atomic test data"

    tmp_files = [f for f in os.listdir(get_cache_dir()) if f.endswith(".tmp")]
    assert len(tmp_files) == 0


def test_clear_cache(mocker):
    cache_dir = get_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    test_file = os.path.join(cache_dir, "testfile.txt")
    with open(test_file, "w") as f:
        f.write("test")
    assert os.path.exists(test_file)

    clear_cache()

    assert not os.path.exists(test_file)


def test_clear_cache_nonexistent(mocker, capsys):
    cache_dir = get_cache_dir()
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    clear_cache()
    captured = capsys.readouterr()
    assert "does not exist" in captured.out


def test_join_basic():
    a = np.rec.fromarrays(
        [np.array([1, 2, 3]), np.array([4.0, 5.0, 6.0])],
        names=["x", "y"],
    )
    b = np.rec.fromarrays(
        [np.array([7.0, 8.0, 9.0])],
        names=["z"],
    )
    result = join(a, b)
    assert list(result.dtype.names) == ["x", "y", "z"]
    assert len(result) == 3
    np.testing.assert_array_equal(result["x"], [1, 2, 3])
    np.testing.assert_array_equal(result["z"], [7.0, 8.0, 9.0])


def test_join_kwargs():
    a = np.rec.fromarrays(
        [np.array([1, 2])],
        names=["a"],
    )
    result = join(a, b=np.array([10, 20]))
    assert list(result.dtype.names) == ["a", "b"]
    np.testing.assert_array_equal(result["b"], [10, 20])
