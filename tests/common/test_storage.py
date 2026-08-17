"""Tests for common.storage — load_json, save_json, open_zip."""

import contextlib
import io
import json
import zipfile
from unittest.mock import MagicMock

import pytest
from idi_ftm2j_shared.storage import load_json, open_zip, save_json


class TestLoadJson:
    """Tests for load_json()."""

    def test_loads_dict_from_local_file(self, tmp_path):
        data = {"key": "value", "number": 42}
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))

        result = load_json(str(f))
        assert result == data

    def test_loads_list_from_local_file(self, tmp_path):
        data = [1, 2, 3]
        f = tmp_path / "list.json"
        f.write_text(json.dumps(data))

        result = load_json(str(f), return_type="list")
        assert result == data

    def test_returns_empty_dict_for_missing_file(self, tmp_path):
        result = load_json(str(tmp_path / "nonexistent.json"), return_type="dict")
        assert result == {}

    def test_returns_empty_list_for_missing_file(self, tmp_path):
        result = load_json(str(tmp_path / "nonexistent.json"), return_type="list")
        assert result == []

    def test_raises_on_invalid_return_type(self, tmp_path):
        # _empty_for_return_type is only reached on FileNotFoundError,
        # so test with a missing file to trigger the ValueError path
        with pytest.raises(ValueError, match="Invalid return type"):
            load_json(str(tmp_path / "nonexistent.json"), return_type="set")


class TestSaveJson:
    """Tests for save_json()."""

    def test_writes_dict_to_local_file(self, tmp_path):
        data = {"cik": "0001234567", "name": "ACME Corp"}
        f = tmp_path / "output.json"

        save_json(str(f), data)

        assert f.exists()
        assert json.loads(f.read_text()) == data

    def test_writes_list_to_local_file(self, tmp_path):
        data = [{"a": 1}, {"b": 2}]
        f = tmp_path / "list.json"

        save_json(str(f), data)

        assert json.loads(f.read_text()) == data

    def test_overwrites_existing_file(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"old": True}))

        save_json(str(f), {"new": True})

        assert json.loads(f.read_text()) == {"new": True}

    def test_roundtrip_load_after_save(self, tmp_path):
        original = {"entries": [["CIK1", "ACC1"]], "reasons": {"CIK1 ACC1": "no_form_data"}}
        f = tmp_path / "registry.json"

        save_json(str(f), original)
        loaded = load_json(str(f))

        assert loaded == original


class TestOpenZip:
    """Tests for open_zip()."""

    def test_opens_local_zip_and_yields_zipfile(self, tmp_path):
        # Create a real zip with one JSON file
        zip_path = tmp_path / "test.zip"
        payload = {"filings": {"recent": {"form": []}}}
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("CIK0000000001.json", json.dumps(payload))

        with open_zip(str(zip_path)) as zf:
            names = zf.namelist()
            assert "CIK0000000001.json" in names

    def test_can_read_file_contents_from_zip(self, tmp_path):
        zip_path = tmp_path / "test.zip"
        payload = {"hello": "world"}
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.json", json.dumps(payload))

        with open_zip(str(zip_path)) as zf, zf.open("data.json") as f:
            data = json.load(f)

        assert data == {"hello": "world"}

    def test_passes_headers_as_transport_params(self, tmp_path, mocker):
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w"):
            pass

        mock_open = mocker.patch("smart_open.open")
        mock_file = io.BytesIO()
        with zipfile.ZipFile(mock_file, "w"):
            pass
        mock_file.seek(0)
        mock_open.return_value.__enter__ = lambda s: mock_file
        mock_open.return_value.__exit__ = MagicMock(return_value=False)

        headers = {"User-Agent": "test test@test.com"}

        # Verify the headers are forwarded — we just check smart_open was called
        # with transport_params containing the headers
        # zip read may fail on the mock — we only care about the call args
        with contextlib.suppress(Exception), open_zip(str(zip_path), headers=headers):
            pass

        mock_open.assert_called_once_with(
            str(zip_path), "rb", transport_params={"headers": headers}
        )

    def test_no_transport_params_when_no_headers(self, tmp_path, mocker):
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w"):
            pass

        mock_open = mocker.patch("smart_open.open")
        mock_file = io.BytesIO()
        with zipfile.ZipFile(mock_file, "w"):
            pass
        mock_file.seek(0)
        mock_open.return_value.__enter__ = lambda s: mock_file
        mock_open.return_value.__exit__ = MagicMock(return_value=False)

        with contextlib.suppress(Exception), open_zip(str(zip_path)):
            pass

        mock_open.assert_called_once_with(str(zip_path), "rb", transport_params={})
