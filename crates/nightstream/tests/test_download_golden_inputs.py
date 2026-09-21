import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.request import Request

import download_golden_inputs as download
from restore_golden_inputs import expected_assets


class GoldenDownloadTests(unittest.TestCase):
    def expected(self, content):
        return {"name": "witness.zip", "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "repository": "owner/repository", "tag": "retained-tag", "label": "original-sources"}

    def test_restoration_and_download_share_all_retained_archives(self):
        assets = expected_assets()
        self.assertEqual({asset["label"] for asset in assets}, {
            "envelope", "children", "reference-later", "original-sources", "independent"})
        self.assertEqual(len({asset["name"] for asset in assets}), len(assets))
        self.assertTrue(all(asset["repository"] == "LFDT-Nightstream/Nightstream" for asset in assets))
        self.assertTrue(all(asset["tag"] for asset in assets))

    def test_cross_host_redirect_drops_authentication(self):
        request = Request("https://api.github.com/repos/owner/repository/releases/assets/1",
                          headers={"Authorization": "Bearer test-secret", "Accept": "application/octet-stream"})
        redirected = download.AssetRedirect().redirect_request(
            request, None, 302, "Found", {}, "https://release-assets.githubusercontent.com/asset")
        self.assertIsNone(redirected.get_header("Authorization"))
        self.assertEqual(redirected.get_header("Accept"), "application/octet-stream")
        same_host = download.AssetRedirect().redirect_request(
            request, None, 302, "Found", {}, "https://api.github.com/other")
        self.assertEqual(same_host.get_header("Authorization"), "Bearer test-secret")

    def test_plain_http_or_embedded_credentials_are_not_followed(self):
        request = Request("https://api.github.com/asset")
        for destination in ("http://release-assets.githubusercontent.com/asset", "https://user:secret@example.com/asset"):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                download.AssetRedirect().redirect_request(request, None, 302, "Found", {}, destination)

    def test_authenticated_initial_requests_cannot_target_another_host(self):
        with self.assertRaises(ValueError), patch.object(download.OPENER, "open") as opened:
            download.open_api("https://example.com/asset", "test-secret")
        opened.assert_not_called()

    def test_asset_lookup_follows_pages_without_dropping_later_assets(self):
        def response(value, link=""):
            stream = io.BytesIO(json.dumps(value).encode())
            stream.headers = {"Link": link}
            return stream

        url = "https://api.github.com/repos/owner/repository/releases/1/assets"
        with patch.object(download, "open_api", side_effect=[
            response({"tag_name": "retained-tag", "id": 1}),
            response([{"name": "other"}], f'<{url}?page=2>; rel="next"'),
            response([{"name": "retained"}]),
        ]) as opened:
            assets = download.release_assets("owner/repository", "retained-tag", "test-secret")
        self.assertEqual(assets, [{"name": "other"}, {"name": "retained"}])
        self.assertEqual(opened.call_args.args[0], url + "?page=2")

    def test_pagination_cannot_move_authentication_to_another_host(self):
        release = io.BytesIO(b'{"tag_name":"retained-tag","id":1}')
        page = io.BytesIO(b"[]")
        page.headers = {"Link": '<https://example.com/asset>; rel="next"'}
        with patch.object(download, "open_api", side_effect=[release, page]) as opened, \
                self.assertRaisesRegex(ValueError, "pagination link"):
            download.release_assets("owner/repository", "retained-tag", "test-secret")
        self.assertEqual(opened.call_count, 2)

    def test_download_uses_matched_asset_id_and_checks_retained_bytes(self):
        content = b"retained witness bytes"
        expected = self.expected(content)
        remote = {"name": expected["name"], "size": len(content), "state": "uploaded", "id": 1,
                  "url": "https://unexpected.example/ignored"}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(download, "expected_assets", return_value=[expected]), \
                patch.object(download, "release_assets", return_value=[remote]), \
                patch.object(download, "open_api", return_value=io.BytesIO(content)) as opened:
            destination = Path(directory) / "archives"
            receipt = download.download(destination, "test-secret")
            self.assertEqual((destination / expected["name"]).read_bytes(), content)
            self.assertEqual(receipt["archives"][0]["sha256"], expected["sha256"])
            self.assertEqual(opened.call_args.args[0],
                             "https://api.github.com/repos/owner/repository/releases/assets/1")

    def test_bad_metadata_and_duplicate_names_fail_before_download(self):
        expected = self.expected(b"content")
        remote = {"name": expected["name"], "size": expected["bytes"], "state": "uploaded", "id": 1}
        for assets in ([], [remote, remote], [{**remote, "size": 0}], [{**remote, "state": "new"}]):
            with self.subTest(assets=assets), tempfile.TemporaryDirectory() as directory, \
                    patch.object(download, "expected_assets", return_value=[expected]), \
                    patch.object(download, "release_assets", return_value=assets), \
                    patch.object(download, "open_api") as opened, self.assertRaises(ValueError):
                download.download(Path(directory) / "archives", "test-secret")
            opened.assert_not_called()

    def test_changed_download_does_not_become_an_accepted_archive(self):
        expected = self.expected(b"original")
        remote = {"name": expected["name"], "size": expected["bytes"], "state": "uploaded", "id": 1}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(download, "expected_assets", return_value=[expected]), \
                patch.object(download, "release_assets", return_value=[remote]), \
                patch.object(download, "open_api", return_value=io.BytesIO(b"modified")):
            destination = Path(directory) / "archives"
            with self.assertRaisesRegex(ValueError, "differs from its retained source record"):
                download.download(destination, "test-secret")
            self.assertEqual(list(destination.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
