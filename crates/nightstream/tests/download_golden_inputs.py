#!/usr/bin/env python3
"""Download the exact golden archives with the repository's GitHub Actions token."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from restore_golden_inputs import checked_asset, expected_assets

API = "https://api.github.com"


class AssetRedirect(HTTPRedirectHandler):
    """GitHub asset redirects must not carry API credentials to another host."""

    def redirect_request(self, request, response, code, message, headers, url):
        destination = urlsplit(url)
        if destination.scheme != "https" or destination.username or destination.password:
            raise ValueError("release asset redirect must use HTTPS without URL credentials")
        redirected = super().redirect_request(request, response, code, message, headers, url)
        if redirected is not None and urlsplit(request.full_url).netloc != destination.netloc:
            for collection in (redirected.headers, redirected.unredirected_hdrs):
                for key in list(collection):
                    if key.lower() == "authorization":
                        del collection[key]
        return redirected


OPENER = build_opener(AssetRedirect())


def open_api(url, token, accept="application/vnd.github+json"):
    if urlsplit(url).scheme != "https" or urlsplit(url).netloc != "api.github.com":
        raise ValueError("authenticated release requests must use the GitHub API")
    return OPENER.open(Request(url, headers={
        "Authorization": f"Bearer {token}", "Accept": accept,
        "X-GitHub-Api-Version": "2026-03-10", "User-Agent": "Nightstream-golden-conformance",
    }))


def release_assets(repository, tag, token):
    repository_path = "/".join(quote(part, safe="") for part in repository.split("/"))
    base = f"{API}/repos/{repository_path}/releases"
    with open_api(f"{base}/tags/{quote(tag, safe='')}", token) as response:
        release = json.load(response)
    if release["tag_name"] != tag or type(release["id"]) is not int:
        raise ValueError("release lookup returned a different tag or invalid identifier")
    asset_path = f"{base}/{release['id']}/assets"
    url, visited, assets = asset_path, set(), []
    while url:
        if url in visited or urlsplit(url)._replace(query="").geturl() != asset_path:
            raise ValueError("invalid or repeated release asset pagination link")
        visited.add(url)
        with open_api(url, token) as response:
            assets.extend(json.load(response))
            links = response.headers.get("Link", "")
        url = None
        for link in links.split(","):
            parts = [part.strip() for part in link.split(";")]
            if 'rel="next"' in parts[1:]:
                url = parts[0].removeprefix("<").removesuffix(">")
    return assets


def download(directory, token):
    if not token:
        raise ValueError("provide --token-stdin or an existing GITHUB_TOKEN")
    directory.mkdir(parents=True, exist_ok=False)
    available, receipts = {}, []
    for expected in expected_assets():
        name = expected["name"]
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("archive name is not a filename")
        key = expected["repository"], expected["tag"]
        if key not in available:
            available[key] = release_assets(*key, token)
        matches = [asset for asset in available[key] if asset["name"] == name]
        if len(matches) != 1:
            raise ValueError(f"release must contain exactly one retained archive: {name}")
        asset, = matches
        if (asset["size"] != expected["bytes"] or asset["state"] != "uploaded"
                or type(asset["id"]) is not int):
            raise ValueError(f"release archive metadata differs from the retained record: {name}")
        temporary = directory / (name + ".part")
        try:
            # Construct the API URL from the matched ID. Do not trust a URL in
            # the response as a destination for the authorization header.
            api_path = "/".join(quote(part, safe="") for part in expected["repository"].split("/"))
            with open_api(f"{API}/repos/{api_path}/releases/assets/{asset['id']}", token,
                          "application/octet-stream") as response, temporary.open("xb") as output:
                shutil.copyfileobj(response, output)
            _, receipt = checked_asset(directory, temporary.name, expected["bytes"], expected["sha256"])
            temporary.rename(directory / name)
        finally:
            temporary.unlink(missing_ok=True)
        receipts.append({**receipt, "file": name, "repository": key[0], "release_tag": key[1]})
    return {"schema": 1, "archives": receipts,
            "scope": "Downloaded archive identity checks only; restoration and semantic checks remain required."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--token-stdin", action="store_true", help="read the Actions token from stdin")
    args = parser.parse_args()
    token = sys.stdin.read().strip() if args.token_stdin else os.environ.get("GITHUB_TOKEN", "")
    try:
        print(json.dumps(download(args.directory.resolve(), token)), flush=True)
        return 0
    except HTTPError as error:
        print(f"golden archive download failed: GitHub HTTP {error.code}", file=sys.stderr)
    except URLError:
        print("golden archive download failed: network request failed", file=sys.stderr)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"golden archive download failed: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
