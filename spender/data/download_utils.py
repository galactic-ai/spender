import requests
from requests.adapters import HTTPAdapter, Retry

from pathlib import Path
from typing import Tuple

# make a small resuable downloader
def _make_session(total_retries=5, backoff=0.5, timeout=15.0):
    session = requests.Session()
    retries = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["HEAD", "GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=64, pool_maxsize=64)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.request_timeout = timeout
    session.headers.update({
        "User-Agent": "downloader/1.0"
    })
    return session


def _download_one(session: requests.Session,
                url: str, dest: Path, chunk: int = 1024 * 1024) -> Tuple[str, bool]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return str(dest), True

    tmp = dest.with_suffix(dest.suffix + '.part')
    print(f"Downloading {url} to {dest}...\n")
    print(f"Temporary file: {tmp}\n")

    # checking if server supports ranges + get total size
    try:
        head = session.head(url, timeout=session.request_timeout, allow_redirects=True)
        head.raise_for_status()
    except Exception as e:
        head = None
    
    supports_range = False
    total_size = None
    if head is not None:
        supports_range = head.headers.get("Accept-Ranges", "").lower() == "bytes"
        try:
            total_size = int(head.headers.get("Content-Length")) if head.headers.get("Content-Length") else None
        except (TypeError, ValueError):
            total_size = None 
        
        # offset
        offset = tmp.stat().st_size if tmp.exists() else 0
        headers = {}

        if offset and supports_range:
            headers["Range"] = f"bytes={offset}-"
        
        try:
            with session.get(url, stream=True, timeout=session.request_timeout, headers=headers) as r:
                # If we asked for a range but got 200, server ignored Range -> restart from scratch
                if headers.get("Range") and r.status_code == 200:
                    offset = 0  # restart clean
                elif headers.get("Range") and r.status_code not in (200, 206):
                    r.raise_for_status()

                mode = "ab" if offset else "wb"
                with open(tmp, mode) as f:
                    for buf in r.iter_content(chunk_size=chunk):
                        if buf:
                            f.write(buf)

            # Optional integrity check: if total size known, verify final length
            if total_size is not None:
                final_size = tmp.stat().st_size
                if final_size != total_size:
                    raise IOError(f"Incomplete download for {dest.name}: {final_size}/{total_size} bytes")

            tmp.replace(dest)
            return str(dest), True

        except Exception:
            # Clean up only if it's an obvious failure with no resume possiblity;
            # keep the .part file to allow resume on the next run.
            # Comment the next 3 lines if you prefer to always delete on failure.
            # try:
            #     if tmp.exists() and not supports_range:
            #         tmp.unlink()
            # except Exception:
            #     pass
            raise