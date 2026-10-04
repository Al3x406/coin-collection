import mimetypes
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError
from flask import has_request_context, request
from PIL import Image, ImageOps


_REQUIRED = (
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_ENDPOINT_URL",
    "R2_BUCKET_NAME",
)

_REFERENCE_CACHE_PREFIX = "uploads/coins/reference_cache/"
_REFERENCE_MAX_BYTES = 10 * 1024 * 1024
_REFERENCE_MAX_DIMENSION = 1000
_REFERENCE_JPEG_QUALITY = 84


def r2_is_configured():
    return all(os.getenv(name) for name in _REQUIRED)


@lru_cache(maxsize=1)
def _client():
    if not r2_is_configured():
        return None

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name=os.getenv("R2_REGION", "auto"),
    )


def _bucket():
    return os.environ["R2_BUCKET_NAME"]


def _trusted_numista_image_url(source_url):
    if not source_url:
        return False

    try:
        parsed = urlsplit(source_url)
    except (TypeError, ValueError):
        return False

    hostname = (parsed.hostname or "").lower()
    return (
        parsed.scheme == "https"
        and bool(hostname)
        and (
            hostname == "numista.com"
            or hostname.endswith(".numista.com")
        )
    )


def _cache_numista_reference(key, source_url):
    """Download one trusted Numista image, optimize it, and store it in R2."""
    if not r2_is_configured() or not _trusted_numista_image_url(source_url):
        return False

    try:
        response = requests.get(
            source_url,
            timeout=20,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; CoinCollectionReferenceCache/1.0)"
                ),
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                "Referer": "https://en.numista.com/",
            },
        )
        response.raise_for_status()

        content_type = (response.headers.get("Content-Type") or "").lower()
        if content_type and not content_type.startswith("image/"):
            return False

        if len(response.content) > _REFERENCE_MAX_BYTES:
            return False

        image = ImageOps.exif_transpose(
            Image.open(BytesIO(response.content))
        ).convert("RGB")
        image.thumbnail(
            (_REFERENCE_MAX_DIMENSION, _REFERENCE_MAX_DIMENSION),
            Image.Resampling.LANCZOS,
        )

        output = BytesIO()
        image.save(
            output,
            "JPEG",
            quality=_REFERENCE_JPEG_QUALITY,
            optimize=True,
            progressive=True,
        )
        output.seek(0)

        _client().put_object(
            Bucket=_bucket(),
            Key=key,
            Body=output.getvalue(),
            ContentType="image/jpeg",
            CacheControl="public, max-age=31536000, immutable",
        )
        return True
    except (
        requests.RequestException,
        OSError,
        ValueError,
        ClientError,
        BotoCoreError,
    ):
        return False


def r2_upload_file(local_path, key):
    if not r2_is_configured():
        return False

    local_path = Path(local_path)
    if not local_path.is_file():
        return False

    content_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"

    _client().upload_file(
        str(local_path),
        _bucket(),
        key,
        ExtraArgs={"ContentType": content_type},
    )
    return True


def r2_download_file(local_path, key):
    if not r2_is_configured():
        return False

    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _client().download_file(_bucket(), key, str(local_path))
        return True
    except (ClientError, BotoCoreError):
        return False


def r2_delete_key(key):
    if not r2_is_configured():
        return False

    try:
        _client().delete_object(Bucket=_bucket(), Key=key)
        return True
    except (ClientError, BotoCoreError):
        return False


def r2_delete_prefix(prefix):
    if not r2_is_configured():
        return False

    continuation = None

    try:
        while True:
            params = {
                "Bucket": _bucket(),
                "Prefix": prefix,
                "MaxKeys": 1000,
            }

            if continuation:
                params["ContinuationToken"] = continuation

            response = _client().list_objects_v2(**params)
            objects = response.get("Contents", [])

            if objects:
                _client().delete_objects(
                    Bucket=_bucket(),
                    Delete={
                        "Objects": [{"Key": item["Key"]} for item in objects],
                        "Quiet": True,
                    },
                )

            if not response.get("IsTruncated"):
                break

            continuation = response.get("NextContinuationToken")

        return True
    except (ClientError, BotoCoreError):
        return False


def r2_presigned_url(key, expires_in=3600):
    if not r2_is_configured():
        return None

    # Numista catalogue images are lazily copied into R2 the first time a
    # browser asks for them. This avoids hot-link failures without downloading
    # the entire catalogue up front.
    if key.startswith(_REFERENCE_CACHE_PREFIX):
        source_url = None
        if has_request_context():
            source_url = (request.args.get("source") or "").strip()

        if not r2_object_exists(key):
            _cache_numista_reference(key, source_url)

        if not r2_object_exists(key):
            # If Numista temporarily blocks the server-side copy, fall back to
            # the original trusted image URL rather than returning a signed 404.
            return source_url if _trusted_numista_image_url(source_url) else None

    try:
        return _client().generate_presigned_url(
            "get_object",
            Params={"Bucket": _bucket(), "Key": key},
            ExpiresIn=expires_in,
        )
    except (ClientError, BotoCoreError):
        return None


def r2_object_exists(key):
    if not r2_is_configured():
        return False

    try:
        _client().head_object(Bucket=_bucket(), Key=key)
        return True
    except (ClientError, BotoCoreError):
        return False


def r2_put_marker(key, body=b"ok"):
    if not r2_is_configured():
        return False

    _client().put_object(
        Bucket=_bucket(),
        Key=key,
        Body=body,
        ContentType="text/plain",
    )
    return True
