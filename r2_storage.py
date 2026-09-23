import mimetypes
import os
from functools import lru_cache
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError


_REQUIRED = (
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_ENDPOINT_URL",
    "R2_BUCKET_NAME",
)


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
