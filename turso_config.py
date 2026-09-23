import os


def turso_is_configured():
    return bool(
        os.getenv("TURSO_DATABASE_URL")
        and os.getenv("TURSO_AUTH_TOKEN")
    )


def turso_sqlalchemy_uri():
    raw = (os.getenv("TURSO_DATABASE_URL") or "").strip()

    if not raw:
        raise RuntimeError("TURSO_DATABASE_URL is not set.")

    for prefix in ("libsql://", "https://", "http://"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break

    separator = "&" if "?" in raw else "?"
    return f"sqlite+libsql://{raw}{separator}secure=true"


def turso_connect_args():
    token = os.getenv("TURSO_AUTH_TOKEN")

    if not token:
        raise RuntimeError("TURSO_AUTH_TOKEN is not set.")

    return {"auth_token": token}
