from pathlib import Path

from r2_storage import (
    r2_is_configured,
    r2_object_exists,
    r2_put_marker,
    r2_upload_file,
)


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "static"
MARKER_KEY = "_migration/initial-uploads-v1-complete"


def main():
    if not r2_is_configured():
        raise SystemExit("R2 is not configured; refusing to run photo migration.")

    if r2_object_exists(MARKER_KEY):
        print("R2 photo migration already completed; nothing to do.")
        return

    upload_roots = [
        STATIC_ROOT / "uploads" / "coins",
        STATIC_ROOT / "uploads" / "artifacts",
    ]

    files = []
    for folder in upload_roots:
        if folder.exists():
            files.extend(path for path in folder.rglob("*") if path.is_file())

    total = len(files)
    print(f"Starting R2 migration for {total} files.")

    for index, path in enumerate(files, start=1):
        key = path.relative_to(STATIC_ROOT).as_posix()

        if not r2_object_exists(key):
            r2_upload_file(path, key)

        if index == 1 or index % 25 == 0 or index == total:
            print(f"R2 migration progress: {index}/{total}")

    r2_put_marker(MARKER_KEY)
    print("R2 photo migration complete.")


if __name__ == "__main__":
    main()
