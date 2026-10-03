from flask import (
    Flask,
    render_template,
    render_template_string,
    request,
    redirect,
    url_for,
    session,
    abort,
    flash,
    send_file,
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import joinedload, selectinload
from sqlalchemy import text, create_engine, select
from datetime import datetime
from io import BytesIO
import os
import uuid
from pathlib import Path
import hmac
from urllib.parse import urlsplit
import requests
import re
import time
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from r2_storage import (
    r2_delete_key,
    r2_delete_prefix,
    r2_download_file,
    r2_is_configured,
    r2_presigned_url,
    r2_upload_file,
)
load_dotenv()

# Heavy image/OCR libraries are loaded only when an image feature is used.
# Keeping them out of normal Flask startup lowers idle RAM on Railway.
cv2 = None
np = None
pytesseract = None
Image = None
ImageOps = None
ImageEnhance = None
ImageDraw = None


def _load_image_tools():
    global cv2, np, pytesseract, Image, ImageOps, ImageEnhance, ImageDraw
    if cv2 is not None:
        return

    import cv2 as _cv2
    import numpy as _np
    import pytesseract as _pytesseract
    from PIL import Image as _Image, ImageOps as _ImageOps, ImageEnhance as _ImageEnhance, ImageDraw as _ImageDraw

    _pytesseract.pytesseract.tesseract_cmd = os.getenv(
        "TESSERACT_CMD",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe" if os.name == "nt" else "tesseract",
    )
    cv2, np, pytesseract = _cv2, _np, _pytesseract
    Image, ImageOps = _Image, _ImageOps
    ImageEnhance, ImageDraw = _ImageEnhance, _ImageDraw

app = Flask(__name__)

# Security settings.
# OWNER_KEY stays private. VISITOR_KEY is the password you can share
# with friends/family. The owner key also signs the Flask session cookie.
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY") or os.getenv("OWNER_KEY")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Routes that friends/family should never be able to use.
OWNER_ONLY_ENDPOINTS = {
    "add_coin",
    "add_mint",
    "edit_coin",
    "update_reference_images",
    "remove_reference_image",
    "delete_coin",
    "upload_coin_photos",
    "remove_coin_photo",
    "numista_search",
    "numista_type_details",
    "numista_type_issues",
    "identify_coin_ocr",
    "identify_coin",
    "create_set",
    "edit_set",
    "delete_set",
    "start_preset_set",
    "create_ww2_goal_set",
    "add_artifact",
    "edit_artifact",
    "delete_artifact",
    "set_featured_coin",
    "edit_coin_photo",
    "edit_artifact_photo",
    "remove_artifact_photo",
    "collection_value",
    "human_story_choose_coin",
    "assign_set_slot_coin",
    "download_database",
}


def is_safe_local_path(target):
    """Only allow redirects back to this Flask site."""
    if not target:
        return False

    parsed = urlsplit(target)

    return (
        not parsed.scheme
        and not parsed.netloc
        and target.startswith("/")
    )


def has_view_access():
    """True for either a visitor session or an owner session."""
    return bool(
        session.get("visitor_access")
        or session.get("is_owner")
    )


@app.before_request
def protect_private_collection():
    """
    Keep the collection publicly viewable, require owner authentication for
    modifying routes, and let R2 transparently serve uploaded media when the
    local Railway copy is not present.
    """

    if request.endpoint == "static":
        uploads_prefix = "/static/uploads/"

        if request.path.startswith(uploads_prefix) and r2_is_configured():
            relative_key = request.path[len("/static/"):]
            local_path = Path(app.static_folder) / relative_key

            if not local_path.exists():
                signed_url = r2_presigned_url(relative_key)
                if signed_url:
                    return redirect(signed_url)

        return None

    if request.endpoint in {
        "owner_login",
        "owner_logout",
        "access_login",
        "access_logout",
    }:
        return None

    if (
        request.endpoint in OWNER_ONLY_ENDPOINTS
        and not session.get("is_owner")
    ):
        if request.method in {"GET", "HEAD"}:
            return redirect(
                url_for(
                    "owner_login",
                    next=request.full_path
                )
            )

        abort(403)

    return None


@app.context_processor
def inject_access_status():
    """Makes access state available to every Jinja template."""
    return {
        "is_owner": bool(session.get("is_owner")),
        "has_view_access": has_view_access(),
    }


@app.route("/owner/download-database")
def download_database():
    if not session.get("is_owner"):
        abort(403)

    # A local SQLite deployment can send its database file directly.
    database_name = db.engine.url.database
    if database_name:
        database_path = Path(database_name)
        if not database_path.is_absolute():
            database_path = Path(app.instance_path) / database_path
        if database_path.exists():
            return send_file(
                database_path,
                as_attachment=True,
                download_name="coin-collection.db",
                mimetype="application/vnd.sqlite3",
                max_age=0,
            )

    # Production uses Turso/libSQL, which has no local database filepath.
    # Build a portable SQLite snapshot by copying every SQLAlchemy table.
    backup_dir = Path(app.instance_path) / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / "coin-collection-download.db"
    if backup_path.exists():
        backup_path.unlink()

    local_engine = create_engine(f"sqlite:///{backup_path}")
    try:
        db.metadata.create_all(local_engine)

        with db.engine.connect() as source, local_engine.begin() as target:
            for table in db.metadata.sorted_tables:
                rows = source.execute(select(table)).mappings().all()
                if rows:
                    target.execute(table.insert(), [dict(row) for row in rows])
    finally:
        local_engine.dispose()

    return send_file(
        backup_path,
        as_attachment=True,
        download_name="coin-collection.db",
        mimetype="application/vnd.sqlite3",
        max_age=0,
    )


@app.route("/access", methods=["GET", "POST"])
def access_login():
    """
    Friends/family can enter VISITOR_KEY for read-only browsing.
    OWNER_KEY also works here and grants full owner access.
    """
    if has_view_access():
        return redirect(url_for("home"))

    error = None
    next_url = request.args.get("next", "")

    if request.method == "POST":
        submitted_key = request.form.get("access_key", "")
        visitor_key = os.getenv("VISITOR_KEY", "")
        owner_key = os.getenv("OWNER_KEY", "")
        next_url = request.form.get("next", "")

        owner_match = (
            bool(owner_key)
            and hmac.compare_digest(
                submitted_key,
                owner_key
            )
        )

        visitor_match = (
            bool(visitor_key)
            and hmac.compare_digest(
                submitted_key,
                visitor_key
            )
        )

        if owner_match or visitor_match:
            session.clear()
            session["visitor_access"] = True

            if owner_match:
                session["is_owner"] = True

            if not is_safe_local_path(next_url):
                next_url = url_for("home")

            return redirect(next_url)

        error = "That access password is not correct."

    return render_template_string(
        """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Collection Access · My Coin Collection</title>
    <style>
        body {
            margin: 0;
            min-height: 100vh;
            display: grid;
            place-items: center;
            background: #07090a;
            color: #f3eee5;
            font-family: Arial, sans-serif;
        }
        .card {
            width: min(92vw, 430px);
            box-sizing: border-box;
            padding: 28px;
            border: 1px solid rgba(230,184,95,.48);
            border-radius: 14px;
            background: rgba(12,14,15,.97);
            box-shadow: 0 18px 55px rgba(0,0,0,.45);
        }
        h1 {
            margin: 0 0 8px;
            color: #e6b85f;
            font-size: 1.65rem;
        }
        p { color: #cfc7ba; line-height: 1.45; }
        label { display: block; margin: 18px 0 7px; }
        input {
            width: 100%;
            box-sizing: border-box;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #6c654f;
            background: #111415;
            color: #fff;
        }
        button {
            width: 100%;
            margin-top: 16px;
            padding: 12px;
            border: 1px solid #e6b85f;
            border-radius: 8px;
            background: #e6b85f;
            color: #15120b;
            font-weight: 700;
            cursor: pointer;
        }
        .error {
            margin-top: 14px;
            padding: 10px;
            border-radius: 8px;
            background: #351616;
            color: #ffd2d2;
        }
    </style>
</head>
<body>
    <main class="card">
        <h1>Private Collection</h1>
        <p>Enter the collection password to view this coin collection.</p>

        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}

        <form method="post">
            <input type="hidden" name="next" value="{{ next_url }}">
            <label for="access_key">Collection password</label>
            <input id="access_key" name="access_key" type="password"
                   autocomplete="current-password" required autofocus>
            <button type="submit">Enter Collection</button>
        </form>
    </main>
</body>
</html>
        """,
        error=error,
        next_url=next_url
    )


@app.route("/access/logout", methods=["POST"])
def access_logout():
    """Fully sign out of visitor and owner access."""
    session.clear()
    return redirect(url_for("access_login"))


@app.route("/owner", methods=["GET", "POST"])
def owner_login():
    """Upgrade this browser to owner access using OWNER_KEY from .env."""
    if session.get("is_owner"):
        return redirect(url_for("home"))

    error = None
    next_url = request.args.get("next", "")

    if request.method == "POST":
        submitted_key = request.form.get("owner_key", "")
        expected_key = os.getenv("OWNER_KEY", "")
        next_url = request.form.get("next", "")

        if (
            expected_key
            and hmac.compare_digest(
                submitted_key,
                expected_key
            )
        ):
            session.clear()
            session["visitor_access"] = True
            session["is_owner"] = True

            if not is_safe_local_path(next_url):
                next_url = url_for("home")

            return redirect(next_url)

        error = "That owner key is not correct."

    return render_template_string(
        """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Owner Access · My Coin Collection</title>
    <style>
        body {
            margin: 0;
            min-height: 100vh;
            display: grid;
            place-items: center;
            background: #07090a;
            color: #f3eee5;
            font-family: Arial, sans-serif;
        }
        .card {
            width: min(92vw, 430px);
            box-sizing: border-box;
            padding: 28px;
            border: 1px solid rgba(230,184,95,.48);
            border-radius: 14px;
            background: rgba(12,14,15,.97);
            box-shadow: 0 18px 55px rgba(0,0,0,.45);
        }
        h1 {
            margin: 0 0 8px;
            color: #e6b85f;
            font-size: 1.65rem;
        }
        p { color: #cfc7ba; line-height: 1.45; }
        label { display: block; margin: 18px 0 7px; }
        input {
            width: 100%;
            box-sizing: border-box;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #6c654f;
            background: #111415;
            color: #fff;
        }
        button {
            width: 100%;
            margin-top: 16px;
            padding: 12px;
            border: 1px solid #e6b85f;
            border-radius: 8px;
            background: #e6b85f;
            color: #15120b;
            font-weight: 700;
            cursor: pointer;
        }
        .error {
            margin-top: 14px;
            padding: 10px;
            border-radius: 8px;
            background: #351616;
            color: #ffd2d2;
        }
        a { color: #e6b85f; }
    </style>
</head>
<body>
    <main class="card">
        <h1>Owner Access</h1>
        <p>Enter your private owner key to unlock Add, Edit, Delete, photo management, and coin identification in this browser.</p>

        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}

        <form method="post">
            <input type="hidden" name="next" value="{{ next_url }}">
            <label for="owner_key">Owner key</label>
            <input id="owner_key" name="owner_key" type="password"
                   autocomplete="current-password" required autofocus>
            <button type="submit">Unlock Owner Mode</button>
        </form>

        <p><a href="{{ url_for('home') }}">Return to collection</a></p>
    </main>
</body>
</html>
        """,
        error=error,
        next_url=next_url
    )


@app.route("/owner/logout", methods=["POST"])
def owner_logout():
    """Leave owner mode but remain signed in as a read-only visitor."""
    session.clear()
    session["visitor_access"] = True
    return redirect(url_for("home"))


# Database configuration.
# Keep SQLite as the default until the one-time Turso migration is verified.
if os.getenv("USE_TURSO") == "1":
    from turso_config import turso_connect_args, turso_sqlalchemy_uri

    app.config["SQLALCHEMY_DATABASE_URI"] = turso_sqlalchemy_uri()
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "connect_args": turso_connect_args(),
        "pool_pre_ping": True,
    }
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
        "DATABASE_URL",
        "sqlite:///coins.db",
    )

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
UPLOAD_FOLDER = os.path.join(
    app.root_path,
    "static",
    "uploads",
    "coins"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)
# Connect SQLAlchemy to Flask
db = SQLAlchemy(app)

class Mint(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(150), nullable=False)
    city = db.Column(db.String(100))
    state = db.Column(db.String(100))
    country = db.Column(db.String(100))

    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)

    # CANOA / historical information
    canoa_id = db.Column(db.Integer)
    start_year = db.Column(db.Integer)
    end_year = db.Column(db.Integer)

    # Numista information
    numista_id = db.Column(db.Integer)
    nomisma_id = db.Column(db.String(100))
    wikidata_id = db.Column(db.String(100))

    coins = db.relationship(
        "Coin",
        back_populates="mint_record"
    )

class HistoricalEntity(db.Model):
    """
    A historical political/geographic entity used by the Geography page.

    Coin.country remains the coin's issuer label. This table only supplies
    historical dates and modern-map reference areas for navigation.
    """
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False, unique=True)
    entity_type = db.Column(db.String(50))
    start_year = db.Column(db.Integer)
    end_year = db.Column(db.Integer)
    modern_map_names = db.Column(db.Text)
    aliases = db.Column(db.Text)
    notes = db.Column(db.Text)


class Coin(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(
        db.String(100),
        nullable=False
    )

    country = db.Column(
        db.String(100)
    )

    # Stable country identity used by generated sets. The original issuer
    # above remains untouched for historically accurate catalog display.
    set_country = db.Column(db.String(100))
    set_country_review = db.Column(db.Boolean, default=False, nullable=False)

    year = db.Column(
        db.Integer
    )

    # Flexible dating. Existing exact dates continue to use year.
    date_start_year = db.Column(db.Integer)
    date_end_year = db.Column(db.Integer)
    date_is_approx = db.Column(db.Boolean, default=False)

    @property
    def date_display(self):
        start = self.date_start_year
        end = self.date_end_year
        if start is None and end is None:
            start = self.year
            end = self.year
        if start is None and end is None:
            return "Unknown"

        def fmt(value):
            if value is None:
                return "?"
            return f"{abs(value)} BC" if value < 0 else str(value)

        if start == end or end is None:
            label = fmt(start)
        elif start is None:
            label = fmt(end)
        else:
            label = f"{fmt(start)}–{fmt(end)}"
        return f"c. {label}" if self.date_is_approx else label

    # Old text fields kept for compatibility
    mint = db.Column(
        db.String(100)
    )

    mint_mark = db.Column(
        db.String(10)
    )

    location = db.Column(
        db.String(200)
    )

    mint_id = db.Column(
        db.Integer,
        db.ForeignKey("mint.id"),
        nullable=True
    )

    mint_record = db.relationship(
        "Mint",
        back_populates="coins"
    )

    denomination = db.Column(
        db.String(50)
    )

    material = db.Column(
        db.String(50)
    )

    condition = db.Column(
        db.String(50)
    )

    quantity = db.Column(
        db.Integer,
        default=1
    )

    date_acquired = db.Column(
        db.String(20)
    )

    notes = db.Column(
        db.Text
    )

    # Numista coin information
    numista_type_id = db.Column(
        db.Integer
    )

    numista_issue_id = db.Column(
        db.Integer
    )

    variant = db.Column(
        db.String(200)
    )

    mintage = db.Column(
        db.Integer
    )
    # Numista reference images
    obverse_image = db.Column(
        db.Text
    )

    reverse_image = db.Column(
        db.Text
    )

    numista_url = db.Column(
        db.Text
    )

    obverse_copyright = db.Column(
        db.String(200)
    )

    obverse_license = db.Column(
        db.String(200)
    )

    reverse_copyright = db.Column(
        db.String(200)
    )

    reverse_license = db.Column(
        db.String(200)
    )
    # Photos of my actual coin
    personal_obverse_image = db.Column(
        db.Text
    )

    personal_reverse_image = db.Column(
        db.Text
    )
        # Estimated collection value
    estimated_value = db.Column(
        db.Float
    )

    # Private owner-only purchase price.
    purchase_price = db.Column(
        db.Float
    )

    # Manually selected homepage featured coin.
    featured = db.Column(
        db.Boolean,
        default=False,
        nullable=False
    )

class HumanStorySelection(db.Model):
    __tablename__ = "human_story_selection"
    id = db.Column(db.Integer, primary_key=True)
    moment_key = db.Column(db.String(220), nullable=False, unique=True)
    coin_id = db.Column(db.Integer, db.ForeignKey("coin.id"), nullable=False)
    coin = db.relationship("Coin")


class SetSlotSelection(db.Model):
    """Owner-selected coin override for a generated set slot."""
    __tablename__ = "set_slot_selection"
    id = db.Column(db.Integer, primary_key=True)
    set_id = db.Column(db.Integer, db.ForeignKey("coin_set.id"), nullable=False, index=True)
    slot_type = db.Column(db.String(30), nullable=False)
    slot_id = db.Column(db.Integer, nullable=False)
    coin_id = db.Column(db.Integer, db.ForeignKey("coin.id"), nullable=False)
    coin = db.relationship("Coin")
    __table_args__ = (
        db.UniqueConstraint("set_id", "slot_type", "slot_id", name="uq_set_slot_selection"),
    )


# --- SETS FEATURE MODELS ---

coin_set_memberships = db.Table(
    "coin_set_memberships",
    db.Column("set_id", db.Integer, db.ForeignKey("coin_set.id"), primary_key=True),
    db.Column("coin_id", db.Integer, db.ForeignKey("coin.id"), primary_key=True)
)


class CoinSet(db.Model):
    __tablename__ = "coin_set"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False, unique=True)
    description = db.Column(db.String(500))

    coins = db.relationship(
        "Coin",
        secondary=coin_set_memberships,
        lazy="select",
        backref=db.backref("coin_sets", lazy="select")
    )



# --- PRESET SET REQUIREMENTS ---

class CoinSetRequirement(db.Model):
    __tablename__ = "coin_set_requirement"

    id = db.Column(db.Integer, primary_key=True)
    set_id = db.Column(db.Integer, db.ForeignKey("coin_set.id"), nullable=False)
    label = db.Column(db.String(160), nullable=False)
    description = db.Column(db.String(300))
    country_terms = db.Column(db.String(300))
    name_terms = db.Column(db.String(300))
    denomination_terms = db.Column(db.String(300))
    year_start = db.Column(db.Integer)
    year_end = db.Column(db.Integer)
    sort_order = db.Column(db.Integer, default=0)

    coin_set = db.relationship(
        "CoinSet",
        backref=db.backref(
            "requirements",
            lazy="select",
            cascade="all, delete-orphan"
        )
    )



# --- WWII GOAL SET REQUIREMENTS ---

class WW2GoalSlot(db.Model):
    __tablename__ = "ww2_goal_slot"
    id = db.Column(db.Integer, primary_key=True)
    set_id = db.Column(db.Integer, db.ForeignKey("coin_set.id"), nullable=False)
    country_label = db.Column(db.String(120), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    country_terms = db.Column(db.String(300))
    sort_order = db.Column(db.Integer, default=0)
    coin_set = db.relationship(
        "CoinSet",
        backref=db.backref("ww2_slots", lazy="select", cascade="all, delete-orphan")
    )


# --- CABINET OF CURIOSITIES MODEL ---
class Artifact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(180), nullable=False)
    category = db.Column(db.String(80), nullable=False)
    year_text = db.Column(db.String(80))
    country = db.Column(db.String(120))
    description = db.Column(db.Text)
    condition = db.Column(db.String(100))
    estimated_value = db.Column(db.Float)
    purchase_price = db.Column(db.Float)
    date_acquired = db.Column(db.String(30))
    source = db.Column(db.String(180))
    notes = db.Column(db.Text)
    image_filename = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    photos = db.relationship(
        "ArtifactPhoto",
        back_populates="artifact",
        cascade="all, delete-orphan",
        order_by="ArtifactPhoto.sort_order",
        lazy="select",
    )


class ArtifactPhoto(db.Model):
    __tablename__ = "artifact_photo"
    id = db.Column(db.Integer, primary_key=True)
    artifact_id = db.Column(db.Integer, db.ForeignKey("artifact.id"), nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)
    sort_order = db.Column(db.Integer, default=0, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    artifact = db.relationship("Artifact", back_populates="photos")


# Home page / collection dashboard
@app.route("/")
def home():
    all_coins = (
        Coin.query
        .options(joinedload(Coin.mint_record))
        .order_by(Coin.id.desc())
        .all()
    )

    total_coins = sum(coin.quantity or 1 for coin in all_coins)

    unique_type_keys = set()
    for coin in all_coins:
        if coin.numista_type_id:
            unique_type_keys.add(("numista", coin.numista_type_id))
        else:
            unique_type_keys.add((
                "local",
                (coin.name or "").strip().lower(),
                (coin.country or "").strip().lower(),
                (coin.denomination or "").strip().lower()
            ))
    unique_types = len(unique_type_keys)

    countries = {
        coin.country.strip()
        for coin in all_coins
        if coin.country and coin.country.strip()
    }
    countries_count = len(countries)

    represented_mints = set()
    for coin in all_coins:
        if coin.mint_id:
            represented_mints.add(("id", coin.mint_id))
        elif coin.mint and coin.mint.strip():
            represented_mints.add(("name", coin.mint.strip().lower()))
    mints_count = len(represented_mints)

    coin_estimated_value = sum(
        (coin.estimated_value or 0) * (coin.quantity or 1)
        for coin in all_coins
    )

    all_artifacts_home = Artifact.query.all()

    artifact_estimated_value = sum(
        artifact.estimated_value or 0
        for artifact in all_artifacts_home
    )

    estimated_value = (
        coin_estimated_value
        + artifact_estimated_value
    )

    recent_coins = all_coins[:6]

    collection_mint_points = []
    seen_mint_ids = set()
    for coin in all_coins:
        mint = coin.mint_record
        if (
            mint
            and mint.id not in seen_mint_ids
            and mint.latitude is not None
            and mint.longitude is not None
        ):
            seen_mint_ids.add(mint.id)
            collection_mint_points.append({
                "id": mint.id,
                "name": mint.name,
                "country": mint.country,
                "lat": mint.latitude,
                "lng": mint.longitude
            })

    # Prefer the coin manually selected by the owner.
    featured_coin = next(
        (
            coin for coin in all_coins
            if getattr(coin, "featured", False)
        ),
        None
    )

    # Fall back to the previous automatic behavior until one is selected.
    if featured_coin is None:
        featured_coin = next(
            (
                coin for coin in all_coins
                if coin.obverse_image
                or coin.reverse_image
                or coin.personal_obverse_image
                or coin.personal_reverse_image
            ),
            all_coins[0] if all_coins else None
        )

    dated_coins = [coin for coin in all_coins if coin.year is not None]
    oldest_coin = min(dated_coins, key=lambda c: c.year) if dated_coins else None
    newest_coin = max(dated_coins, key=lambda c: c.year) if dated_coins else None

    valued_coins = [
        coin for coin in all_coins
        if coin.estimated_value is not None
    ]
    most_valuable_coin = max(
        valued_coins,
        key=lambda c: c.estimated_value
    ) if valued_coins else None

    home_sets = {}

    coin_sets = (
        CoinSet.query
        .options(
            joinedload(CoinSet.ww2_slots),
            joinedload(CoinSet.requirements),
        )
        .order_by(CoinSet.name.asc())
        .all()
    )

    matching_coins = sorted(
        all_coins,
        key=lambda coin: (
            coin.year is not None,
            coin.year if coin.year is not None else 0,
            (coin.name or "").lower(),
        ),
    )

    for coin_set in coin_sets:
        key = None

        if hasattr(coin_set, "ww2_slots") and coin_set.ww2_slots:
            key = "ww2"
            matches = _ww2_slot_matches(coin_set, matching_coins)

        elif coin_set.name.startswith("A Coin From Every Year"):
            key = "timeline"
            matches = _match_set_requirements(coin_set, matching_coins)

        elif coin_set.name.startswith("Around the World"):
            key = "world"
            matches = _match_set_requirements(coin_set, matching_coins)

        else:
            continue

        total = len(matches)
        completed = sum(1 for item in matches if item["coin"] is not None)
        percent = round((completed / total * 100) if total else 0)

        home_sets[key] = {
            "set": coin_set,
            "completed": completed,
            "total": total,
            "percent": percent
        }

    return render_template(
        "index.html",
        coins=all_coins,
        total_coins=total_coins,
        unique_types=unique_types,
        countries_count=countries_count,
        mints_count=mints_count,
        estimated_value=estimated_value,
        coin_estimated_value=coin_estimated_value,
        artifact_estimated_value=artifact_estimated_value,
        recent_coins=recent_coins,
        featured_coin=featured_coin,
        collection_mint_points=collection_mint_points,
        oldest_coin=oldest_coin,
        newest_coin=newest_coin,
        most_valuable_coin=most_valuable_coin,
        home_sets=home_sets
    )


_coin_date_schema_checked = False

@app.before_request
def ensure_coin_date_schema():
    """Add flexible coin-date columns to older SQLite databases without losing data."""
    global _coin_date_schema_checked
    if _coin_date_schema_checked:
        return
    try:
        columns = {row[1] for row in db.session.execute(text("PRAGMA table_info(coin)")).fetchall()}
        additions = {
            "date_start_year": "INTEGER",
            "date_end_year": "INTEGER",
            "date_is_approx": "BOOLEAN DEFAULT 0",
            "set_country": "VARCHAR(100)",
            "set_country_review": "BOOLEAN DEFAULT 0",
        }
        for column, sql_type in additions.items():
            if column not in columns:
                db.session.execute(text(f"ALTER TABLE coin ADD COLUMN {column} {sql_type}"))

        # Backfill existing coins once. Clear catalog aliases are normalized;
        # ambiguous historical/multi-territory issuers are flagged for review.
        rows = db.session.execute(
            text("SELECT id, country, set_country FROM coin")
        ).fetchall()
        for coin_id, issuer, existing_set_country in rows:
            if existing_set_country:
                continue
            canonical, needs_review = _canonical_set_country(issuer)
            db.session.execute(
                text("UPDATE coin SET set_country = :country, set_country_review = :review WHERE id = :id"),
                {"country": canonical, "review": 1 if needs_review else 0, "id": coin_id},
            )
        db.session.commit()
        _coin_date_schema_checked = True
    except Exception:
        db.session.rollback()
        raise


@app.route("/add", methods=["GET", "POST"])
def add_coin():

    all_mints = Mint.query.order_by(
        Mint.country,
        Mint.name
    ).all()

    # --- ADD SIMILAR COIN PREFILL ---
    prefill_coin = None
    copy_from = request.args.get("copy_from")

    if request.method == "GET" and copy_from:
        try:
            source_coin = db.session.get(Coin, int(copy_from))
        except (TypeError, ValueError):
            source_coin = None

        if source_coin is not None:
            prefill_coin = {
                "_source_name": source_coin.name or "",
                "_source_year": source_coin.year if source_coin.year is not None else "",
                "name": source_coin.name or "",
                "country": source_coin.country or "",
                "denomination": source_coin.denomination or "",
                "material": source_coin.material or "",
                "mint_id": source_coin.mint_id if source_coin.mint_id is not None else "",
                "numista_type_id": source_coin.numista_type_id if source_coin.numista_type_id is not None else "",
                "obverse_image": source_coin.obverse_image or "",
                "reverse_image": source_coin.reverse_image or "",
                "numista_url": source_coin.numista_url or "",
                "obverse_copyright": source_coin.obverse_copyright or "",
                "obverse_license": source_coin.obverse_license or "",
                "reverse_copyright": source_coin.reverse_copyright or "",
                "reverse_license": source_coin.reverse_license or "",
                "year": "",
                "date_start_year": "",
                "date_end_year": "",
                "date_is_approx": False,
                "mint_mark": "",
                "numista_issue_id": "",
                "variant": "",
                "mintage": "",
                "condition": "",
                "estimated_value": "",
                "purchase_price": "",
                "quantity": "1",
                "date_acquired": "",
                "notes": ""
            }


    if request.method == "POST":

        mint_id = request.form.get("mint_id")

        selected_mint = None

        if mint_id:
            selected_mint = db.session.get(
                Mint,
                int(mint_id)
            )

        mint_name = ""
        mint_location = ""

        if selected_mint:

            mint_name = selected_mint.name

            location_parts = [
                selected_mint.city,
                selected_mint.state,
                selected_mint.country
            ]

            mint_location = ", ".join(
                part
                for part in location_parts
                if part
            )



        # --- NORMALIZED ADD-COIN VALUES ---
        year_raw = request.form.get("year", "").strip()
        try:
            normalized_year = int(year_raw) if year_raw else None
        except (TypeError, ValueError):
            normalized_year = None

        def parse_optional_year(field):
            raw = request.form.get(field, "").strip()
            try:
                return int(raw) if raw else None
            except (TypeError, ValueError):
                return None

        normalized_date_start = parse_optional_year("date_start_year")
        normalized_date_end = parse_optional_year("date_end_year")
        normalized_date_approx = request.form.get("date_is_approx") == "1"

        # A single Year remains the canonical exact date. If a range is supplied,
        # year uses its start so older sorting/timeline features keep working.
        if normalized_date_start is not None or normalized_date_end is not None:
            if normalized_date_start is None:
                normalized_date_start = normalized_date_end
            if normalized_date_end is None:
                normalized_date_end = normalized_date_start
            if normalized_date_start > normalized_date_end:
                normalized_date_start, normalized_date_end = normalized_date_end, normalized_date_start
            normalized_year = normalized_date_start

        quantity_raw = request.form.get("quantity", "").strip()
        try:
            normalized_quantity = int(quantity_raw) if quantity_raw else 1
        except (TypeError, ValueError):
            normalized_quantity = 1

        estimated_value_raw = request.form.get("estimated_value", "").strip()
        try:
            normalized_estimated_value = (
                float(estimated_value_raw)
                if estimated_value_raw
                else None
            )
        except (TypeError, ValueError):
            normalized_estimated_value = None

        purchase_price_raw = request.form.get("purchase_price", "").strip()
        try:
            normalized_purchase_price = (
                float(purchase_price_raw)
                if purchase_price_raw
                else None
            )
        except (TypeError, ValueError):
            normalized_purchase_price = None

        numista_type_raw = request.form.get("numista_type_id", "").strip()
        try:
            normalized_numista_type_id = (
                int(numista_type_raw)
                if numista_type_raw
                else None
            )
        except (TypeError, ValueError):
            normalized_numista_type_id = None

        numista_issue_raw = request.form.get("numista_issue_id", "").strip()
        try:
            normalized_numista_issue_id = (
                int(numista_issue_raw)
                if numista_issue_raw
                else None
            )
        except (TypeError, ValueError):
            normalized_numista_issue_id = None

        mintage_raw = request.form.get("mintage", "").strip()
        try:
            normalized_mintage = int(mintage_raw) if mintage_raw else None
        except (TypeError, ValueError):
            normalized_mintage = None

        coin = Coin(

            name=request.form["name"],

            country=request.form["country"],
            set_country=_canonical_set_country(request.form["country"])[0],
            set_country_review=_canonical_set_country(request.form["country"])[1],

            year=normalized_year,
            date_start_year=normalized_date_start,
            date_end_year=normalized_date_end,
            date_is_approx=normalized_date_approx,


            mint_id=(
                selected_mint.id
                if selected_mint
                else None
            ),

            mint=mint_name,

            mint_mark=request.form[
                "mint_mark"
            ],

            location=mint_location,


            denomination=request.form[
                "denomination"
            ],

            material=request.form[
                "material"
            ],

            condition=request.form[
                "condition"
            ],
            estimated_value=normalized_estimated_value,
            # Private owner-only field.
            purchase_price=normalized_purchase_price,

            quantity=normalized_quantity,

            date_acquired=request.form[
                "date_acquired"
            ],

            notes=request.form[
                "notes"
            ],


            # Numista coin information
            numista_type_id=normalized_numista_type_id,

            numista_issue_id=normalized_numista_issue_id,

                       variant=(
                request.form.get(
                    "variant"
                )
                or None
            ),

            mintage=normalized_mintage,

            # Numista reference images
            obverse_image=(
                request.form.get(
                    "obverse_image"
                )
                or None
            ),

            reverse_image=(
                request.form.get(
                    "reverse_image"
                )
                or None
            ),

            numista_url=(
                request.form.get(
                    "numista_url"
                )
                or None
            ),

            obverse_copyright=(
                request.form.get(
                    "obverse_copyright"
                )
                or None
            ),

            obverse_license=(
                request.form.get(
                    "obverse_license"
                )
                or None
            ),

            reverse_copyright=(
                request.form.get(
                    "reverse_copyright"
                )
                or None
            ),

            reverse_license=(
                request.form.get(
                    "reverse_license"
                )
                or None
            )

        )


        db.session.add(coin)

        # We need the new coin ID before naming its uploaded photos.
        db.session.flush()

        obverse_file = request.files.get(
            "personal_obverse_image"
        )

        reverse_file = request.files.get(
            "personal_reverse_image"
        )

        guided_obverse = (
            request.form.get("guided_obverse") == "1"
        )

        guided_reverse = (
            request.form.get("guided_reverse") == "1"
        )

        if (
            obverse_file
            and obverse_file.filename
        ):
            result = process_coin_photo(
                obverse_file,
                coin.id,
                "obverse",
                guided_capture=guided_obverse
            )

            coin.personal_obverse_image = (
                result["processed_filename"]
            )

        if (
            reverse_file
            and reverse_file.filename
        ):
            result = process_coin_photo(
                reverse_file,
                coin.id,
                "reverse",
                guided_capture=guided_reverse
            )

            coin.personal_reverse_image = (
                result["processed_filename"]
            )

        db.session.commit()

        if request.form.get("save_action") == "add_similar":
            return redirect(
                url_for(
                    "add_coin",
                    copy_from=coin.id
                )
            )



        return redirect(
            url_for(
                "coin_detail",
                coin_id=coin.id
            )
        )


    return render_template(
        "add_coin.html",
        mints=all_mints,
        prefill_coin=prefill_coin
    )
@app.route("/mint/<int:mint_id>")
def mint_detail(mint_id):

    mint = Mint.query.get_or_404(
        mint_id
    )

    return render_template(
        "mint_detail.html",
        mint=mint
    )
@app.route("/coins")
def coins():
    search = request.args.get("search", "").strip()
    country = request.args.get("country", "").strip()
    year = request.args.get("year", "").strip()
    mint = request.args.get("mint", "").strip()
    reference_image = request.args.get("reference_image", "").strip()
    sort_by = request.args.get("sort", "age")
    order = request.args.get("order", "desc")

    complete_collection = Coin.query.all()
    all_coins = list(complete_collection)

    if search:
        needle = search.casefold()
        all_coins = [
            coin for coin in all_coins
            if needle in " ".join([
                coin.name or "",
                coin.country or "",
                coin.mint or "",
                coin.mint_mark or "",
                coin.denomination or "",
                coin.material or "",
            ]).casefold()
        ]

    if country:
        all_coins = [
            coin for coin in all_coins
            if coin.country == country
        ]

    if year:
        try:
            selected_year = int(year)
        except ValueError:
            selected_year = None

        if selected_year is not None:
            all_coins = [
                coin for coin in all_coins
                if (
                    coin.year == selected_year
                    or (
                        coin.date_start_year is not None
                        and coin.date_end_year is not None
                        and coin.date_start_year <= selected_year <= coin.date_end_year
                    )
                )
            ]

    if mint:
        all_coins = [
            coin for coin in all_coins
            if coin.mint == mint
        ]

    # Catalogue/reference image filter. A coin counts as having a
    # reference image when either catalogue side is populated.
    if reference_image == "missing":
        all_coins = [
            coin for coin in all_coins
            if not (coin.obverse_image or coin.reverse_image)
        ]
    elif reference_image == "has":
        all_coins = [
            coin for coin in all_coins
            if coin.obverse_image or coin.reverse_image
        ]

    def coin_sort_value(coin):
        if sort_by == "alphabet":
            return coin.name or ""
        if sort_by == "mintage":
            return (
                coin.mintage
                if coin.mintage is not None
                else float("-inf")
            )
        if sort_by == "value":
            return (
                coin.estimated_value
                if coin.estimated_value is not None
                else float("-inf")
            )
        return (
            coin.year
            if coin.year is not None
            else float("-inf")
        )

    reverse = order != "asc"

    all_coins.sort(
        key=lambda coin: (
            coin_sort_value(coin),
            coin.id,
        ),
        reverse=reverse,
    )

    total_coins = sum(
        coin.quantity or 1
        for coin in complete_collection
    )

    countries = sorted({
        coin.country.strip()
        for coin in complete_collection
        if coin.country and coin.country.strip()
    })

    year_values = set()
    for coin in complete_collection:
        if coin.year is None:
            continue
        try:
            year_values.add(int(str(coin.year).strip()))
        except (TypeError, ValueError):
            continue

    years = sorted(year_values, reverse=True)

    mints = sorted({
        coin.mint.strip()
        for coin in complete_collection
        if coin.mint and coin.mint.strip()
    })

    unique_types = set()

    for coin in complete_collection:
        if coin.numista_type_id:
            unique_types.add(
                ("numista", coin.numista_type_id)
            )
        else:
            unique_types.add((
                "local",
                (coin.name or "").strip().lower(),
                (coin.country or "").strip().lower(),
                (coin.denomination or "").strip().lower(),
            ))

    estimated_value = sum(
        (coin.estimated_value or 0)
        * (coin.quantity or 1)
        for coin in complete_collection
    )

    collection_stats = {
        "total_coins": total_coins,
        "countries": len(countries),
        "unique_types": len(unique_types),
        "estimated_value": estimated_value,
        "displayed_records": len(all_coins),
    }

    return render_template(
        "coins.html",
        coins=all_coins,
        search=search,
        country=country,
        year=year,
        mint=mint,
        reference_image=reference_image,
        sort_by=sort_by,
        order=order,
        filter_countries=countries,
        filter_years=years,
        filter_mints=mints,
        collection_stats=collection_stats,
    )


# --- COLLECTION VALUE DASHBOARD ---
@app.route("/value")
def collection_value():
    all_coins = Coin.query.all()
    all_artifacts = Artifact.query.all()

    total_coin_count = sum((coin.quantity or 1) for coin in all_coins)
    total_artifact_count = len(all_artifacts)

    coin_estimated_value = sum(
        (coin.estimated_value or 0) * (coin.quantity or 1)
        for coin in all_coins
    )
    artifact_estimated_value = sum(
        artifact.estimated_value or 0
        for artifact in all_artifacts
    )
    total_estimated_value = coin_estimated_value + artifact_estimated_value

    coin_purchase_cost = sum(
        (coin.purchase_price or 0) * (coin.quantity or 1)
        for coin in all_coins
    )
    artifact_purchase_cost = sum(
        artifact.purchase_price or 0
        for artifact in all_artifacts
    )
    total_purchase_cost = coin_purchase_cost + artifact_purchase_cost

    value_breakdown = []
    for label, value in [
        ("Coins", coin_estimated_value),
        ("Cabinet", artifact_estimated_value),
    ]:
        percent = (
            round(value / total_estimated_value * 100, 1)
            if total_estimated_value
            else 0
        )
        value_breakdown.append({
            "label": label,
            "value": value,
            "percent": percent,
        })

    top_items = []

    for coin in all_coins:
        if coin.estimated_value is not None:
            top_items.append({
                "name": coin.name,
                "kind": "Coin",
                "detail": str(coin.year) if coin.year is not None else coin.country,
                "value": coin.estimated_value * (coin.quantity or 1),
            })

    for artifact in all_artifacts:
        if artifact.estimated_value is not None:
            top_items.append({
                "name": artifact.name,
                "kind": "Artifact",
                "detail": artifact.year_text or artifact.category,
                "value": artifact.estimated_value,
            })

    top_items = sorted(
        top_items,
        key=lambda item: item["value"],
        reverse=True,
    )[:6]

    artifact_categories = []
    for label in [
        "Paper Money",
        "Books",
        "Medals & Tokens",
        "Antiques",
        "Other Collectibles",
    ]:
        matching = [
            artifact
            for artifact in all_artifacts
            if artifact.category == label
        ]
        artifact_categories.append({
            "label": "Other" if label == "Other Collectibles" else label,
            "count": len(matching),
            "value": sum(
                artifact.estimated_value or 0
                for artifact in matching
            ),
        })

    return render_template(
        "collection_value.html",
        total_estimated_value=total_estimated_value,
        coin_estimated_value=coin_estimated_value,
        artifact_estimated_value=artifact_estimated_value,
        total_purchase_cost=total_purchase_cost,
        total_coin_count=total_coin_count,
        total_artifact_count=total_artifact_count,
        value_breakdown=value_breakdown,
        top_items=top_items,
        artifact_categories=artifact_categories,
    )



# --- MANUAL FEATURED COIN ROUTE ---
@app.route("/coin/<int:coin_id>/feature", methods=["POST"])
def set_featured_coin(coin_id):
    coin = Coin.query.get_or_404(coin_id)

    Coin.query.update(
        {Coin.featured: False},
        synchronize_session=False
    )

    coin.featured = True
    db.session.commit()

    return redirect(
        url_for(
            "coin_detail",
            coin_id=coin.id
        )
    )


# --- COLLECTOR-STYLE PRESET SETS ---

PRESET_SET_DEFINITIONS = {
    "wwii-major-powers": {
        "name": "World War II · Major Powers",
        "description": "One wartime coin from each major participating issuer.",
        "category": "Historical",
        "requirements": [
            {"label": "United States · 1939–1945", "country_terms": "United States|USA", "year_start": 1939, "year_end": 1945},
            {"label": "United Kingdom · 1939–1945", "country_terms": "United Kingdom|Great Britain", "year_start": 1939, "year_end": 1945},
            {"label": "Germany · 1939–1945", "country_terms": "Germany|German Reich", "year_start": 1939, "year_end": 1945},
            {"label": "Italy · 1939–1945", "country_terms": "Italy", "year_start": 1939, "year_end": 1945},
            {"label": "Japan · 1939–1945", "country_terms": "Japan", "year_start": 1939, "year_end": 1945},
            {"label": "Soviet Union · 1939–1945", "country_terms": "Soviet Union|USSR", "year_start": 1939, "year_end": 1945},
            {"label": "France · 1939–1945", "country_terms": "France", "year_start": 1939, "year_end": 1945},
            {"label": "Canada · 1939–1945", "country_terms": "Canada", "year_start": 1939, "year_end": 1945}
        ]
    },
    "us-20th-century-type": {
        "name": "U.S. 20th Century Type Set",
        "description": "One representative example of major U.S. circulating coin designs used during the 1900s.",
        "category": "Type Set",
        "requirements": [
            {"label": "Lincoln Wheat Cent", "country_terms": "United States|USA", "name_terms": "Lincoln|Wheat", "denomination_terms": "Cent|Penny", "year_start": 1909, "year_end": 1958},
            {"label": "Buffalo Nickel", "country_terms": "United States|USA", "name_terms": "Buffalo|Indian Head", "denomination_terms": "Nickel|5 Cents", "year_start": 1913, "year_end": 1938},
            {"label": "Jefferson Nickel", "country_terms": "United States|USA", "name_terms": "Jefferson", "denomination_terms": "Nickel|5 Cents", "year_start": 1938, "year_end": 1999},
            {"label": "Mercury Dime", "country_terms": "United States|USA", "name_terms": "Mercury|Winged Liberty", "denomination_terms": "Dime|10 Cents", "year_start": 1916, "year_end": 1945},
            {"label": "Roosevelt Dime", "country_terms": "United States|USA", "name_terms": "Roosevelt", "denomination_terms": "Dime|10 Cents", "year_start": 1946, "year_end": 1999},
            {"label": "Washington Quarter", "country_terms": "United States|USA", "name_terms": "Washington", "denomination_terms": "Quarter|25 Cents", "year_start": 1932, "year_end": 1999},
            {"label": "Walking Liberty Half Dollar", "country_terms": "United States|USA", "name_terms": "Walking Liberty", "denomination_terms": "Half|50 Cents", "year_start": 1916, "year_end": 1947},
            {"label": "Franklin Half Dollar", "country_terms": "United States|USA", "name_terms": "Franklin", "denomination_terms": "Half|50 Cents", "year_start": 1948, "year_end": 1963},
            {"label": "Kennedy Half Dollar", "country_terms": "United States|USA", "name_terms": "Kennedy", "denomination_terms": "Half|50 Cents", "year_start": 1964, "year_end": 1999},
            {"label": "Peace Dollar", "country_terms": "United States|USA", "name_terms": "Peace", "denomination_terms": "Dollar", "year_start": 1921, "year_end": 1935}
        ]
    },
    "1900s-decade-set": {
        "name": "1900s · One Coin Per Decade",
        "description": "One coin from every decade of the 20th century.",
        "category": "Timeline",
        "requirements": [
            {"label": "1900s", "year_start": 1900, "year_end": 1909},
            {"label": "1910s", "year_start": 1910, "year_end": 1919},
            {"label": "1920s", "year_start": 1920, "year_end": 1929},
            {"label": "1930s", "year_start": 1930, "year_end": 1939},
            {"label": "1940s", "year_start": 1940, "year_end": 1949},
            {"label": "1950s", "year_start": 1950, "year_end": 1959},
            {"label": "1960s", "year_start": 1960, "year_end": 1969},
            {"label": "1970s", "year_start": 1970, "year_end": 1979},
            {"label": "1980s", "year_start": 1980, "year_end": 1989},
            {"label": "1990s", "year_start": 1990, "year_end": 1999}
        ]
    },
    "ancient-roman-sampler": {
        "name": "Ancient Roman Sampler",
        "description": "A compact Roman set with one coin from several broad periods.",
        "category": "Ancient",
        "requirements": [
            {"label": "Roman Republic", "country_terms": "Roman Republic|Rome", "year_start": -509, "year_end": -27},
            {"label": "Early Roman Empire", "country_terms": "Roman Empire|Rome", "year_start": -27, "year_end": 96},
            {"label": "High Empire", "country_terms": "Roman Empire|Rome", "year_start": 96, "year_end": 192},
            {"label": "3rd Century", "country_terms": "Roman Empire|Rome", "year_start": 193, "year_end": 284},
            {"label": "Tetrarchy / Constantinian Era", "country_terms": "Roman Empire|Rome", "year_start": 284, "year_end": 364},
            {"label": "Late Roman Empire", "country_terms": "Roman Empire|Rome", "year_start": 364, "year_end": 476}
        ]
    }
}


def _preset_terms(value):
    return [item.strip().lower() for item in (value or "").split("|") if item.strip()]


def _normalize_issuer(value):
    """Normalize issuer text without allowing accidental substring matches."""
    value = (value or "").casefold().replace("’", "'")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


# Catalog issuer labels that should never be guessed into a single modern
# country. They stay historically accurate and are surfaced for owner review.
AMBIGUOUS_SET_ISSUERS = {
    "artsakh", "east africa", "eastern caribbean states", "roman empire",
    "roman republic", "soviet union", "ussr", "yugoslavia",
    "austria hungary", "ottoman empire", "netherlands east indies",
    "dutch east indies", "malaya and british borneo", "papal states",
}

CANONICAL_SET_COUNTRIES = {
    "usa": "United States",
    "united states of america": "United States",
    "russian federation": "Russia",
    "bahamas the": "Bahamas",
    "federal republic of germany": "Germany",
    "germany federal republic of": "Germany",
    "great britain": "United Kingdom",
    "britain": "United Kingdom",
    "eire": "Ireland",
    "turkiye": "Turkey",
    "swiss confederation": "Switzerland",
    "kingdom of sweden": "Sweden",
    "portuguese republic": "Portugal",
    "spanish state": "Spain",
    "kingdom of egypt": "Egypt",
    "union of south africa": "South Africa",
    "siam": "Thailand",
}


def _issuer_without_catalog_dates(value):
    """Remove only a trailing catalog date qualifier such as (1949-date)."""
    value = (value or "").strip()
    return re.sub(
        r"\s*\(\s*\d{1,4}\s*[-–]\s*(?:date|present|\d{1,4})\s*\)\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()


def _canonical_set_country(issuer):
    """Return (country, needs_review) without changing the catalog issuer."""
    cleaned = _issuer_without_catalog_dates(issuer)
    norm = _normalize_issuer(cleaned)
    if not norm:
        return None, False
    if norm in AMBIGUOUS_SET_ISSUERS:
        return None, True
    return CANONICAL_SET_COUNTRIES.get(norm, cleaned), False


def _coin_set_issuer(coin):
    """Prefer stored canonical country; safely derive it for older rows."""
    if getattr(coin, "set_country", None):
        return coin.set_country
    country, needs_review = _canonical_set_country(coin.country)
    return None if needs_review else country


# Explicit equivalents that are safe for automatic set matching. Ambiguous
# historical/territorial relationships stay manual (Artsakh, East Africa,
# Eastern Caribbean States, etc.).
ISSUER_EQUIVALENTS = {
    "united states": {"united states", "usa", "united states of america"},
    "united kingdom": {"united kingdom", "great britain", "britain"},
    "soviet union": {"soviet union", "ussr"},
    "germany": {"germany", "german reich", "germany 1871 1948", "federal republic of germany"},
    "india": {"india", "british india"},
    "south africa": {"south africa", "union of south africa"},
    "switzerland": {"switzerland", "swiss confederation"},
    "portugal": {"portugal", "portuguese republic"},
    "spain": {"spain", "spanish state"},
    "ireland": {"ireland", "eire"},
    "turkey": {"turkey", "turkiye"},
    "czechoslovakia": {"czechoslovakia", "bohemia and moravia", "protectorate of bohemia and moravia"},
    "hungary": {"hungary", "kingdom of hungary"},
    "slovakia": {"slovakia", "slovak republic"},
    "croatia": {"croatia", "independent state of croatia"},
    "thailand": {"thailand", "siam"},
    "egypt": {"egypt", "kingdom of egypt"},
    "china": {"china", "republic of china"},
}


def _issuer_matches_term(issuer, term):
    issuer_norm = _normalize_issuer(issuer)
    term_norm = _normalize_issuer(term)
    if not issuer_norm or not term_norm:
        return False
    if issuer_norm == term_norm:
        return True

    equivalents = {
        _normalize_issuer(key): {_normalize_issuer(item) for item in values}
        for key, values in ISSUER_EQUIVALENTS.items()
    }
    for canonical, names in equivalents.items():
        group = names | {canonical}
        if term_norm in group and issuer_norm in group:
            return True

    # Numista-style labels often add a date period after an otherwise exact
    # issuer name, e.g. "Germany (1871-1948)". Only accept a numeric suffix;
    # never arbitrary text such as "Prussia" for "Russia".
    if issuer_norm.startswith(term_norm + " "):
        suffix = issuer_norm[len(term_norm) + 1:]
        if suffix and all(part.isdigit() for part in suffix.split()):
            return True

    return False


# Around the World should represent the modern country itself, not any empire,
# federation, colony, or predecessor that once covered some/all of its land.
# These remain available through the owner's manual assignment picker.
WORLD_MANUAL_ONLY_TERMS = {
    "ghana": {"gold coast"},
    "indonesia": {"netherlands east indies", "dutch east indies"},
    "india": {"british india"},
    "malaysia": {"malaya"},
    "turkey": {"ottoman empire"},
    "austria": {"austrian empire", "austria hungary"},
    "czechia": {"czechoslovakia"},
    "russia": {"russian empire", "soviet union", "ussr"},
    "serbia": {"yugoslavia"},
    "slovakia": {"czechoslovakia"},
    "slovenia": {"yugoslavia"},
    "united kingdom": {"england", "scotland"},
    "vatican city": {"papal states"},
    "papua new guinea": {"new guinea"},
}


def _world_term_allowed_automatically(requirement, term):
    coin_set = getattr(requirement, "coin_set", None)
    if not coin_set or not (coin_set.name or "").startswith("Around the World"):
        return True

    label = _normalize_issuer(requirement.label)
    term_norm = _normalize_issuer(term)
    return term_norm not in WORLD_MANUAL_ONLY_TERMS.get(label, set())


def _coin_matches_requirement(coin, requirement):
    try:
        coin_year = int(str(coin.year).strip()) if coin.year is not None else None
    except (TypeError, ValueError):
        coin_year = None

    if requirement.year_start is not None:
        if coin_year is None or coin_year < requirement.year_start:
            return False

    if requirement.year_end is not None:
        if coin_year is None or coin_year > requirement.year_end:
            return False

    country_terms = _preset_terms(requirement.country_terms)
    if country_terms:
        if not any(
            _world_term_allowed_automatically(requirement, term)
            and _issuer_matches_term(_coin_set_issuer(coin), term)
            for term in country_terms
        ):
            return False

    name_terms = _preset_terms(requirement.name_terms)
    if name_terms:
        name = (coin.name or "").lower()
        if not any(term in name for term in name_terms):
            return False

    denomination_terms = _preset_terms(requirement.denomination_terms)
    if denomination_terms:
        denomination = (coin.denomination or "").lower()
        if not any(term in denomination for term in denomination_terms):
            return False

    return True


def _match_set_requirements(coin_set, all_coins=None):
    if all_coins is None:
        all_coins = Coin.query.order_by(Coin.year.asc(), Coin.name.asc()).all()

    SetSlotSelection.__table__.create(bind=db.engine, checkfirst=True)
    overrides = {
        row.slot_id: row
        for row in SetSlotSelection.query.filter_by(
            set_id=coin_set.id,
            slot_type="requirement"
        ).all()
    }
    coin_by_id = {coin.id: coin for coin in all_coins}
    used_ids = set()
    matches = []

    requirements = sorted(
        coin_set.requirements,
        key=lambda requirement: (
            requirement.sort_order or 0,
            requirement.id
        )
    )

    # Reserve manually selected coins first so automatic matching never
    # displaces an owner's choice.
    for row in overrides.values():
        if row.coin_id in coin_by_id:
            used_ids.add(row.coin_id)

    for requirement in requirements:
        selection = overrides.get(requirement.id)
        match = coin_by_id.get(selection.coin_id) if selection else None
        manual = match is not None

        if match is None:
            for coin in all_coins:
                if coin.id in used_ids:
                    continue
                if _coin_matches_requirement(coin, requirement):
                    match = coin
                    used_ids.add(coin.id)
                    break

        matches.append({
            "requirement": requirement,
            "coin": match,
            "manual": manual
        })

    return matches



# --- WWII GOAL SET CONFIG ---

WW2_GOAL_COUNTRIES = [
    {"label":'United States',"terms":'United States|USA|United States of America'},
    {"label":'United Kingdom',"terms":'United Kingdom|Great Britain|Britain'},
    {"label":'Soviet Union',"terms":'Soviet Union|USSR'},
    {"label":'China',"terms":'China|Republic of China'},
    {"label":'France',"terms":'France'},
    {"label":'Germany',"terms":'Germany|German Reich'},
    {"label":'Italy',"terms":'Italy'},
    {"label":'Japan',"terms":'Japan'},
    {"label":'Canada',"terms":'Canada'},
    {"label":'Australia',"terms":'Australia'},
    {"label":'New Zealand',"terms":'New Zealand'},
    {"label":'India',"terms":'India|British India'},
    {"label":'South Africa',"terms":'South Africa|Union of South Africa'},
    {"label":'Iceland',"terms":'Iceland'},
    {"label":'Switzerland',"terms":'Switzerland|Swiss Confederation'},
    {"label":'Sweden',"terms":'Sweden|Kingdom of Sweden'},
    {"label":'Portugal',"terms":'Portugal|Portuguese Republic'},
    {"label":'Spain',"terms":'Spain|Spanish State'},
    {"label":'Ireland',"terms":'Ireland|Éire|Eire'},
    {"label":'Turkey',"terms":'Turkey|Türkiye|Turkiye'},
    {"label":'Philippines',"terms":'Philippines'},
    {"label":'Netherlands',"terms":'Netherlands'},
    {"label":'Belgium',"terms":'Belgium'},
    {"label":'Norway',"terms":'Norway'},
    {"label":'Denmark',"terms":'Denmark'},
    {"label":'Luxembourg',"terms":'Luxembourg'},
    {"label":'Poland',"terms":'Poland'},
    {"label":'Czechoslovakia',"terms":'Czechoslovakia|Bohemia and Moravia|Protectorate of Bohemia and Moravia'},
    {"label":'Yugoslavia',"terms":'Yugoslavia'},
    {"label":'Greece',"terms":'Greece'},
    {"label":'Finland',"terms":'Finland'},
    {"label":'Hungary',"terms":'Hungary|Kingdom of Hungary'},
    {"label":'Romania',"terms":'Romania'},
    {"label":'Bulgaria',"terms":'Bulgaria'},
    {"label":'Slovakia',"terms":'Slovakia|Slovak Republic'},
    {"label":'Croatia',"terms":'Croatia|Independent State of Croatia'},
    {"label":'Thailand',"terms":'Thailand|Siam'},
    {"label":'Brazil',"terms":'Brazil'},
    {"label":'Mexico',"terms":'Mexico'},
    {"label":'Ethiopia',"terms":'Ethiopia'},
    {"label":'Egypt',"terms":'Egypt|Kingdom of Egypt'},
    {"label":'Liberia',"terms":'Liberia'},
    {"label":'Morocco',"terms":'Morocco'},
    {"label":'Tunisia',"terms":'Tunisia'},
    {"label":'Algeria',"terms":'Algeria'},
    {"label":'Libya',"terms":'Libya'},
]

def _ww2_terms(value):
    return [x.strip().lower() for x in (value or '').split('|') if x.strip()]

def _ww2_coin_matches_slot(coin, slot):
    try:
        coin_year = int(str(coin.year).strip()) if coin.year is not None else None
    except (TypeError, ValueError):
        coin_year = None

    if coin_year != slot.year:
        return False

    return any(
        _issuer_matches_term(_coin_set_issuer(coin), term)
        for term in _ww2_terms(slot.country_terms)
    )

def _ww2_slot_matches(coin_set, all_coins=None):
    if all_coins is None:
        all_coins = Coin.query.order_by(Coin.year.asc(), Coin.name.asc()).all()

    SetSlotSelection.__table__.create(bind=db.engine, checkfirst=True)
    overrides = {
        row.slot_id: row
        for row in SetSlotSelection.query.filter_by(
            set_id=coin_set.id,
            slot_type="ww2"
        ).all()
    }
    coin_by_id = {coin.id: coin for coin in all_coins}
    used_ids = {
        row.coin_id for row in overrides.values()
        if row.coin_id in coin_by_id
    }
    matches = []
    slots = sorted(coin_set.ww2_slots, key=lambda s: (s.sort_order or 0, s.country_label.lower(), s.year))
    for slot in slots:
        selection = overrides.get(slot.id)
        match = coin_by_id.get(selection.coin_id) if selection else None
        manual = match is not None
        if match is None:
            for coin in all_coins:
                if coin.id in used_ids:
                    continue
                if _ww2_coin_matches_slot(coin, slot):
                    match = coin
                    used_ids.add(coin.id)
                    break
        matches.append({"slot":slot,"coin":match,"manual":manual})
    return matches

# --- SETS FEATURE ROUTES ---

@app.route("/sets")
def sets():
    coin_sets = (
        CoinSet.query
        .options(
            joinedload(CoinSet.ww2_slots),
            joinedload(CoinSet.requirements),
            joinedload(CoinSet.coins),
        )
        .order_by(CoinSet.name.asc())
        .all()
    )

    all_coins = Coin.query.order_by(
        Coin.year.asc(),
        Coin.name.asc(),
    ).all()

    set_progress = {}

    for coin_set in coin_sets:
        completed = 0
        total = 0

        if coin_set.ww2_slots:
            matches = _ww2_slot_matches(coin_set, all_coins)
            total = len(matches)
            completed = sum(
                1 for item in matches
                if item["coin"] is not None
            )
        elif coin_set.requirements:
            matches = _match_set_requirements(coin_set, all_coins)
            total = len(matches)
            completed = sum(
                1 for item in matches
                if item["coin"] is not None
            )
        else:
            total = len(coin_set.coins)
            completed = total

        set_progress[coin_set.id] = {
            "completed": completed,
            "total": total,
            "percent": round((completed / total * 100) if total else 0),
        }

    return render_template(
        "sets.html",
        coin_sets=coin_sets,
        set_progress=set_progress,
    )


@app.route("/sets/create", methods=["GET", "POST"])
def create_set():
    error = None

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        description = request.form.get("description", "").strip() or None

        if not name:
            error = "Please enter a set name."
        elif CoinSet.query.filter(db.func.lower(CoinSet.name) == name.lower()).first():
            error = "A set with that name already exists."
        else:
            coin_set = CoinSet(name=name, description=description)
            db.session.add(coin_set)
            db.session.commit()
            return redirect(url_for("edit_set", set_id=coin_set.id))

    return render_template("create_set.html", error=error)



@app.route("/sets/presets")
def preset_sets():
    presets = []

    for slug, definition in PRESET_SET_DEFINITIONS.items():
        presets.append({
            "slug": slug,
            "name": definition["name"],
            "description": definition["description"],
            "category": definition.get("category", "Preset"),
            "slot_count": len(definition["requirements"])
        })

    return render_template("preset_sets.html", presets=presets)


@app.route("/sets/presets/<slug>/start", methods=["POST"])
def start_preset_set(slug):
    definition = PRESET_SET_DEFINITIONS.get(slug)

    if not definition:
        abort(404)

    base_name = definition["name"]
    name = base_name
    counter = 2

    while CoinSet.query.filter(
        db.func.lower(CoinSet.name) == name.lower()
    ).first():
        name = f"{base_name} ({counter})"
        counter += 1

    coin_set = CoinSet(
        name=name,
        description=definition["description"]
    )

    db.session.add(coin_set)
    db.session.flush()

    for index, item in enumerate(definition["requirements"], start=1):
        db.session.add(
            CoinSetRequirement(
                set_id=coin_set.id,
                label=item["label"],
                description=item.get("description"),
                country_terms=item.get("country_terms"),
                name_terms=item.get("name_terms"),
                denomination_terms=item.get("denomination_terms"),
                year_start=item.get("year_start"),
                year_end=item.get("year_end"),
                sort_order=index
            )
        )

    db.session.commit()
    return redirect(url_for("set_detail", set_id=coin_set.id))





@app.route("/sets/goal/ww2/create", methods=["POST"])
def create_ww2_goal_set():
    base_name = "World War II · 1939–1945"
    name = base_name
    counter = 2
    while CoinSet.query.filter(db.func.lower(CoinSet.name) == name.lower()).first():
        name = f"{base_name} ({counter})"
        counter += 1
    coin_set = CoinSet(name=name, description="One coin from each year of World War II, organized by country.")
    db.session.add(coin_set)
    db.session.flush()
    order = 0
    for country in WW2_GOAL_COUNTRIES:
        for year in range(1939, 1946):
            order += 1
            db.session.add(WW2GoalSlot(set_id=coin_set.id, country_label=country["label"], year=year, country_terms=country["terms"], sort_order=order))
    db.session.commit()
    return redirect(url_for("set_detail", set_id=coin_set.id))


# --- WWI INTERACTIVE MAP ROUTE RESTORED ---
@app.route("/sets/ww1")
def ww1_set():
    coin_set = CoinSet.query.filter(
        db.func.lower(CoinSet.name)
        == "world war i · 1914–1918".lower()
    ).first()

    if coin_set is None:
        return redirect(url_for("sets"))

    matches = _match_set_requirements(coin_set)
    completed = sum(
        1
        for item in matches
        if item.get("coin") is not None
    )

    all_coins = Coin.query.order_by(Coin.year.asc(), Coin.name.asc()).all()

    return render_template(
        "ww1_set.html",
        coin_set=coin_set,
        matches=matches,
        completed=completed,
        total=len(matches),
        all_coins=all_coins,
    )




# ============================================================
# THE HUMAN STORY · HISTORY THROUGH COINS
# ============================================================
@app.route("/human-story")
def human_story():
    from human_story_data import build_human_story

    coins = Coin.query.order_by(
        Coin.year.asc(),
        Coin.id.asc()
    ).all()

    # Create the small override table automatically on existing databases.
    HumanStorySelection.__table__.create(bind=db.engine, checkfirst=True)
    selections = {
        row.moment_key: row.coin_id
        for row in HumanStorySelection.query.all()
    }

    eras, completed, total, percent = build_human_story(coins, selections)

    return render_template(
        "human_story.html",
        eras=eras,
        completed=completed,
        total=total,
        percent=percent,
        all_story_coins=coins,
    )


@app.route("/human-story/choose-coin", methods=["POST"])
def human_story_choose_coin():
    if not session.get("is_owner"):
        abort(403)

    moment_key = (request.form.get("moment_key") or "").strip()
    coin_id = request.form.get("coin_id", type=int)
    if not moment_key:
        abort(400)

    HumanStorySelection.__table__.create(bind=db.engine, checkfirst=True)
    selection = HumanStorySelection.query.filter_by(moment_key=moment_key).first()

    if coin_id:
        coin = db.session.get(Coin, coin_id)
        if coin is None:
            abort(404)
        if selection is None:
            selection = HumanStorySelection(moment_key=moment_key, coin_id=coin.id)
            db.session.add(selection)
        else:
            selection.coin_id = coin.id
    elif selection is not None:
        db.session.delete(selection)

    db.session.commit()
    return redirect(url_for("human_story"))


@app.route("/sets/<int:set_id>/assign-slot", methods=["POST"])
def assign_set_slot_coin(set_id):
    coin_set = CoinSet.query.get_or_404(set_id)
    slot_type = (request.form.get("slot_type") or "").strip()
    slot_id = request.form.get("slot_id", type=int)
    coin_id = request.form.get("coin_id", type=int)
    next_url = request.form.get("next") or url_for("set_detail", set_id=set_id)

    if slot_type not in {"requirement", "ww2"} or not slot_id:
        abort(400)

    if slot_type == "requirement":
        slot = CoinSetRequirement.query.filter_by(id=slot_id, set_id=set_id).first_or_404()
    else:
        slot = WW2GoalSlot.query.filter_by(id=slot_id, set_id=set_id).first_or_404()

    SetSlotSelection.__table__.create(bind=db.engine, checkfirst=True)
    selection = SetSlotSelection.query.filter_by(
        set_id=set_id,
        slot_type=slot_type,
        slot_id=slot.id,
    ).first()

    if coin_id:
        coin = db.session.get(Coin, coin_id)
        if coin is None:
            abort(404)
        if selection is None:
            selection = SetSlotSelection(
                set_id=set_id,
                slot_type=slot_type,
                slot_id=slot.id,
                coin_id=coin.id,
            )
            db.session.add(selection)
        else:
            selection.coin_id = coin.id
    elif selection is not None:
        db.session.delete(selection)

    db.session.commit()
    if not is_safe_local_path(next_url):
        next_url = url_for("set_detail", set_id=set_id)
    return redirect(next_url)


@app.route("/sets/<int:set_id>")
def set_detail(set_id):
    coin_set = (
        CoinSet.query
        .options(
            joinedload(CoinSet.coins),
            joinedload(CoinSet.ww2_slots),
            joinedload(CoinSet.requirements),
        )
        .filter(CoinSet.id == set_id)
        .first_or_404()
    )

    coins = sorted(
        coin_set.coins,
        key=lambda coin: (
            coin.year is None,
            coin.year if coin.year is not None else 999999,
            (coin.name or "").lower(),
        ),
    )

    all_coins = Coin.query.order_by(
        Coin.year.asc(),
        Coin.name.asc(),
    ).all()

    ww2_matches = (
        _ww2_slot_matches(coin_set, all_coins)
        if coin_set.ww2_slots
        else []
    )

    ww2_complete = sum(
        1 for item in ww2_matches
        if item["coin"] is not None
    )

    grouped_ww2 = {}
    for item in ww2_matches:
        grouped_ww2.setdefault(
            item["slot"].country_label,
            [],
        ).append(item)

    requirement_matches = (
        _match_set_requirements(coin_set, all_coins)
        if coin_set.requirements
        else []
    )

    completed_requirements = sum(
        1 for item in requirement_matches
        if item["coin"] is not None
    )

    return render_template(
        "set_detail.html",
        coin_set=coin_set,
        coins=coins,
        ww2_matches=ww2_matches,
        grouped_ww2=grouped_ww2,
        ww2_complete=ww2_complete,
        requirement_matches=requirement_matches,
        completed_requirements=completed_requirements,
        all_coins=all_coins,
    )

@app.route("/sets/<int:set_id>/edit", methods=["GET", "POST"])
def edit_set(set_id):
    coin_set = CoinSet.query.get_or_404(set_id)
    error = None

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        description = request.form.get("description", "").strip() or None

        existing = (
            CoinSet.query
            .filter(
                db.func.lower(CoinSet.name) == name.lower(),
                CoinSet.id != coin_set.id
            )
            .first()
        )

        if not name:
            error = "Please enter a set name."
        elif existing:
            error = "A set with that name already exists."
        else:
            selected_ids = []

            for value in request.form.getlist("coin_ids"):
                try:
                    selected_ids.append(int(value))
                except (TypeError, ValueError):
                    pass

            selected_coins = []

            if selected_ids:
                selected_coins = Coin.query.filter(
                    Coin.id.in_(selected_ids)
                ).all()

            coin_set.name = name
            coin_set.description = description
            coin_set.coins = selected_coins
            db.session.commit()

            return redirect(url_for("set_detail", set_id=coin_set.id))

    all_coins = Coin.query.order_by(
        Coin.year.asc(),
        Coin.name.asc()
    ).all()

    selected_ids = {coin.id for coin in coin_set.coins}

    return render_template(
        "edit_set.html",
        coin_set=coin_set,
        all_coins=all_coins,
        selected_ids=selected_ids,
        error=error
    )


@app.route("/sets/<int:set_id>/delete", methods=["POST"])
def delete_set(set_id):
    coin_set = CoinSet.query.get_or_404(set_id)
    db.session.delete(coin_set)
    db.session.commit()
    return redirect(url_for("sets"))



# --- CABINET OF CURIOSITIES ROUTES ---
ARTIFACT_CATEGORIES = [
    "Paper Money",
    "Books",
    "Medals & Tokens",
    "Antiques",
    "Other Collectibles",
]
ARTIFACT_UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads", "artifacts")
os.makedirs(ARTIFACT_UPLOAD_FOLDER, exist_ok=True)


def _artifact_filename_from_photo_record(photo_record):
    for attr in ("image_filename", "filename", "file_name"):
        value = getattr(photo_record, attr, None)
        if value:
            return value, attr
    return None, None


def get_artifact_photo_target(artifact, photo_id=None):
    """
    Returns:
        (source_path, image_filename, photo_record)

    Supports either:
    - a simple artifact.image_filename
    - or a related ArtifactPhoto / photos gallery setup
    """
    photo_record = None
    image_filename = None

    # If you have a separate ArtifactPhoto model
    PhotoModel = globals().get("ArtifactPhoto")

    if photo_id is not None and PhotoModel is not None:
        photo_record = PhotoModel.query.filter_by(
            id=photo_id,
            artifact_id=artifact.id
        ).first_or_404()

        image_filename, _ = _artifact_filename_from_photo_record(photo_record)

    # Try common relationship names if no explicit photo_id worked
    if not image_filename:
        for rel_name in ("photos", "artifact_photos", "gallery_photos", "images"):
            rel = getattr(artifact, rel_name, None)
            if rel:
                chosen = None

                if photo_id is not None:
                    for item in rel:
                        if getattr(item, "id", None) == photo_id:
                            chosen = item
                            break
                else:
                    try:
                        chosen = rel[0] if len(rel) > 0 else None
                    except TypeError:
                        chosen = None

                if chosen is not None:
                    photo_record = chosen
                    image_filename, _ = _artifact_filename_from_photo_record(chosen)
                    if image_filename:
                        break

    # Fallback to single-image artifact
    if not image_filename:
        image_filename = getattr(artifact, "image_filename", None)

    if not image_filename:
        abort(404)

    source_path = Path(ARTIFACT_UPLOAD_FOLDER) / image_filename

    if not source_path.exists():
        r2_download_file(
            source_path,
            f"uploads/artifacts/{image_filename}"
        )

    if not source_path.exists():
        abort(404)

    return source_path, image_filename, photo_record


def save_artifact_edited_photo(source_path, crop_x, crop_y, crop_width, crop_height, guide_shape):
    _load_image_tools()
    image = Image.open(source_path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    width, height = image.size

    crop_x = max(0, min(crop_x, width - 1))
    crop_y = max(0, min(crop_y, height - 1))
    crop_width = max(50, min(crop_width, width - crop_x))
    crop_height = max(50, min(crop_height, height - crop_y))

    if guide_shape in {"square", "circle"}:
        size = min(crop_width, crop_height, width - crop_x, height - crop_y)
        crop_width = crop_height = size

    crop_box = (
        int(crop_x), int(crop_y),
        int(crop_x + crop_width), int(crop_y + crop_height),
    )
    cropped = image.crop(crop_box)

    if guide_shape == "rectangle":
        max_dimension = 1600
        scale = min(max_dimension / cropped.width, max_dimension / cropped.height, 1)
        output = cropped.resize(
            (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))),
            Image.LANCZOS,
        ) if scale < 1 else cropped
    else:
        cropped = cropped.resize((1200, 1200), Image.LANCZOS)
        if guide_shape == "circle":
            output = Image.new("RGB", (1200, 1200), "black")
            mask = Image.new("L", (1200, 1200), 0)
            ImageDraw.Draw(mask).ellipse((0, 0, 1199, 1199), fill=255)
            output.paste(cropped, (0, 0), mask)
        else:
            output = cropped

    output.save(source_path, quality=95)
    r2_upload_file(source_path, f"uploads/artifacts/{source_path.name}")


@app.route("/artifacts/<int:artifact_id>/photos/edit", methods=["GET", "POST"], defaults={"photo_id": None})
@app.route("/artifacts/<int:artifact_id>/photos/<int:photo_id>/edit", methods=["GET", "POST"])
def edit_artifact_photo(artifact_id, photo_id):
    artifact = Artifact.query.get_or_404(artifact_id)

    source_path, image_filename, photo_record = get_artifact_photo_target(
        artifact,
        photo_id=photo_id
    )

    if request.method == "POST":
        guide_shape = request.form.get("guide_shape", "rectangle").strip().lower()
        if guide_shape not in {"rectangle", "circle", "square"}:
            guide_shape = "rectangle"

        crop_x = float(request.form.get("crop_x", 0))
        crop_y = float(request.form.get("crop_y", 0))
        crop_width = float(request.form.get("crop_width", request.form.get("crop_size", 500)))
        crop_height = float(request.form.get("crop_height", request.form.get("crop_size", 500)))

        save_artifact_edited_photo(
            source_path=source_path,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_width=crop_width,
            crop_height=crop_height,
            guide_shape=guide_shape,
        )

        return redirect(url_for("artifact_detail", artifact_id=artifact.id))

    image_url = url_for(
        "static",
        filename=f"uploads/artifacts/{image_filename}"
    )

    return render_template(
        "artifact_photo_editor.html",
        artifact=artifact,
        photo_id=photo_id,
        image_url=image_url,
        source_filename=source_path.name,
    )
ARTIFACT_UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads", "artifacts")
os.makedirs(ARTIFACT_UPLOAD_FOLDER, exist_ok=True)

def save_artifact_image(file_storage, artifact_id, photo_index=None):
    if not file_storage or not file_storage.filename:
        return None
    original = secure_filename(file_storage.filename)
    suffix = Path(original).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"

    unique = uuid.uuid4().hex[:10]
    if photo_index is None:
        filename = f"artifact_{artifact_id}_{unique}{suffix}"
    else:
        filename = f"artifact_{artifact_id}_{photo_index}_{unique}{suffix}"

    local_path = Path(ARTIFACT_UPLOAD_FOLDER) / filename
    file_storage.save(local_path)

    r2_upload_file(
        local_path,
        f"uploads/artifacts/{filename}"
    )

    return filename


def save_artifact_images(file_storages, artifact):
    saved = []
    existing_count = ArtifactPhoto.query.filter_by(artifact_id=artifact.id).count()
    for offset, file_storage in enumerate(file_storages, start=1):
        if not file_storage or not file_storage.filename:
            continue
        sort_order = existing_count + offset
        filename = save_artifact_image(file_storage, artifact.id, sort_order)
        if not filename:
            continue
        photo = ArtifactPhoto(
            artifact_id=artifact.id,
            filename=filename,
            sort_order=sort_order,
        )
        db.session.add(photo)
        saved.append(filename)

    if saved and not artifact.image_filename:
        artifact.image_filename = saved[0]

    return saved

@app.route("/artifacts")
def artifacts():
    category = request.args.get("category", "").strip()
    search = request.args.get("search", "").strip()

    all_items = Artifact.query.order_by(Artifact.id.desc()).all()
    items = all_items

    if category and category in ARTIFACT_CATEGORIES:
        items = [
            artifact for artifact in items
            if artifact.category == category
        ]

    if search:
        needle = search.casefold()
        items = [
            artifact for artifact in items
            if needle in " ".join([
                artifact.name or "",
                artifact.country or "",
                artifact.year_text or "",
                artifact.description or "",
            ]).casefold()
        ]

    stats = {
        "total": len(all_items),
        "categories": len({
            artifact.category
            for artifact in all_items
            if artifact.category
        }),
        "estimated_value": sum(
            artifact.estimated_value or 0
            for artifact in all_items
        ),
        "paper_money": sum(
            1 for artifact in all_items
            if artifact.category == "Paper Money"
        ),
        "books": sum(
            1 for artifact in all_items
            if artifact.category == "Books"
        ),
        "medals": sum(
            1 for artifact in all_items
            if artifact.category == "Medals & Tokens"
        ),
        "antiques": sum(
            1 for artifact in all_items
            if artifact.category == "Antiques"
        ),
        "other": sum(
            1 for artifact in all_items
            if artifact.category == "Other Collectibles"
        ),
    }

    return render_template(
        "artifacts.html",
        artifacts=items,
        categories=ARTIFACT_CATEGORIES,
        selected_category=category,
        search=search,
        stats=stats,
    )

@app.route("/artifacts/add", methods=["GET", "POST"])
def add_artifact():
    if request.method == "POST":
        artifact = Artifact(
            name=request.form.get("name", "").strip(),
            category=request.form.get("category", "").strip(),
            year_text=request.form.get("year_text", "").strip() or None,
            country=request.form.get("country", "").strip() or None,
            description=request.form.get("description", "").strip() or None,
            condition=request.form.get("condition", "").strip() or None,
            estimated_value=float(request.form["estimated_value"]) if request.form.get("estimated_value") else None,
            purchase_price=float(request.form["purchase_price"]) if request.form.get("purchase_price") else None,
            date_acquired=request.form.get("date_acquired", "").strip() or None,
            source=request.form.get("source", "").strip() or None,
            notes=request.form.get("notes", "").strip() or None,
        )
        if not artifact.name:
            return render_template("add_artifact.html", categories=ARTIFACT_CATEGORIES, error="Name is required.")
        if artifact.category not in ARTIFACT_CATEGORIES:
            return render_template("add_artifact.html", categories=ARTIFACT_CATEGORIES, error="Choose a valid category.")
        db.session.add(artifact)
        db.session.flush()
        uploaded_images = request.files.getlist("images")
        if not any(image and image.filename for image in uploaded_images):
            legacy_image = request.files.get("image")
            uploaded_images = [legacy_image] if legacy_image and legacy_image.filename else []
        save_artifact_images(uploaded_images, artifact)
        db.session.commit()
        return redirect(url_for("artifact_detail", artifact_id=artifact.id))
    return render_template("add_artifact.html", categories=ARTIFACT_CATEGORIES)

@app.route("/artifacts/<int:artifact_id>")
def artifact_detail(artifact_id):
    return render_template("artifact_detail.html", artifact=Artifact.query.get_or_404(artifact_id))

@app.route("/artifacts/<int:artifact_id>/edit", methods=["GET", "POST"])
def edit_artifact(artifact_id):
    artifact = Artifact.query.get_or_404(artifact_id)
    if request.method == "POST":
        artifact.name = request.form.get("name", "").strip()
        artifact.category = request.form.get("category", "").strip()
        artifact.year_text = request.form.get("year_text", "").strip() or None
        artifact.country = request.form.get("country", "").strip() or None
        artifact.description = request.form.get("description", "").strip() or None
        artifact.condition = request.form.get("condition", "").strip() or None
        artifact.estimated_value = float(request.form["estimated_value"]) if request.form.get("estimated_value") else None
        artifact.purchase_price = float(request.form["purchase_price"]) if request.form.get("purchase_price") else None
        artifact.date_acquired = request.form.get("date_acquired", "").strip() or None
        artifact.source = request.form.get("source", "").strip() or None
        artifact.notes = request.form.get("notes", "").strip() or None
        uploaded_images = request.files.getlist("images")
        if not any(image and image.filename for image in uploaded_images):
            legacy_image = request.files.get("image")
            uploaded_images = [legacy_image] if legacy_image and legacy_image.filename else []
        save_artifact_images(uploaded_images, artifact)
        db.session.commit()
        return redirect(url_for("artifact_detail", artifact_id=artifact.id))
    return render_template("edit_artifact.html", artifact=artifact, categories=ARTIFACT_CATEGORIES)

@app.route("/artifacts/<int:artifact_id>/photos/<int:photo_id>/remove", methods=["POST"])
def remove_artifact_photo(artifact_id, photo_id):
    artifact = Artifact.query.get_or_404(artifact_id)
    photo = ArtifactPhoto.query.filter_by(id=photo_id, artifact_id=artifact.id).first_or_404()
    filename = photo.filename

    local_path = Path(ARTIFACT_UPLOAD_FOLDER) / filename
    if local_path.is_file():
        local_path.unlink()
    r2_delete_key(f"uploads/artifacts/{filename}")

    db.session.delete(photo)
    db.session.flush()

    if artifact.image_filename == filename:
        remaining = ArtifactPhoto.query.filter_by(artifact_id=artifact.id).order_by(
            ArtifactPhoto.sort_order.asc(), ArtifactPhoto.id.asc()
        ).first()
        artifact.image_filename = remaining.filename if remaining else None

    db.session.commit()
    return redirect(url_for("artifact_detail", artifact_id=artifact.id))


@app.route("/artifacts/<int:artifact_id>/delete", methods=["POST"])
def delete_artifact(artifact_id):
    artifact = Artifact.query.get_or_404(artifact_id)
    filenames = {
        photo.filename
        for photo in artifact.photos
        if photo.filename
    }
    if artifact.image_filename:
        filenames.add(artifact.image_filename)

    for filename in filenames:
        path = os.path.join(ARTIFACT_UPLOAD_FOLDER, filename)
        if os.path.isfile(path):
            os.remove(path)
        r2_delete_key(f"uploads/artifacts/{filename}")

    db.session.delete(artifact)
    db.session.commit()
    return redirect(url_for("artifacts"))


@app.route("/mints")
def mints():
    all_mints = Mint.query.order_by(Mint.country, Mint.name).all()
    return render_template("mints.html", mints=all_mints)


@app.route("/mints/add", methods=["GET", "POST"])
def add_mint():
    if request.method == "POST":
        mint = Mint(
            name=request.form["name"],
            city=request.form["city"],
            state=request.form["state"],
            country=request.form["country"],
            latitude=request.form["latitude"],
            longitude=request.form["longitude"]
        )

        db.session.add(mint)
        db.session.commit()

        return redirect(url_for("mints"))

    return render_template("add_mint.html")
@app.route("/globe")
def globe():
    all_coins = Coin.query.order_by(Coin.year.asc(), Coin.id.asc()).all()

    # Turso is remote, so never lazy-load mint.coins one mint at a time.
    # Build every mint's collection record count in one grouped query.
    mint_coin_counts = dict(
        db.session.query(
            Coin.mint_id,
            db.func.count(Coin.id),
        )
        .filter(Coin.mint_id.isnot(None))
        .group_by(Coin.mint_id)
        .all()
    )

    # Mints without coordinates cannot be displayed on either map view.
    all_mints = (
        Mint.query
        .filter(
            Mint.latitude.isnot(None),
            Mint.longitude.isnot(None),
        )
        .all()
    )

    current_year = datetime.now().year

    # Historical geography now comes from the HistoricalEntity table.
    # Coin.country is never rewritten; this table only controls map navigation.
    historical_entities = HistoricalEntity.query.order_by(
        HistoricalEntity.name.asc()
    ).all()

    historical_entity_lookup = {}

    for entity in historical_entities:
        map_names = [
            value.strip()
            for value in (entity.modern_map_names or "").split("|")
            if value.strip()
        ]

        aliases = [
            value.strip()
            for value in (entity.aliases or "").split("|")
            if value.strip()
        ]

        entity_payload = {
            "id": entity.id,
            "name": entity.name,
            "entity_type": entity.entity_type,
            "start_year": entity.start_year,
            "end_year": entity.end_year,
            "map_names": map_names or [entity.name],

            # Send the aliases to the Geography page too.
            # This lets a historical OHM polygon match coins even when
            # Coin.country uses a different but equivalent issuer label.
            "aliases": aliases,

            "notes": entity.notes,
        }

        historical_entity_lookup[
            entity.name.strip().lower()
        ] = entity_payload

        for alias in aliases:
            historical_entity_lookup[
                alias.lower()
            ] = entity_payload


    # Extra issuer -> historical-map label aliases discovered from the
    # ACTUAL GeoJSON snapshots installed in static/historical_maps/geojson.
    #
    # This does not rename Coin.country and does not change the database.
    # It only gives the Geography page extra labels it may use when matching
    # a coin issuer to a historical boundary polygon.
    issuer_historical_map_aliases = {
        # Ancient collection issuers. These aliases are names that actually
        # appear in the installed Cliopatria snapshots.
        "Ancient Greece": [
            "Greek City-States",
        ],
        "Lysias": [
            "Roman Empire",
        ],
        "Rome": [
            "Roman Empire",
        ],
        "Roman Empire": [
            "Roman Empire",
        ],

        "Hungary": [
            "Kingdom of Hungary",
        ],
        "Iceland": [
            "Kingdom of Iceland",
        ],
        "United Kingdom": [
            "United Kingdom of Great Britain and Ireland",
        ],
        "Yugoslavia": [
            "FPR of Yugoslavia",
        ],
        "USSR": [
            "Soviet Union",
        ],
        "Germany (1871-1948)": [
            "German Reich",
            "Germany",
        ],
        "Switzerland (1848-date)": [
            "Switzerland",
        ],
        "Frankfurt, Free imperial city of": [
            "Frankfur",
            "Frankfurt",
            "Free City of Frankfurt",
        ],
        "Germany, Federal Republic of": [
            "West Germany",
            "Federal Republic of Germany",
            "Germany",
        ],
        "Hong Kong": [
            "British Hong Kong",
            "Hong Kong",
        ],
        "Japan": [
            "Empire of Japan (1931-1945)",
            "Japan (1953-1968)",
            "Japan",
        ],
        "Netherlands": [
            "Kingdom of the Netherlands",
            "Netherlands",
        ],
        "Aruba": [
            "Aruba",
        ],
    }

    grouped = {}
    for coin in all_coins:
        issuer = (coin.country or "").strip() or "Unknown issuer"
        historical_entity = historical_entity_lookup.get(
            issuer.lower()
        )

        group = grouped.setdefault(issuer, {
            "name": issuer,
            "map_names": list(dict.fromkeys(
                (
                    historical_entity["map_names"]
                    if historical_entity
                    else [issuer]
                )
                + issuer_historical_map_aliases.get(
                    issuer,
                    []
                )
            )),
            "historical_entity": historical_entity,
            "coin_count": 0,
            "record_count": 0,
            "estimated_value": 0.0,
            "years": [],
            "coins": []
        })
        qty = coin.quantity or 1
        group["coin_count"] += qty
        group["record_count"] += 1
        group["estimated_value"] += (coin.estimated_value or 0.0) * qty
        if coin.year is not None:
            group["years"].append(coin.year)
        group["coins"].append({
            "id": coin.id, "name": coin.name, "year": coin.year,
            "denomination": coin.denomination, "quantity": qty,
            "mint": coin.mint, "mint_mark": coin.mint_mark,
            "personal_obverse_image": coin.personal_obverse_image,
            "personal_reverse_image": coin.personal_reverse_image,
            "obverse_image": coin.obverse_image, "reverse_image": coin.reverse_image
        })

    country_data = []
    for group in grouped.values():
        years = group.pop("years")
        group["earliest_year"] = min(years) if years else None
        group["latest_year"] = max(years) if years else None
        country_data.append(group)
    country_data.sort(key=lambda item: item["name"].lower())

    mint_data = []
    historical_years = []
    for mint in all_mints:
        start_year = mint.start_year
        raw_end_year = mint.end_year
        if start_year is not None and start_year > current_year:
            start_year = None
        is_present = bool(raw_end_year is not None and raw_end_year >= current_year)
        end_year = current_year if is_present else raw_end_year
        if start_year is not None:
            historical_years.append(start_year)
        mint_data.append({
            "id": mint.id, "name": mint.name, "city": mint.city,
            "state": mint.state, "country": mint.country,
            "lat": mint.latitude, "lng": mint.longitude,
            "start_year": start_year, "end_year": end_year,
            "is_present": is_present,
            "coin_count": int(mint_coin_counts.get(mint.id, 0)),
        })

    return render_template(
        "globe.html", country_data=country_data, mint_data=mint_data,
        timeline_min=min(historical_years) if historical_years else -500,
        timeline_max=current_year
    )


@app.route("/coin/<int:coin_id>")
def coin_detail(coin_id):
    coin = Coin.query.get_or_404(coin_id)
    return render_template("coin_detail.html", coin=coin)
def save_reference_image_upload(file_storage, coin_id, side):
    """Save a user-supplied catalogue reference image locally and to R2."""
    _load_image_tools()
    file_storage.stream.seek(0)
    image = ImageOps.exif_transpose(Image.open(file_storage.stream)).convert("RGB")
    image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)

    filename = f"coin_{coin_id}_reference_{side}.jpg"
    path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(path, "JPEG", quality=92, optimize=True)
    r2_upload_file(path, f"uploads/coins/{filename}")

    # Store the same static URL shape already used for uploaded coin media.
    return url_for("static", filename=f"uploads/coins/{filename}")


@app.route("/coin/<int:coin_id>/reference-images", methods=["POST"])
def update_reference_images(coin_id):
    coin = Coin.query.get_or_404(coin_id)

    obverse_file = request.files.get("reference_obverse_image")
    reverse_file = request.files.get("reference_reverse_image")

    if obverse_file and obverse_file.filename:
        coin.obverse_image = save_reference_image_upload(obverse_file, coin.id, "obverse")

    if reverse_file and reverse_file.filename:
        coin.reverse_image = save_reference_image_upload(reverse_file, coin.id, "reverse")

    coin.numista_url = (request.form.get("numista_url") or "").strip() or coin.numista_url
    db.session.commit()
    flash("Catalogue reference images updated.", "success")

    next_url = (request.form.get("next") or "").strip()
    if is_safe_local_path(next_url):
        return redirect(next_url)
    return redirect(url_for("coin_detail", coin_id=coin.id))


@app.route("/coin/<int:coin_id>/reference-images/<side>/remove", methods=["POST"])
def remove_reference_image(coin_id, side):
    coin = Coin.query.get_or_404(coin_id)
    if side not in {"obverse", "reverse"}:
        abort(404)

    current = coin.obverse_image if side == "obverse" else coin.reverse_image
    if current and current.startswith("/static/uploads/coins/"):
        filename = current.rsplit("/", 1)[-1]
        local_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
        except OSError:
            pass
        r2_delete_key(f"uploads/coins/{filename}")

    if side == "obverse":
        coin.obverse_image = None
    else:
        coin.reverse_image = None

    db.session.commit()
    flash(f"{side.title()} reference image removed.", "success")
    return redirect(url_for("coin_detail", coin_id=coin.id) + "#catalogue-reference")


@app.route(
    "/coin/<int:coin_id>/edit",
    methods=["GET", "POST"]
)
def edit_coin(coin_id):

    coin = Coin.query.get_or_404(
        coin_id
    )

    all_mints = Mint.query.order_by(
        Mint.country,
        Mint.name
    ).all()


    if request.method == "POST":

        # ==============================================
        # MINT
        # ==============================================

        mint_id = request.form.get(
            "mint_id"
        )

        selected_mint = None

        if mint_id:

            selected_mint = db.session.get(
                Mint,
                int(mint_id)
            )


        mint_name = ""
        mint_location = ""

        if selected_mint:

            mint_name = selected_mint.name

            location_parts = [
                selected_mint.city,
                selected_mint.state,
                selected_mint.country
            ]

            mint_location = ", ".join(
                part
                for part in location_parts
                if part
            )


        # ==============================================
        # BASIC COIN INFORMATION
        # ==============================================

        coin.name = request.form.get(
            "name",
            coin.name
        )

        coin.country = request.form.get(
            "country",
            coin.country
        )
        coin.set_country, coin.set_country_review = _canonical_set_country(
            coin.country
        )


        year_value = request.form.get(
            "year"
        )

        if year_value:
            coin.year = int(year_value)
        else:
            coin.year = None

        def parse_edit_year(field):
            raw = request.form.get(field, "").strip()
            try:
                return int(raw) if raw else None
            except (TypeError, ValueError):
                return None

        coin.date_start_year = parse_edit_year("date_start_year")
        coin.date_end_year = parse_edit_year("date_end_year")
        coin.date_is_approx = request.form.get("date_is_approx") == "1"

        if coin.date_start_year is not None or coin.date_end_year is not None:
            if coin.date_start_year is None:
                coin.date_start_year = coin.date_end_year
            if coin.date_end_year is None:
                coin.date_end_year = coin.date_start_year
            if coin.date_start_year > coin.date_end_year:
                coin.date_start_year, coin.date_end_year = coin.date_end_year, coin.date_start_year
            coin.year = coin.date_start_year


        coin.denomination = request.form.get(
            "denomination",
            ""
        )

        coin.material = request.form.get(
            "material",
            ""
        )


        # ==============================================
        # MINT INFORMATION
        # ==============================================

        coin.mint_id = (
            selected_mint.id
            if selected_mint
            else None
        )

        # Old text fields kept for compatibility
        coin.mint = mint_name

        coin.mint_mark = request.form.get(
            "mint_mark",
            ""
        )

        coin.location = mint_location


        # ==============================================
        # COLLECTION INFORMATION
        # ==============================================

        coin.condition = request.form.get(
            "condition",
            ""
        )


        quantity_value = request.form.get(
            "quantity"
        )

        if quantity_value:

            coin.quantity = int(
                quantity_value
            )

        else:

            coin.quantity = 1


        coin.date_acquired = request.form.get(
            "date_acquired",
            ""
        )

        coin.notes = request.form.get(
            "notes",
            ""
        )


        estimated_value = request.form.get(
            "estimated_value"
        )

        if estimated_value:

            coin.estimated_value = float(
                estimated_value
            )

        else:

            coin.estimated_value = None


        purchase_price = request.form.get(
            "purchase_price"
        )

        if purchase_price:
            coin.purchase_price = float(
                purchase_price
            )
        else:
            coin.purchase_price = None


        # ==============================================
        # NUMISTA TYPE ID
        # ==============================================

        if "numista_type_id" in request.form:

            value = request.form.get(
                "numista_type_id"
            )

            coin.numista_type_id = (
                int(value)
                if value
                else None
            )


        # ==============================================
        # NUMISTA ISSUE ID
        # ==============================================

        if "numista_issue_id" in request.form:

            value = request.form.get(
                "numista_issue_id"
            )

            coin.numista_issue_id = (
                int(value)
                if value
                else None
            )


        # ==============================================
        # VARIANT
        # ==============================================

        if "variant" in request.form:

            coin.variant = (
                request.form.get(
                    "variant"
                )
                or None
            )


        # ==============================================
        # MINTAGE
        # ==============================================

        if "mintage" in request.form:

            value = request.form.get(
                "mintage"
            )

            coin.mintage = (
                int(value)
                if value
                else None
            )


        # ==============================================
        # NUMISTA REFERENCE INFORMATION
        # ==============================================

        if "obverse_image" in request.form:

            coin.obverse_image = (
                request.form.get(
                    "obverse_image"
                )
                or None
            )


        if "reverse_image" in request.form:

            coin.reverse_image = (
                request.form.get(
                    "reverse_image"
                )
                or None
            )


        if "numista_url" in request.form:

            coin.numista_url = (
                request.form.get(
                    "numista_url"
                )
                or None
            )


        if "obverse_copyright" in request.form:

            coin.obverse_copyright = (
                request.form.get(
                    "obverse_copyright"
                )
                or None
            )


        if "obverse_license" in request.form:

            coin.obverse_license = (
                request.form.get(
                    "obverse_license"
                )
                or None
            )


        if "reverse_copyright" in request.form:

            coin.reverse_copyright = (
                request.form.get(
                    "reverse_copyright"
                )
                or None
            )


        if "reverse_license" in request.form:

            coin.reverse_license = (
                request.form.get(
                    "reverse_license"
                )
                or None
            )


        # ==============================================
        # SAVE
        # ==============================================

        db.session.commit()


        return redirect(
            url_for(
                "coin_detail",
                coin_id=coin.id
            )
        )


    return render_template(
        "edit_coin.html",
        coin=coin,
        mints=all_mints
    )
@app.route("/coin/<int:coin_id>/delete", methods=["POST"])
def delete_coin(coin_id):
    coin = Coin.query.get_or_404(coin_id)

    db.session.delete(coin)
    db.session.commit()

    return redirect(url_for("coins"))


# ============================================================
# NUMISTA API
# ============================================================

NUMISTA_BASE_URL = "https://api.numista.com/api/v3"

# Cache successful Numista responses in memory so repeated searches/details
# do not consume API quota. Railway may restart the process, so this cache is
# intentionally best-effort rather than persistent.
NUMISTA_CACHE_TTL = 60 * 60 * 24
NUMISTA_MIN_REQUEST_INTERVAL = 1.25
_numista_cache = {}
_numista_last_request_at = 0.0
_numista_cooldown_until = 0.0
NUMISTA_DEFAULT_COOLDOWN = 60


def _numista_get(path, headers, params=None):
    """GET Numista data with caching, gentle throttling, and friendly 429s."""
    global _numista_last_request_at, _numista_cooldown_until

    normalized_params = tuple(sorted((params or {}).items()))
    cache_key = (path, normalized_params)
    now = time.time()
    cached = _numista_cache.get(cache_key)

    if cached and now - cached[0] < NUMISTA_CACHE_TTL:
        return cached[1], None

    if now < _numista_cooldown_until:
        remaining = max(1, int(_numista_cooldown_until - now))
        minutes = max(1, (remaining + 59) // 60)
        return None, ({
            "error": f"Numista is temporarily limiting searches. Search is paused for about {minutes} more minute{'s' if minutes != 1 else ''} to avoid sending more requests."
        }, 429)

    elapsed = time.monotonic() - _numista_last_request_at
    if elapsed < NUMISTA_MIN_REQUEST_INTERVAL:
        time.sleep(NUMISTA_MIN_REQUEST_INTERVAL - elapsed)

    try:
        response = requests.get(
            f"{NUMISTA_BASE_URL}{path}",
            headers=headers,
            params=params,
            timeout=30,
        )
        _numista_last_request_at = time.monotonic()

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            try:
                cooldown_seconds = max(NUMISTA_DEFAULT_COOLDOWN, int(retry_after or 0))
            except (TypeError, ValueError):
                cooldown_seconds = NUMISTA_DEFAULT_COOLDOWN
            rate_headers = {
                key: value
                for key, value in response.headers.items()
                if key.lower() in {
                    "retry-after",
                    "x-ratelimit-limit",
                    "x-ratelimit-remaining",
                    "x-ratelimit-reset",
                }
            }
            app.logger.warning(
                "Numista 429 path=%s params=%r rate_headers=%r body=%r",
                path,
                params,
                rate_headers,
                response.text[:500],
            )
            # A monthly quota exhaustion is different from a short-term rate limit.
            # Do not start a misleading cooldown; fail fast with a clear message.
            if "quota exceeded" in response.text.lower():
                return None, ({
                    "error": (
                        "Numista monthly API quota has been reached. "
                        "You can still add the coin manually; Numista search will be available again when the quota resets."
                    ),
                    "quota_exceeded": True,
                }, 429)

            _numista_cooldown_until = time.time() + cooldown_seconds
            minutes = max(1, (cooldown_seconds + 59) // 60)
            return None, ({
                "error": f"Numista is temporarily limiting searches. Search has been paused for about {minutes} minutes so repeated taps do not send more requests."
            }, 429)

        response.raise_for_status()
        data = response.json()
        _numista_cache[cache_key] = (time.time(), data)
        return data, None

    except requests.RequestException:
        return None, ({
            "error": "Numista is temporarily unavailable. Please try again shortly."
        }, 502)
    except ValueError:
        return None, ({
            "error": "Numista returned an unexpected response. Please try again shortly."
        }, 502)


def get_numista_headers():
    api_key = os.getenv("NUMISTA_API_KEY")

    if not api_key:
        return None

    return {
        "Numista-API-Key": api_key
    }


@app.route("/api/numista/search")
def numista_search():

    headers = get_numista_headers()

    if not headers:
        return {
            "error": "NUMISTA_API_KEY was not found."
        }, 500

    query = request.args.get(
        "q",
        ""
    ).strip()

    year = request.args.get(
        "year",
        ""
    ).strip()

    if not query:
        return {
            "error": "A search term is required."
        }, 400

    params = {
        "q": f'"{query}"',
        "category": "coin",
        "lang": "en",
        "page": 1,
        "count": 20
    }

    if year:
        params["year"] = year

    data, api_error = _numista_get("/types", headers, params)
    if api_error:
        return api_error

    results = []

    for coin in data.get(
        "types",
        []
    ):

        issuer = (
            coin.get("issuer")
            or {}
        )

        results.append({
            "id": coin.get("id"),
            "title": coin.get("title"),
            "issuer": issuer.get("name"),
            "min_year": coin.get("min_year"),
            "max_year": coin.get("max_year")
        })

    return {
        "count": data.get(
            "count",
            len(results)
        ),
        "results": results
    }


@app.route("/api/numista/type/<int:type_id>")
def numista_type_details(type_id):

    headers = get_numista_headers()

    if not headers:
        return {
            "error":
            "NUMISTA_API_KEY was not found."
        }, 500

    coin, api_error = _numista_get(
        f"/types/{type_id}",
        headers,
        {"lang": "en"},
    )
    if api_error:
        return api_error

    issuer = (
        coin.get("issuer")
        or {}
    )

    value = (
        coin.get("value")
        or {}
    )

    composition = (
        coin.get("composition")
        or {}
    )

    obverse = (
        coin.get("obverse")
        or {}
    )

    reverse = (
        coin.get("reverse")
        or {}
    )


    mint_results = []

    for numista_mint in coin.get(
        "mints",
        []
    ):

        numista_mint_id = (
            numista_mint.get("id")
        )

        try:

            numista_mint_id = int(
                numista_mint_id
            )

        except (
            TypeError,
            ValueError
        ):

            continue


        local_mint = (
            Mint.query
            .filter_by(
                numista_id=
                numista_mint_id
            )
            .first()
        )


        mint_results.append({

            "numista_id":
            numista_mint_id,

            "name":
            numista_mint.get("name"),

            "local_mint_id":
            (
                local_mint.id
                if local_mint
                else None
            ),

            "local_mint_name":
            (
                local_mint.name
                if local_mint
                else None
            )
        })


    return {

        "id":
        coin.get("id"),

        "title":
        coin.get("title"),

        "issuer":
        issuer.get("name"),

        "min_year":
        coin.get("min_year"),

        "max_year":
        coin.get("max_year"),

        "denomination":
        value.get("text"),

        "composition":
        composition.get("text"),

        "weight":
        coin.get("weight"),

        "diameter":
        coin.get("size"),

        "thickness":
        coin.get("thickness"),

        "shape":
        coin.get("shape"),

        "orientation":
        coin.get("orientation"),


        # Numista page
        "numista_url":
        coin.get("url"),


        # Obverse
        "obverse_thumbnail":
        obverse.get("thumbnail"),

        "obverse_image":
        (
            obverse.get("picture")
            or
            obverse.get("image")
            or
            obverse.get("thumbnail")
        ),

        "obverse_copyright":
        (
            obverse.get(
                "picture_copyright"
            )
            or
            obverse.get("copyright")
        ),

        "obverse_license":
        (
            obverse.get(
                "picture_license_name"
            )
            or
            obverse.get("license")
        ),


        # Reverse
        "reverse_thumbnail":
        reverse.get("thumbnail"),

        "reverse_image":
        (
            reverse.get("picture")
            or
            reverse.get("image")
            or
            reverse.get("thumbnail")
        ),

        "reverse_copyright":
        (
            reverse.get(
                "picture_copyright"
            )
            or
            reverse.get("copyright")
        ),

        "reverse_license":
        (
            reverse.get(
                "picture_license_name"
            )
            or
            reverse.get("license")
        ),


        "mints":
        mint_results
    }
@app.route("/api/numista/type/<int:type_id>/issues")
def numista_type_issues(type_id):

    headers = get_numista_headers()

    if not headers:
        return {
            "error": "NUMISTA_API_KEY was not found."
        }, 500

    data, api_error = _numista_get(
        f"/types/{type_id}/issues",
        headers,
        {"lang": "en"},
    )
    if api_error:
        return api_error

    return data

def process_coin_photo(file_storage, coin_id, side, guided_capture=False):
    _load_image_tools()
    """
    Save the original upload and create a cleaned display copy.

    Improvements:
    - evaluates multiple circular edges instead of automatically choosing
      the largest circle, which helps avoid coin capsules/holders
    - prefers a strong inner metal edge near the center
    - crops tightly around the actual coin
    - attempts a conservative automatic upright rotation using OCR
    - preserves the original upload unchanged
    - uses a dark neutral background outside the detected coin
    - applies only mild contrast and sharpening
    """

    file_storage.stream.seek(0)

    original_image = ImageOps.exif_transpose(
        Image.open(file_storage.stream)
    ).convert("RGB")

    safe_name = secure_filename(
        file_storage.filename
    )

    original_filename = (
        f"coin_{coin_id}_{side}_original_{safe_name}"
    )

    original_path = os.path.join(
        UPLOAD_FOLDER,
        original_filename
    )

    original_image.save(
        original_path,
        quality=95
    )

    r2_upload_file(
        original_path,
        f"uploads/coins/{original_filename}"
    )

    # Guided Camera already performs the crop in the browser.
    # Do NOT run circle detection/Hough cropping a second time.
    if guided_capture:
        # The browser already cropped this guided photo.
        # Enhance the REAL pixels only: mild local contrast, brightness
        # normalization, and sharpening. No AI/redrawing is involved.
        guided_rgb = np.array(original_image)

        guided_lab = cv2.cvtColor(
            guided_rgb,
            cv2.COLOR_RGB2LAB
        )

        l_channel, a_channel, b_channel = cv2.split(
            guided_lab
        )

        clahe = cv2.createCLAHE(
            clipLimit=1.8,
            tileGridSize=(8, 8)
        )

        enhanced_l = clahe.apply(l_channel)

        guided_lab = cv2.merge(
            (enhanced_l, a_channel, b_channel)
        )

        enhanced_rgb = cv2.cvtColor(
            guided_lab,
            cv2.COLOR_LAB2RGB
        )

        # Gentle unsharp mask: makes lettering/relief crisper without
        # inventing details or changing the coin's actual design.
        blurred = cv2.GaussianBlur(
            enhanced_rgb,
            (0, 0),
            1.15
        )

        sharpened = cv2.addWeighted(
            enhanced_rgb,
            1.32,
            blurred,
            -0.32,
            0
        )

        # Guided Camera framing tells us the coin should be centered.
        # Replace only the area OUTSIDE a conservative centered circle
        # with a dark neutral background. The circle is intentionally
        # slightly generous so the real rim is preserved.
        h, w = sharpened.shape[:2]
        cx = w // 2
        cy = h // 2
        radius = int(min(h, w) * 0.455)

        yy, xx = np.ogrid[:h, :w]
        distance = np.sqrt(
            (xx - cx) ** 2
            + (yy - cy) ** 2
        )

        # Soft 8-pixel transition avoids a harsh cutout edge.
        feather = max(6, int(min(h, w) * 0.008))
        alpha = np.clip(
            (radius + feather - distance)
            / max(1, feather),
            0.0,
            1.0
        )[..., None]

        dark_background = np.full_like(
            sharpened,
            8
        )

        dark_backed = (
            sharpened.astype(np.float32) * alpha
            + dark_background.astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)

        guided_image = Image.fromarray(
            dark_backed
        )

        guided_image = ImageOps.fit(
            guided_image,
            (1000, 1000),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5)
        )

        processed_filename = (
            f"coin_{coin_id}_{side}_processed.jpg"
        )

        processed_path = os.path.join(
            UPLOAD_FOLDER,
            processed_filename
        )

        guided_image.save(
            processed_path,
            format="JPEG",
            quality=95,
            optimize=True
        )

        r2_upload_file(
            processed_path,
            f"uploads/coins/{processed_filename}"
        )

        return {
            "original_filename": original_filename,
            "processed_filename": processed_filename
        }

    rgb = np.array(original_image)

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    height, width = gray.shape[:2]

    # ------------------------------------------------------------
    # CIRCLE DETECTION
    # ------------------------------------------------------------

    detection_scale = min(
        1.0,
        1000.0 / max(height, width)
    )

    small_width = max(
        1,
        int(width * detection_scale)
    )

    small_height = max(
        1,
        int(height * detection_scale)
    )

    small_gray = cv2.resize(
        gray,
        (small_width, small_height),
        interpolation=(
            cv2.INTER_AREA
            if detection_scale < 1.0
            else cv2.INTER_LINEAR
        )
    )

    small_gray = cv2.GaussianBlur(
        small_gray,
        (7, 7),
        1.6
    )

    small_min_dimension = min(
        small_gray.shape[:2]
    )

    # A gradient image lets us score how strongly each proposed circle
    # follows a real physical edge.
    grad_x = cv2.Sobel(
        small_gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    grad_y = cv2.Sobel(
        small_gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient = cv2.magnitude(
        grad_x,
        grad_y
    )

    def circle_edge_strength(
        center_x,
        center_y,
        radius
    ):
        """
        Average gradient along a thin circular band.
        Strong real coin rims score higher than weak reflections.
        """

        angles = np.linspace(
            0,
            2 * np.pi,
            240,
            endpoint=False
        )

        strengths = []

        for offset in (-3, -1, 0, 1, 3):
            test_radius = max(
                2,
                radius + offset
            )

            xs = np.rint(
                center_x
                + test_radius * np.cos(angles)
            ).astype(int)

            ys = np.rint(
                center_y
                + test_radius * np.sin(angles)
            ).astype(int)

            valid = (
                (xs >= 0)
                & (xs < gradient.shape[1])
                & (ys >= 0)
                & (ys < gradient.shape[0])
            )

            if np.any(valid):
                strengths.append(
                    float(
                        np.mean(
                            gradient[
                                ys[valid],
                                xs[valid]
                            ]
                        )
                    )
                )

        return max(
            strengths,
            default=0.0
        )

    # Run Hough at several sensitivities. This intentionally keeps
    # multiple possible circles so an inner coin edge can beat a holder.
    circle_sets = []

    for hough_threshold in (
        42,
        36,
        30
    ):
        found = cv2.HoughCircles(
            small_gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            # Keep this small so Hough can return both the coin edge
            # and a larger, nearly-concentric holder/capsule edge.
            minDist=12,
            param1=120,
            param2=hough_threshold,
            # The coin itself should occupy a substantial part of the
            # photographed holder. Ignoring tiny circles prevents details
            # inside the design from being mistaken for the coin rim.
            minRadius=max(
                20,
                int(
                    small_min_dimension
                    * 0.20
                )
            ),
            maxRadius=max(
                24,
                int(
                    small_min_dimension
                    * 0.46
                )
            )
        )

        if found is not None:
            circle_sets.extend(
                np.round(
                    found[0]
                ).astype(int)
            )

    detected = None

    if circle_sets:
        image_center_x = (
            small_width / 2
        )

        image_center_y = (
            small_height / 2
        )

        candidates = []

        # De-duplicate very similar Hough results.
        unique_candidates = []

        for x, y, radius in circle_sets:
            duplicate = False

            for (
                old_x,
                old_y,
                old_radius
            ) in unique_candidates:
                if (
                    abs(x - old_x) < 8
                    and abs(y - old_y) < 8
                    and abs(radius - old_radius) < 8
                ):
                    duplicate = True
                    break

            if not duplicate:
                unique_candidates.append(
                    (
                        x,
                        y,
                        radius
                    )
                )

        for (
            x,
            y,
            radius
        ) in unique_candidates:

            center_distance = (
                (
                    x
                    - image_center_x
                ) ** 2
                +
                (
                    y
                    - image_center_y
                ) ** 2
            ) ** 0.5

            normalized_center_distance = (
                center_distance
                / max(
                    1.0,
                    small_min_dimension
                )
            )

            edge_strength = (
                circle_edge_strength(
                    x,
                    y,
                    radius
                )
            )

            # --------------------------------------------------------
            # HOLDER / CAPSULE PENALTY
            # --------------------------------------------------------
            #
            # If another circle is nearly concentric and slightly
            # smaller, the larger circle is commonly the holder.
            # Penalize the outer circle and favor the inner metal rim.
            holder_penalty = 0.0
            inner_bonus = 0.0

            for (
                other_x,
                other_y,
                other_radius
            ) in unique_candidates:

                if other_radius >= radius:
                    continue

                concentric_distance = (
                    (
                        x - other_x
                    ) ** 2
                    +
                    (
                        y - other_y
                    ) ** 2
                ) ** 0.5

                radius_ratio = (
                    other_radius
                    / max(
                        radius,
                        1
                    )
                )

                if (
                    concentric_distance
                    < radius * 0.20
                    and 0.72
                    <= radius_ratio
                    <= 0.92
                ):
                    holder_penalty = max(
                        holder_penalty,
                        90.0
                    )

            for (
                other_x,
                other_y,
                other_radius
            ) in unique_candidates:

                if other_radius <= radius:
                    continue

                concentric_distance = (
                    (
                        x - other_x
                    ) ** 2
                    +
                    (
                        y - other_y
                    ) ** 2
                ) ** 0.5

                radius_ratio = (
                    radius
                    / max(
                        other_radius,
                        1
                    )
                )

                if (
                    concentric_distance
                    < other_radius * 0.20
                    and 0.72
                    <= radius_ratio
                    <= 0.92
                ):
                    inner_bonus = max(
                        inner_bonus,
                        18.0
                    )

            # Strong rim + centered coin are the main signals.
            # Radius is only a mild bonus now, so a capsule cannot win
            # simply because it is larger.
            score = (
                edge_strength * 1.15
                + radius * 0.035
                - normalized_center_distance * 110
                + inner_bonus
                - holder_penalty
            )

            candidates.append(
                (
                    score,
                    x,
                    y,
                    radius
                )
            )

        if candidates:
            (
                _,
                center_x,
                center_y,
                radius
            ) = max(
                candidates,
                key=lambda item: item[0]
            )

            detected = (
                center_x
                / detection_scale,
                center_y
                / detection_scale,
                radius
                / detection_scale
            )

    # ------------------------------------------------------------
    # CROP + MASK
    # ------------------------------------------------------------

    if detected is None:
        cropped = crop_coin_region(
            original_image
        )

        processed = np.array(
            cropped
        )

        mask = None

    else:
        (
            center_x,
            center_y,
            radius
        ) = detected

        # Tighter than the old 10% margin while still leaving breathing room.
        margin = int(
            radius * 0.055
        )

        half_size = int(
            radius + margin
        )

        left = max(
            0,
            int(
                center_x
                - half_size
            )
        )

        top = max(
            0,
            int(
                center_y
                - half_size
            )
        )

        right = min(
            width,
            int(
                center_x
                + half_size
            )
        )

        bottom = min(
            height,
            int(
                center_y
                + half_size
            )
        )

        processed = rgb[
            top:bottom,
            left:right
        ].copy()

        local_center_x = (
            center_x
            - left
        )

        local_center_y = (
            center_y
            - top
        )

        crop_height, crop_width = (
            processed.shape[:2]
        )

        y_grid, x_grid = np.ogrid[
            :crop_height,
            :crop_width
        ]

        distance = np.sqrt(
            (
                x_grid
                - local_center_x
            ) ** 2
            +
            (
                y_grid
                - local_center_y
            ) ** 2
        )

        # Keep the physical metal coin and fade only the very edge.
        mask = np.clip(
            (
                radius
                + 1.5
                - distance
            )
            / 3.0,
            0,
            1
        )[..., None]

    # ------------------------------------------------------------
    # ORIENTATION
    # ------------------------------------------------------------
    #
    # Preserve the orientation of the original phone photo.
    # ImageOps.exif_transpose() above already applies the phone's EXIF
    # orientation correctly. OCR-based automatic rotation was removed
    # because raised/worn coin lettering can be misread and rotate an
    # already-upright coin the wrong way.
    #
    # A separate manual rotate control can be added to the coin page
    # later for the uncommon case where the photographed coin itself
    # was physically turned.

    # ------------------------------------------------------------
    # GENTLE ENHANCEMENT
    # ------------------------------------------------------------

    lab = cv2.cvtColor(
        processed,
        cv2.COLOR_RGB2LAB
    )

    (
        lightness,
        channel_a,
        channel_b
    ) = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=1.30,
        tileGridSize=(8, 8)
    )

    lightness = clahe.apply(
        lightness
    )

    enhanced = cv2.cvtColor(
        cv2.merge(
            [
                lightness,
                channel_a,
                channel_b
            ]
        ),
        cv2.COLOR_LAB2RGB
    )

    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        1.0
    )

    enhanced = cv2.addWeighted(
        enhanced,
        1.16,
        blurred,
        -0.16,
        0
    )

    # ------------------------------------------------------------
    # DARK BACKGROUND
    # ------------------------------------------------------------

    if mask is not None:
        background = np.full_like(
            enhanced,
            12
        )

        enhanced = (
            enhanced * mask
            +
            background
            * (
                1 - mask
            )
        ).astype(
            np.uint8
        )

    processed_image = Image.fromarray(
        enhanced
    )

    processed_image = ImageOps.fit(
        processed_image,
        (1000, 1000),
        method=Image.Resampling.LANCZOS
    )

    processed_filename = (
        f"coin_{coin_id}_{side}_processed.jpg"
    )

    processed_path = os.path.join(
        UPLOAD_FOLDER,
        processed_filename
    )

    processed_image.save(
        processed_path,
        quality=95
    )

    r2_upload_file(
        processed_path,
        f"uploads/coins/{processed_filename}"
    )

    return {
        "original_filename":
            original_filename,
        "processed_filename":
            processed_filename
    }


@app.route(
    "/coin/<int:coin_id>/photos",
    methods=["POST"]
)
def upload_coin_photos(coin_id):

    coin = Coin.query.get_or_404(
        coin_id
    )

    obverse_file = request.files.get(
        "personal_obverse_image"
    )

    reverse_file = request.files.get(
        "personal_reverse_image"
    )

    guided_obverse = (
        request.form.get("guided_obverse") == "1"
    )

    guided_reverse = (
        request.form.get("guided_reverse") == "1"
    )

    if (
        obverse_file
        and obverse_file.filename
    ):
        result = process_coin_photo(
            obverse_file,
            coin.id,
            "obverse",
            guided_capture=guided_obverse
        )

        coin.personal_obverse_image = (
            result["processed_filename"]
        )

    if (
        reverse_file
        and reverse_file.filename
    ):
        result = process_coin_photo(
            reverse_file,
            coin.id,
            "reverse",
            guided_capture=guided_reverse
        )

        coin.personal_reverse_image = (
            result["processed_filename"]
        )

    db.session.commit()

    return redirect(
        url_for(
            "coin_detail",
            coin_id=coin.id
        )
    )

# --- COIN PHOTO EDITOR ---
@app.route(
    "/coin/<int:coin_id>/photos/<side>/edit",
    methods=["GET", "POST"]
)
def edit_coin_photo(coin_id, side):
    _load_image_tools()
    coin = Coin.query.get_or_404(coin_id)

    if side not in {"obverse", "reverse"}:
        abort(404)

    current_filename = (
        coin.personal_obverse_image
        if side == "obverse"
        else coin.personal_reverse_image
    )

    if not current_filename:
        return redirect(url_for("coin_detail", coin_id=coin.id))

    original_prefix = f"coin_{coin.id}_{side}_original_"

    originals = sorted(
        Path(UPLOAD_FOLDER).glob(original_prefix + "*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    source_path = (
        originals[0]
        if originals
        else Path(UPLOAD_FOLDER) / current_filename
    )

    if not source_path.exists():
        r2_download_file(
            source_path,
            f"uploads/coins/{current_filename}"
        )

    if not source_path.exists():
        abort(404)

    if request.method == "POST":

        def number(name, default):
            try:
                return float(request.form.get(name, default))
            except (TypeError, ValueError):
                return default

        left_pct = max(0.0, min(99.0, number("crop_left", 0)))
        top_pct = max(0.0, min(99.0, number("crop_top", 0)))
        right_pct = max(left_pct + 1.0, min(100.0, number("crop_right", 100)))
        bottom_pct = max(top_pct + 1.0, min(100.0, number("crop_bottom", 100)))

        brightness = max(0.5, min(1.8, number("brightness", 1)))
        contrast = max(0.5, min(1.8, number("contrast", 1)))
        sharpness = max(0.5, min(2.5, number("sharpness", 1)))

        try:
            rotate_steps = int(request.form.get("rotate_steps", 0))
        except (TypeError, ValueError):
            rotate_steps = 0

        rotate_steps %= 4

        image = ImageOps.exif_transpose(
            Image.open(source_path)
        ).convert("RGB")

        if rotate_steps:
            image = image.rotate(
                -90 * rotate_steps,
                expand=True,
                resample=Image.Resampling.BICUBIC
            )

        width, height = image.size

        left = int(width * left_pct / 100.0)
        top = int(height * top_pct / 100.0)
        right = int(width * right_pct / 100.0)
        bottom = int(height * bottom_pct / 100.0)

        left = max(0, min(width - 2, left))
        top = max(0, min(height - 2, top))
        right = max(left + 2, min(width, right))
        bottom = max(top + 2, min(height, bottom))

        image = image.crop((left, top, right, bottom))

        image.thumbnail(
            (1000, 1000),
            Image.Resampling.LANCZOS
        )

        canvas = Image.new(
            "RGB",
            (1000, 1000),
            (12, 12, 12)
        )

        x = (1000 - image.width) // 2
        y = (1000 - image.height) // 2

        canvas.paste(image, (x, y))

        canvas = ImageEnhance.Brightness(canvas).enhance(brightness)
        canvas = ImageEnhance.Contrast(canvas).enhance(contrast)
        canvas = ImageEnhance.Sharpness(canvas).enhance(sharpness)

        processed_filename = f"coin_{coin.id}_{side}_processed.jpg"
        processed_path = Path(UPLOAD_FOLDER) / processed_filename

        canvas.save(
            processed_path,
            "JPEG",
            quality=95
        )

        r2_upload_file(
            processed_path,
            f"uploads/coins/{processed_filename}"
        )

        if side == "obverse":
            coin.personal_obverse_image = processed_filename
        else:
            coin.personal_reverse_image = processed_filename

        db.session.commit()

        return redirect(
            url_for(
                "coin_detail",
                coin_id=coin.id
            )
        )

    return render_template(
        "coin_photo_editor.html",
        coin=coin,
        side=side,
        source_filename=source_path.name
    )


@app.route(
    "/coin/<int:coin_id>/photos/<side>/remove",
    methods=["POST"]
)
def remove_coin_photo(coin_id, side):

    coin = Coin.query.get_or_404(
        coin_id
    )

    if side == "obverse":

        filename = (
            coin.personal_obverse_image
        )

        coin.personal_obverse_image = None

    elif side == "reverse":

        filename = (
            coin.personal_reverse_image
        )

        coin.personal_reverse_image = None

    else:

        return redirect(
            url_for(
                "coin_detail",
                coin_id=coin.id
            )
        )


    # Delete the actual image file
    if filename:

        file_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        if os.path.isfile(file_path):
            os.remove(file_path)

        r2_delete_key(
            f"uploads/coins/{filename}"
        )

        r2_delete_prefix(
            f"uploads/coins/coin_{coin.id}_{side}_original_"
        )


    db.session.commit()


    return redirect(
        url_for(
            "coin_detail",
            coin_id=coin.id
        )
    )


def crop_coin_region(image):
    _load_image_tools()
    """
    Try to isolate the coin from the surrounding photo.
    If circle detection fails, use a centered square crop.
    """

    rgb = np.array(image)

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (9, 9),
        2
    )

    height, width = gray.shape[:2]

    min_radius = int(
        min(height, width) * 0.18
    )

    max_radius = int(
        min(height, width) * 0.52
    )

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(height, width) // 2,
        param1=120,
        param2=45,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None:

        circles = np.round(
            circles[0, :]
        ).astype("int")

        # Prefer the largest detected circle.
        x, y, radius = max(
            circles,
            key=lambda item: item[2]
        )

        padding = int(
            radius * 0.08
        )

        left = max(
            0,
            x - radius - padding
        )

        top = max(
            0,
            y - radius - padding
        )

        right = min(
            width,
            x + radius + padding
        )

        bottom = min(
            height,
            y + radius + padding
        )

        cropped = image.crop(
            (
                left,
                top,
                right,
                bottom
            )
        )

        return cropped


    # Fallback: centered square crop.
    side = int(
        min(width, height) * 0.92
    )

    left = max(
        0,
        (width - side) // 2
    )

    top = max(
        0,
        (height - side) // 2
    )

    return image.crop(
        (
            left,
            top,
            left + side,
            top + side
        )
    )


def prepare_coin_base_image(file_storage):
    _load_image_tools()
    """
    Load, rotate correctly, crop around the coin, and resize to a
    predictable working size.
    """

    file_storage.stream.seek(0)

    image = Image.open(
        file_storage.stream
    )

    image = ImageOps.exif_transpose(
        image
    ).convert("RGB")

    image = crop_coin_region(
        image
    )

    target_size = 1400

    scale = (
        target_size
        /
        max(image.size)
    )

    if scale != 1:

        image = image.resize(
            (
                max(
                    1,
                    int(image.width * scale)
                ),
                max(
                    1,
                    int(image.height * scale)
                )
            )
        )

    return image


def detect_text_regions(image):
    _load_image_tools()
    """
    Look for text-like horizontal groups of edges inside the coin crop.
    This is intentionally conservative: it is better to return a few
    candidate bands than to OCR the entire textured coin surface.
    """

    rgb = np.array(
        image
    )

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    height, width = gray.shape[:2]

    # Improve local contrast without making every scratch too strong.
    clahe = cv2.createCLAHE(
        clipLimit=1.8,
        tileGridSize=(8, 8)
    )

    contrast = clahe.apply(
        gray
    )

    # Horizontal gradient emphasizes letter strokes arranged in lines.
    grad_x = cv2.Sobel(
        contrast,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    grad_x = np.absolute(
        grad_x
    )

    if grad_x.max() > 0:
        grad_x = (
            grad_x
            /
            grad_x.max()
            *
            255
        ).astype("uint8")
    else:
        grad_x = grad_x.astype(
            "uint8"
        )

    grad_x = cv2.GaussianBlur(
        grad_x,
        (5, 5),
        0
    )

    _, binary = cv2.threshold(
        grad_x,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    # Join nearby character strokes into text-like bands.
    kernel_width = max(
        15,
        width // 45
    )

    kernel_height = max(
        3,
        height // 220
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            kernel_width,
            kernel_height
        )
    )

    connected = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        connected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        area = (
            w
            *
            h
        )

        if area <= 0:
            continue

        width_ratio = (
            w
            /
            width
        )

        height_ratio = (
            h
            /
            height
        )

        aspect = (
            w
            /
            max(h, 1)
        )

        # Require a reasonably line-like shape.
        if width_ratio < 0.12:
            continue

        if height_ratio < 0.015:
            continue

        if height_ratio > 0.28:
            continue

        if aspect < 1.4:
            continue

        padding_x = int(
            w * 0.08
        )

        padding_y = int(
            h * 0.45
        )

        left = max(
            0,
            x - padding_x
        )

        top = max(
            0,
            y - padding_y
        )

        right = min(
            width,
            x + w + padding_x
        )

        bottom = min(
            height,
            y + h + padding_y
        )

        candidates.append(
            (
                left,
                top,
                right,
                bottom,
                area
            )
        )


    candidates.sort(
        key=lambda item: item[4],
        reverse=True
    )


    # Remove heavily overlapping boxes.
    final_boxes = []

    for box in candidates:

        left, top, right, bottom, area = box

        keep = True

        for existing in final_boxes:

            e_left, e_top, e_right, e_bottom, _ = existing

            inter_left = max(
                left,
                e_left
            )

            inter_top = max(
                top,
                e_top
            )

            inter_right = min(
                right,
                e_right
            )

            inter_bottom = min(
                bottom,
                e_bottom
            )

            if (
                inter_right > inter_left
                and
                inter_bottom > inter_top
            ):

                intersection = (
                    inter_right - inter_left
                ) * (
                    inter_bottom - inter_top
                )

                smaller = min(
                    (
                        right - left
                    ) * (
                        bottom - top
                    ),
                    (
                        e_right - e_left
                    ) * (
                        e_bottom - e_top
                    )
                )

                if (
                    smaller > 0
                    and
                    intersection / smaller > 0.55
                ):
                    keep = False
                    break

        if keep:
            final_boxes.append(
                box
            )

        if len(final_boxes) >= 12:
            break


    regions = []

    for index, box in enumerate(
        final_boxes,
        start=1
    ):

        left, top, right, bottom, _ = box

        regions.append(
            (
                f"detected_{index}",
                image.crop(
                    (
                        left,
                        top,
                        right,
                        bottom
                    )
                )
            )
        )


    # Add a few fallback bands in case contour detection misses curved text.
    def add_band(
        name,
        left,
        top,
        right,
        bottom
    ):

        regions.append(
            (
                name,
                image.crop(
                    (
                        int(left),
                        int(top),
                        int(right),
                        int(bottom)
                    )
                )
            )
        )


    add_band(
        "top_band",
        width * 0.08,
        height * 0.05,
        width * 0.92,
        height * 0.30
    )

    add_band(
        "bottom_band",
        width * 0.08,
        height * 0.70,
        width * 0.92,
        height * 0.95
    )

    add_band(
        "middle_band",
        width * 0.12,
        height * 0.35,
        width * 0.88,
        height * 0.65
    )

    return regions


def build_region_variants(region_image):
    _load_image_tools()
    """
    Create OCR-friendly versions of a detected text region.
    """

    rgb = np.array(
        region_image
    )

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    height, width = gray.shape[:2]

    longest = max(
        height,
        width
    )

    scale = max(
        1.0,
        1000
        /
        max(longest, 1)
    )

    scale = min(
        scale,
        4.0
    )

    if scale > 1.0:

        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    clahe = cv2.createCLAHE(
        clipLimit=1.8,
        tileGridSize=(8, 8)
    )

    contrast = clahe.apply(
        gray
    )

    # Mild bilateral filtering keeps letter edges while reducing texture.
    denoised = cv2.bilateralFilter(
        contrast,
        7,
        45,
        45
    )

    blur = cv2.GaussianBlur(
        denoised,
        (0, 0),
        0.8
    )

    sharpened = cv2.addWeighted(
        denoised,
        1.5,
        blur,
        -0.5,
        0
    )

    _, otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    return [
        denoised,
        sharpened,
        otsu
    ]


def normalize_ocr_line(raw_line):
    line = re.sub(
        r"\s+",
        " ",
        raw_line
    ).strip()

    line = re.sub(
        r"^[^A-Za-z0-9]+",
        "",
        line
    )

    line = re.sub(
        r"[^A-Za-z0-9]+$",
        "",
        line
    )

    return line.strip()


def is_useful_ocr_line(line):
    if not line:
        return False

    if len(line) < 3:
        return False

    if len(line) > 40:
        return False

    letters = sum(
        char.isalpha()
        for char in line
    )

    digits = sum(
        char.isdigit()
        for char in line
    )

    alphanumeric = (
        letters
        +
        digits
    )

    if alphanumeric < 3:
        return False

    ratio = (
        alphanumeric
        /
        max(len(line), 1)
    )

    if ratio < 0.72:
        return False

    if re.fullmatch(
        r"(1[0-9]{3}|20[0-9]{2})",
        line
    ):
        return True

    if letters < 3:
        return False

    # Reject highly alternating gibberish that often comes from texture.
    transitions = 0

    previous_type = None

    for char in line:

        if char.isalpha():
            current_type = "A"

        elif char.isdigit():
            current_type = "D"

        else:
            current_type = "X"

        if (
            previous_type is not None
            and
            current_type != previous_type
        ):
            transitions += 1

        previous_type = current_type

    if (
        len(line) >= 12
        and
        transitions > len(line) * 0.55
    ):
        return False

    return True


def clean_ocr_text(text):
    useful_lines = []

    for raw_line in text.splitlines():

        line = normalize_ocr_line(
            raw_line
        )

        if not is_useful_ocr_line(
            line
        ):
            continue

        if line not in useful_lines:
            useful_lines.append(
                line
            )

    return "\n".join(
        useful_lines[:6]
    )


def score_ocr_text(text):
    if not text:
        return -100

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    score = 0

    for line in lines:

        letters = sum(
            char.isalpha()
            for char in line
        )

        digits = sum(
            char.isdigit()
            for char in line
        )

        alphanumeric = (
            letters
            +
            digits
        )

        ratio = (
            alphanumeric
            /
            max(len(line), 1)
        )

        if re.search(
            r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)",
            line
        ):
            score += 18

        # Reward actual word-like runs.
        word_runs = re.findall(
            r"[A-Za-z]{3,}",
            line
        )

        score += min(
            len(word_runs) * 4,
            12
        )

        if letters >= 8:
            score += 5

        elif letters >= 4:
            score += 2

        if ratio >= 0.90:
            score += 4

        elif ratio >= 0.80:
            score += 2

        if digits > 0:
            score += 1

    if len(lines) > 5:
        score -= (
            len(lines) - 5
        ) * 3

    return score


def normalize_possible_year_token(token):
    token = token.strip()

    if len(token) != 4:
        return ""

    substitutions = {
        "I": "1",
        "l": "1",
        "|": "1",
        "O": "0",
        "o": "0",
        "S": "5",
        "B": "8"
    }

    converted = ""

    for char in token:

        if char.isdigit():
            converted += char

        elif char in substitutions:
            converted += substitutions[char]

        else:
            converted += "?"

    if "?" in converted:
        return ""

    try:
        year = int(
            converted
        )
    except ValueError:
        return ""

    current_year = datetime.now().year

    if 1000 <= year <= current_year:
        return str(year)

    return ""


def find_possible_year(text):
    current_year = datetime.now().year

    exact_matches = re.findall(
        r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)",
        text
    )

    for match in exact_matches:

        year = int(
            match
        )

        if 1000 <= year <= current_year:
            return str(year)


    rough_tokens = re.findall(
        r"(?<![A-Za-z0-9])[A-Za-z0-9|]{4}(?![A-Za-z0-9])",
        text
    )

    for token in rough_tokens:

        corrected = normalize_possible_year_token(
            token
        )

        if corrected:
            return corrected

    return ""


def run_region_ocr(region_image):
    _load_image_tools()
    variants = build_region_variants(
        region_image
    )

    configs = [
        "--oem 3 --psm 7",
        "--oem 3 --psm 8",
        "--oem 3 --psm 11"
    ]

    best_text = ""
    best_score = -1000

    for image in variants:

        for config in configs:

            raw_text = pytesseract.image_to_string(
                image,
                lang="eng",
                config=config
            )

            cleaned = clean_ocr_text(
                raw_text
            )

            candidate_score = score_ocr_text(
                cleaned
            )

            if candidate_score > best_score:

                best_score = candidate_score
                best_text = cleaned

    return (
        best_text,
        best_score
    )


def run_coin_ocr(file_storage):
    """
    Detect likely text bands first, OCR only those regions, and keep
    a very small number of the strongest lines.
    """

    base_image = prepare_coin_base_image(
        file_storage
    )

    regions = detect_text_regions(
        base_image
    )

    region_results = []

    for region_name, region_image in regions:

        text, score = run_region_ocr(
            region_image
        )

        if (
            text
            and
            score >= 8
        ):

            region_results.append(
                {
                    "region":
                    region_name,

                    "text":
                    text,

                    "score":
                    score
                }
            )


    region_results.sort(
        key=lambda item: item["score"],
        reverse=True
    )


    final_lines = []

    for result in region_results[:4]:

        for line in result["text"].splitlines():

            if (
                line
                and
                line not in final_lines
            ):

                final_lines.append(
                    line
                )

            if len(final_lines) >= 6:
                break

        if len(final_lines) >= 6:
            break


    final_text = "\n".join(
        final_lines
    )

    if score_ocr_text(
        final_text
    ) < 10:
        return ""

    return final_text


@app.route(
    "/api/identify/ocr",
    methods=["POST"]
)
def identify_coin_ocr():
    _load_image_tools()

    obverse_file = request.files.get(
        "obverse"
    )

    reverse_file = request.files.get(
        "reverse"
    )

    if not (
        obverse_file
        and obverse_file.filename
    ) and not (
        reverse_file
        and reverse_file.filename
    ):

        return {
            "error":
            "Choose at least one coin photo."
        }, 400

    results = {
        "obverse_text": "",
        "reverse_text": "",
        "combined_text": "",
        "suggested_year": ""
    }

    try:

        if (
            obverse_file
            and obverse_file.filename
        ):

            results["obverse_text"] = (
                run_coin_ocr(
                    obverse_file
                )
            )

        if (
            reverse_file
            and reverse_file.filename
        ):

            results["reverse_text"] = (
                run_coin_ocr(
                    reverse_file
                )
            )

    except (
        pytesseract.TesseractNotFoundError,
        FileNotFoundError
    ):

        return {
            "error":
            "Tesseract could not be found at "
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        }, 500

    except Exception as exc:

        return {
            "error":
            "OCR could not process the image.",
            "details":
            str(exc)
        }, 500

    combined_parts = [
        results["obverse_text"],
        results["reverse_text"]
    ]

    results["combined_text"] = (
        "\n".join(
            part
            for part in combined_parts
            if part
        )
    )

    results["suggested_year"] = (
        find_possible_year(
            results["combined_text"]
        )
    )

    return results



@app.route("/identify", methods=["GET", "POST"])
def identify_coin():

    results = []
    error = None

    clues = {
        "year": "",
        "country": "",
        "denomination": "",
        "mint_mark": "",
        "visible_text": ""
    }

    if request.method == "POST":

        clues["year"] = request.form.get(
            "year",
            ""
        ).strip()

        clues["country"] = request.form.get(
            "country",
            ""
        ).strip()

        clues["denomination"] = request.form.get(
            "denomination",
            ""
        ).strip()

        clues["mint_mark"] = request.form.get(
            "mint_mark",
            ""
        ).strip()

        clues["visible_text"] = request.form.get(
            "visible_text",
            ""
        ).strip()

        headers = get_numista_headers()

        if not headers:

            error = (
                "NUMISTA_API_KEY was not found."
            )

        else:

            # -------------------------------------------------
            # BUILD SEVERAL SEARCHES INSTEAD OF ONLY ONE
            # -------------------------------------------------
            search_queries = []

            denomination = clues["denomination"]
            country = clues["country"]
            visible_text = clues["visible_text"]

            if denomination and country:
                search_queries.append(
                    f"{denomination} {country}"
                )

            if denomination:
                search_queries.append(
                    denomination
                )

            if visible_text and denomination:
                search_queries.append(
                    f"{denomination} {visible_text}"
                )

            if visible_text:
                search_queries.append(
                    visible_text
                )

            if country and visible_text:
                search_queries.append(
                    f"{country} {visible_text}"
                )

            if country and not denomination:
                search_queries.append(
                    country
                )

            # Remove duplicate searches while keeping order.
            search_queries = list(
                dict.fromkeys(
                    query.strip()
                    for query in search_queries
                    if query.strip()
                )
            )

            if not search_queries:

                error = (
                    "Enter at least a denomination, "
                    "country/issuer, or visible text."
                )

            else:

                # Merge candidates by Numista type ID so the
                # same coin found by several searches appears once.
                merged = {}

                try:

                    for query in search_queries:

                        params = {
                            "q": query,
                            "category": "coin",
                            "lang": "en",
                            "page": 1,
                            "count": 50
                        }

                        # First try with the exact year.
                        if clues["year"]:
                            params["year"] = clues["year"]

                        data, api_error = _numista_get(
                            "/types",
                            headers,
                            params
                        )

                        if api_error:
                            error = api_error[0].get_json().get(
                                "error",
                                "Numista search failed."
                            )
                            break

                        for item in data.get(
                            "types",
                            []
                        ):

                            type_id = item.get("id")

                            if type_id is not None:
                                merged[type_id] = item


                    # If the year-filtered searches found very few
                    # candidates, run denomination searches again
                    # without the year filter. This helps when Numista's
                    # date metadata/search behavior is unexpected.
                    if (
                        clues["year"]
                        and
                        len(merged) < 20
                    ):

                        fallback_queries = []

                        if denomination:
                            fallback_queries.append(
                                denomination
                            )

                        if denomination and country:
                            fallback_queries.append(
                                f"{denomination} {country}"
                            )

                        fallback_queries = list(
                            dict.fromkeys(
                                fallback_queries
                            )
                        )

                        for query in fallback_queries:

                            params = {
                                "q": query,
                                "category": "coin",
                                "lang": "en",
                                "page": 1,
                                "count": 50
                            }

                            data, api_error = _numista_get(
                                "/types",
                                headers,
                                params
                            )

                            if api_error:
                                error = api_error[0].get_json().get(
                                    "error",
                                    "Numista search failed."
                                )
                                break

                            for item in data.get(
                                "types",
                                []
                            ):

                                type_id = item.get("id")

                                if type_id is not None:
                                    merged[type_id] = item


                    for item in merged.values():

                        issuer = (
                            item.get("issuer")
                            or {}
                        )

                        title = (
                            item.get("title")
                            or ""
                        )

                        issuer_name = (
                            issuer.get("name")
                            or ""
                        )

                        score = 0

                        title_lower = title.lower()
                        issuer_lower = issuer_name.lower()


                        # -----------------------------
                        # DENOMINATION
                        # -----------------------------
                        if clues["denomination"]:

                            denomination_lower = (
                                clues[
                                    "denomination"
                                ].lower()
                            )

                            entered_tokens = re.findall(
                                r"[a-z0-9]+",
                                denomination_lower
                            )

                            title_tokens = re.findall(
                                r"[a-z0-9]+",
                                title_lower
                            )

                            entered_number = next(
                                (
                                    token
                                    for token in entered_tokens
                                    if token.isdigit()
                                ),
                                None
                            )

                            entered_unit = next(
                                (
                                    token
                                    for token in entered_tokens
                                    if token.isalpha()
                                ),
                                None
                            )

                            title_numbers = {
                                token
                                for token in title_tokens
                                if token.isdigit()
                            }

                            title_words = {
                                token
                                for token in title_tokens
                                if token.isalpha()
                            }

                            exact_number_match = (
                                entered_number is not None
                                and
                                entered_number in title_numbers
                            )

                            exact_unit_match = (
                                entered_unit is not None
                                and
                                entered_unit in title_words
                            )

                            if (
                                exact_number_match
                                and
                                exact_unit_match
                            ):
                                score += 18

                            elif exact_unit_match:

                                # Same currency unit but a different value
                                # should stay visible, just much lower.
                                score -= 4

                            elif entered_unit is not None:

                                # Completely different denomination family
                                # such as Pfennig when the user entered Mark.
                                score -= 18

                            elif (
                                denomination_lower
                                in title_lower
                            ):
                                score += 8


                        # -----------------------------
                        # COUNTRY / ISSUER
                        # -----------------------------
                        if clues["country"]:

                            country_lower = (
                                clues[
                                    "country"
                                ].lower()
                            )

                            if (
                                country_lower
                                == issuer_lower
                            ):
                                score += 8

                            elif (
                                country_lower
                                in issuer_lower
                                or
                                issuer_lower
                                in country_lower
                            ):
                                score += 6


                        # -----------------------------
                        # YEAR
                        # -----------------------------
                        if clues["year"]:

                            try:

                                target_year = int(
                                    clues["year"]
                                )

                                min_year = item.get(
                                    "min_year"
                                )

                                max_year = item.get(
                                    "max_year"
                                )

                                if (
                                    min_year is not None
                                    and
                                    max_year is not None
                                ):

                                    min_year = int(
                                        min_year
                                    )

                                    max_year = int(
                                        max_year
                                    )

                                    if (
                                        min_year
                                        <= target_year
                                        <= max_year
                                    ):
                                        score += 10

                                        if (
                                            min_year
                                            == target_year
                                            and
                                            max_year
                                            == target_year
                                        ):
                                            score += 4

                                    else:
                                        # Strongly demote coins that
                                        # cannot belong to the entered year.
                                        score -= 12

                            except (
                                TypeError,
                                ValueError
                            ):
                                pass


                        # -----------------------------
                        # VISIBLE TEXT
                        # -----------------------------
                        if clues["visible_text"]:

                            visible_words = [
                                word.lower()
                                for word in re.findall(
                                    r"[A-Za-z0-9]+",
                                    clues[
                                        "visible_text"
                                    ]
                                )
                                if len(word) >= 3
                            ]

                            title_matches = sum(
                                1
                                for word
                                in visible_words
                                if word in title_lower
                            )

                            issuer_matches = sum(
                                1
                                for word
                                in visible_words
                                if word in issuer_lower
                            )

                            score += (
                                title_matches * 3
                                +
                                issuer_matches * 2
                            )


                        # -----------------------------
                        # MINT MARK
                        # -----------------------------
                        if clues["mint_mark"]:

                            mint_mark = clues[
                                "mint_mark"
                            ].strip().lower()

                            if (
                                mint_mark
                                and
                                re.search(
                                    rf"(?<![a-z0-9])"
                                    rf"{re.escape(mint_mark)}"
                                    rf"(?![a-z0-9])",
                                    title_lower
                                )
                            ):
                                score += 2


                        results.append({
                            "id": item.get("id"),
                            "title": title,
                            "issuer": issuer_name,
                            "min_year": item.get(
                                "min_year"
                            ),
                            "max_year": item.get(
                                "max_year"
                            ),
                            "score": score
                        })


                    results.sort(
                        key=lambda item: (
                            item["score"],
                            item["title"]
                        ),
                        reverse=True
                    )

                    # Keep the page manageable while allowing a
                    # substantially larger candidate pool.
                    results = results[:40]


                except requests.RequestException as exc:

                    error = (
                        "Numista search failed: "
                        + str(exc)
                    )


    return render_template(
        "identify_coin.html",
        results=results,
        error=error,
        clues=clues
    )

@app.route("/timeline")
def collection_timeline():

    all_coins = Coin.query.order_by(
        Coin.year.asc()
    ).all()

    return render_template(
        "timeline.html",
        coins=all_coins
    )  
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=False)