"""Banknote collection routes. Additive; artifacts and existing coin routes are unchanged."""
from datetime import datetime
from flask import Blueprint, abort, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename
from pathlib import Path
import uuid

def register_banknotes(app, db):
    class Banknote(db.Model):
        __tablename__ = "banknote"
        id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.String(180), nullable=False)
        country = db.Column(db.String(120))
        set_country = db.Column(db.String(120))
        denomination = db.Column(db.String(100))
        currency = db.Column(db.String(100))
        year = db.Column(db.Integer)
        date_start_year = db.Column(db.Integer)
        date_end_year = db.Column(db.Integer)
        date_is_approx = db.Column(db.Boolean, nullable=False, default=False)
        issuing_authority = db.Column(db.String(180))
        serial_number = db.Column(db.String(120))
        catalog_number = db.Column(db.String(120))
        portrait = db.Column(db.String(180))
        reverse_design = db.Column(db.String(180))
        condition = db.Column(db.String(100))
        estimated_value = db.Column(db.Float)
        purchase_price = db.Column(db.Float)
        date_acquired = db.Column(db.String(30))
        source = db.Column(db.String(180))
        notes = db.Column(db.Text)
        created_at = db.Column(db.DateTime, default=datetime.utcnow)
        photos = db.relationship("BanknotePhoto", back_populates="banknote", cascade="all, delete-orphan", order_by="BanknotePhoto.sort_order")

    class BanknotePhoto(db.Model):
        __tablename__ = "banknote_photo"
        id = db.Column(db.Integer, primary_key=True)
        banknote_id = db.Column(db.Integer, db.ForeignKey("banknote.id"), nullable=False, index=True)
        filename = db.Column(db.String(255), nullable=False)
        side = db.Column(db.String(20))
        sort_order = db.Column(db.Integer, nullable=False, default=0)
        created_at = db.Column(db.DateTime, default=datetime.utcnow)
        banknote = db.relationship("Banknote", back_populates="photos")

    class CoinSetAllowedType(db.Model):
        __tablename__ = "coin_set_allowed_type"
        set_id = db.Column(db.Integer, db.ForeignKey("coin_set.id"), primary_key=True)
        collectible_type = db.Column(db.String(20), primary_key=True)

    class BanknoteSetMembership(db.Model):
        __tablename__ = "banknote_set_memberships"
        set_id = db.Column(db.Integer, db.ForeignKey("coin_set.id"), primary_key=True)
        banknote_id = db.Column(db.Integer, db.ForeignKey("banknote.id"), primary_key=True)

    class BanknoteSetSlotSelection(db.Model):
        __tablename__ = "banknote_set_slot_selection"
        set_id = db.Column(db.Integer, db.ForeignKey("coin_set.id"), primary_key=True)
        slot_type = db.Column(db.String(30), primary_key=True)
        slot_id = db.Column(db.Integer, primary_key=True)
        banknote_id = db.Column(db.Integer, db.ForeignKey("banknote.id"), nullable=False)

    bp = Blueprint("banknotes", __name__)
    def owner_required():
        if not session.get("is_owner"):
            abort(403)

    def number(value, caster):
        if not value or not value.strip():
            return None
        return caster(value)

    def save_fields(note):
        fields = ("name","country","set_country","denomination","currency","issuing_authority",
                  "serial_number","catalog_number","portrait","reverse_design","condition",
                  "date_acquired","source","notes")
        for field in fields:
            setattr(note, field, request.form.get(field, "").strip() or None)
        if not note.name:
            raise ValueError("Name is required")
        for field in ("year","date_start_year","date_end_year"):
            setattr(note, field, number(request.form.get(field), int))
        for field in ("estimated_value","purchase_price"):
            setattr(note, field, number(request.form.get(field), float))
        note.date_is_approx = request.form.get("date_is_approx") == "on"
        if note.date_start_year is not None and note.date_end_year is not None and note.date_start_year > note.date_end_year:
            raise ValueError("Start year must not exceed end year")

    def save_photos(note):
        # Store in the same R2-compatible static namespace as existing uploads.
        from r2_storage import r2_is_configured, r2_upload_file
        for idx, side in enumerate(("front","back")):
            file = request.files.get(side)
            if not file or not file.filename:
                continue
            suffix = Path(secure_filename(file.filename)).suffix.lower()
            if suffix not in (".jpg",".jpeg",".png",".webp"):
                raise ValueError("Photos must be JPG, PNG or WEBP")
            key = f"uploads/banknotes/{uuid.uuid4().hex}{suffix}"
            local = Path(app.static_folder) / key
            local.parent.mkdir(parents=True, exist_ok=True)
            file.save(local)
            if r2_is_configured():
                r2_upload_file(str(local), key)
            db.session.add(BanknotePhoto(banknote=note, filename=key, side=side, sort_order=idx))

    @bp.route("/banknotes")
    def list_notes():
        return render_template("banknotes.html", notes=Banknote.query.order_by(Banknote.id.desc()).all())

    @bp.route("/banknotes/add", methods=["GET","POST"])
    def add_note():
        owner_required()
        note = Banknote()
        error = None
        if request.method == "POST":
            try:
                save_fields(note)
                db.session.add(note)
                save_photos(note)
                db.session.commit()
                return redirect(url_for("banknotes.list_notes"))
            except (ValueError, TypeError) as exc:
                db.session.rollback()
                error = str(exc)
        return render_template("banknote_form.html", note=note, error=error)

    @bp.route("/banknotes/<int:note_id>/edit", methods=["GET","POST"])
    def edit_note(note_id):
        owner_required()
        note = db.get_or_404(Banknote, note_id)
        error = None
        if request.method == "POST":
            try:
                save_fields(note)
                save_photos(note)
                db.session.commit()
                return redirect(url_for("banknotes.list_notes"))
            except (ValueError, TypeError) as exc:
                db.session.rollback()
                error = str(exc)
        return render_template("banknote_form.html", note=note, error=error)

    app.register_blueprint(bp)
    return Banknote
