-- Phase 1: additive banknote schema. Run only after backing up the active database.
-- Existing coin, set, slot-selection and artifact tables are not modified.
CREATE TABLE IF NOT EXISTS banknote (
 id INTEGER PRIMARY KEY,
 name VARCHAR(180) NOT NULL,
 country VARCHAR(120),
 set_country VARCHAR(120),
 denomination VARCHAR(100),
 currency VARCHAR(100),
 year INTEGER,
 date_start_year INTEGER,
 date_end_year INTEGER,
 date_is_approx BOOLEAN NOT NULL DEFAULT 0,
 issuing_authority VARCHAR(180),
 serial_number VARCHAR(120),
 catalog_number VARCHAR(120),
 portrait VARCHAR(180),
 reverse_design VARCHAR(180),
 condition VARCHAR(100),
 estimated_value FLOAT,
 purchase_price FLOAT,
 date_acquired VARCHAR(30),
 source VARCHAR(180),
 notes TEXT,
 created_at DATETIME
);
CREATE TABLE IF NOT EXISTS banknote_photo (
 id INTEGER PRIMARY KEY,
 banknote_id INTEGER NOT NULL REFERENCES banknote(id),
 filename VARCHAR(255) NOT NULL,
 side VARCHAR(20),
 sort_order INTEGER NOT NULL DEFAULT 0,
 created_at DATETIME
);
CREATE INDEX IF NOT EXISTS ix_banknote_photo_banknote_id ON banknote_photo(banknote_id);
