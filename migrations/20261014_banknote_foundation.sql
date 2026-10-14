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
-- Default absence of rows means coin-only, preserving existing set behavior.
CREATE TABLE IF NOT EXISTS coin_set_allowed_type (
 set_id INTEGER NOT NULL REFERENCES coin_set(id),
 collectible_type VARCHAR(20) NOT NULL CHECK(collectible_type IN ('coin','banknote')),
 PRIMARY KEY (set_id, collectible_type)
);
-- Existing coin_set_memberships remain authoritative for coins.
CREATE TABLE IF NOT EXISTS banknote_set_memberships (
 set_id INTEGER NOT NULL REFERENCES coin_set(id),
 banknote_id INTEGER NOT NULL REFERENCES banknote(id),
 PRIMARY KEY(set_id,banknote_id)
);
-- Existing set_slot_selection remains authoritative for coin overrides.
CREATE TABLE IF NOT EXISTS banknote_set_slot_selection (
 set_id INTEGER NOT NULL REFERENCES coin_set(id),
 slot_type VARCHAR(30) NOT NULL,
 slot_id INTEGER NOT NULL,
 banknote_id INTEGER NOT NULL REFERENCES banknote(id),
 PRIMARY KEY(set_id,slot_type,slot_id)
);
