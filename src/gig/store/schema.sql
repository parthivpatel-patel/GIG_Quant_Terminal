-- GIG research lake (DuckDB). No server. One file on disk.

CREATE TABLE IF NOT EXISTS bars (
    dt DATE NOT NULL,
    symbol VARCHAR NOT NULL,
    close DOUBLE,
    volume DOUBLE,
    dollar_volume DOUBLE,
    PRIMARY KEY (dt, symbol)
);

CREATE TABLE IF NOT EXISTS universe (
    symbol VARCHAR NOT NULL,
    sector VARCHAR,
    asof_date DATE NOT NULL,
    PRIMARY KEY (symbol, asof_date)
);

-- Current US listings (Nasdaq Trader). Not a CRSP PIT tape.
CREATE TABLE IF NOT EXISTS listings (
    symbol VARCHAR NOT NULL,
    yahoo_symbol VARCHAR NOT NULL,
    exchange VARCHAR,
    name VARCHAR,
    is_etf BOOLEAN,
    is_test BOOLEAN,
    cap_bucket VARCHAR,
    asof_date DATE NOT NULL,
    PRIMARY KEY (symbol, asof_date)
);

CREATE TABLE IF NOT EXISTS news (
    published TIMESTAMP NOT NULL,
    asof_date DATE NOT NULL,
    symbol VARCHAR NOT NULL,
    title VARCHAR,
    source VARCHAR,
    sentiment DOUBLE,
    url VARCHAR
);

CREATE TABLE IF NOT EXISTS factors (
    dt DATE NOT NULL,
    symbol VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    value DOUBLE,
    PRIMARY KEY (dt, symbol, name)
);

CREATE TABLE IF NOT EXISTS experiments (
    ts TIMESTAMP NOT NULL,
    name VARCHAR NOT NULL,
    config_hash VARCHAR,
    metrics VARCHAR
);

CREATE TABLE IF NOT EXISTS model_predictions (
    dt DATE NOT NULL,
    symbol VARCHAR NOT NULL,
    model VARCHAR NOT NULL,
    train_end DATE,
    score DOUBLE,
    PRIMARY KEY (dt, symbol, model)
);

CREATE TABLE IF NOT EXISTS macro (
    dt DATE NOT NULL,
    series_id VARCHAR NOT NULL,
    value DOUBLE,
    PRIMARY KEY (dt, series_id)
);

CREATE TABLE IF NOT EXISTS filings (
    filed_at TIMESTAMP NOT NULL,
    asof_date DATE NOT NULL,
    symbol VARCHAR NOT NULL,
    form VARCHAR,
    accession VARCHAR,
    title VARCHAR,
    sentiment DOUBLE
);

CREATE TABLE IF NOT EXISTS quotes (
    ts TIMESTAMP NOT NULL,
    symbol VARCHAR NOT NULL,
    bid DOUBLE,
    ask DOUBLE,
    last DOUBLE,
    source VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS calendar (
    event_date DATE NOT NULL,
    symbol VARCHAR NOT NULL,
    event_type VARCHAR NOT NULL,
    extra VARCHAR
);
