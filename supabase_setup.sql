-- ============================================================
-- Attorney.AI — Supabase / pgvector Schema Migration
-- Run this once in your Supabase SQL Editor
-- https://supabase.com → SQL Editor → New Query → Paste & Run
-- ============================================================

-- 1. Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Main legal chunks table
--    Change vector(1024) if you use a different embedding model:
--      BGE-large-en-v1.5  → vector(1024)   ← default
--      legal-bert          → vector(768)
--      MiniLM-L6-v2        → vector(384)
--      OpenAI 3-large      → vector(3072)
CREATE TABLE IF NOT EXISTS legal_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id        TEXT UNIQUE NOT NULL,
    doc_id          TEXT NOT NULL,

    -- Display
    title           TEXT NOT NULL DEFAULT '',
    citation        TEXT NOT NULL DEFAULT '',
    source_url      TEXT NOT NULL DEFAULT '',

    -- Jurisdiction / court
    jurisdiction    TEXT NOT NULL DEFAULT 'US-Federal',
    court_or_agency TEXT,
    court_level     TEXT,

    -- Date
    decision_date   DATE,
    date_str        TEXT,

    -- Classification
    source_type     TEXT NOT NULL DEFAULT 'case',   -- case|statute|regulation|contract|filing
    practice_area   TEXT,

    -- Document structure
    parent_section  TEXT,
    start_char      INTEGER DEFAULT 0,
    end_char        INTEGER DEFAULT 0,

    -- Content
    text            TEXT NOT NULL DEFAULT '',

    -- Extra
    docket_number   TEXT,
    author_judge    TEXT,

    -- Vector embedding (change dimension to match your model)
    embedding       vector(1024),

    -- Full-text search column (auto-maintained by trigger)
    fts             tsvector GENERATED ALWAYS AS (
                        to_tsvector('english',
                            coalesce(title, '') || ' ' ||
                            coalesce(citation, '') || ' ' ||
                            coalesce(text, '')
                        )
                    ) STORED,

    -- Timestamps
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- 3. Indexes for performance
-- Vector similarity index (IVFFlat — good for up to ~1M rows)
CREATE INDEX IF NOT EXISTS legal_chunks_embedding_idx
    ON legal_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Full-text search index
CREATE INDEX IF NOT EXISTS legal_chunks_fts_idx
    ON legal_chunks
    USING gin (fts);

-- Metadata filter indexes
CREATE INDEX IF NOT EXISTS legal_chunks_jurisdiction_idx ON legal_chunks (jurisdiction);
CREATE INDEX IF NOT EXISTS legal_chunks_source_type_idx  ON legal_chunks (source_type);
CREATE INDEX IF NOT EXISTS legal_chunks_court_level_idx  ON legal_chunks (court_level);
CREATE INDEX IF NOT EXISTS legal_chunks_doc_id_idx       ON legal_chunks (doc_id);
CREATE INDEX IF NOT EXISTS legal_chunks_decision_date_idx ON legal_chunks (decision_date);

-- 4. Vector similarity search function
--    Called from Python as: supabase.rpc('match_legal_chunks', {...})
CREATE OR REPLACE FUNCTION match_legal_chunks(
    query_embedding     vector(1024),
    match_count         INT             DEFAULT 20,
    filter_jurisdiction TEXT            DEFAULT NULL,
    filter_source_type  TEXT            DEFAULT NULL,
    filter_court_level  TEXT            DEFAULT NULL
)
RETURNS TABLE (
    id              UUID,
    chunk_id        TEXT,
    doc_id          TEXT,
    title           TEXT,
    citation        TEXT,
    source_url      TEXT,
    jurisdiction    TEXT,
    court_or_agency TEXT,
    court_level     TEXT,
    date_str        TEXT,
    decision_date   DATE,
    source_type     TEXT,
    parent_section  TEXT,
    start_char      INTEGER,
    end_char        INTEGER,
    text            TEXT,
    docket_number   TEXT,
    author_judge    TEXT,
    practice_area   TEXT,
    similarity      FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        lc.id,
        lc.chunk_id,
        lc.doc_id,
        lc.title,
        lc.citation,
        lc.source_url,
        lc.jurisdiction,
        lc.court_or_agency,
        lc.court_level,
        lc.date_str,
        lc.decision_date,
        lc.source_type,
        lc.parent_section,
        lc.start_char,
        lc.end_char,
        lc.text,
        lc.docket_number,
        lc.author_judge,
        lc.practice_area,
        1 - (lc.embedding <=> query_embedding) AS similarity
    FROM legal_chunks lc
    WHERE
        (filter_jurisdiction IS NULL OR lc.jurisdiction = filter_jurisdiction)
        AND (filter_source_type  IS NULL OR lc.source_type  = filter_source_type)
        AND (filter_court_level  IS NULL OR lc.court_level  = filter_court_level)
    ORDER BY lc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- 5. Full-text search function
CREATE OR REPLACE FUNCTION search_legal_chunks_fts(
    query_text          TEXT,
    match_count         INT     DEFAULT 20,
    filter_jurisdiction TEXT    DEFAULT NULL,
    filter_source_type  TEXT    DEFAULT NULL
)
RETURNS TABLE (
    chunk_id        TEXT,
    doc_id          TEXT,
    title           TEXT,
    citation        TEXT,
    source_url      TEXT,
    jurisdiction    TEXT,
    court_or_agency TEXT,
    court_level     TEXT,
    date_str        TEXT,
    source_type     TEXT,
    parent_section  TEXT,
    start_char      INTEGER,
    end_char        INTEGER,
    text            TEXT,
    docket_number   TEXT,
    author_judge    TEXT,
    practice_area   TEXT,
    rank            FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        lc.chunk_id,
        lc.doc_id,
        lc.title,
        lc.citation,
        lc.source_url,
        lc.jurisdiction,
        lc.court_or_agency,
        lc.court_level,
        lc.date_str,
        lc.source_type,
        lc.parent_section,
        lc.start_char,
        lc.end_char,
        lc.text,
        lc.docket_number,
        lc.author_judge,
        lc.practice_area,
        ts_rank_cd(lc.fts, plainto_tsquery('english', query_text)) AS rank
    FROM legal_chunks lc
    WHERE
        lc.fts @@ plainto_tsquery('english', query_text)
        AND (filter_jurisdiction IS NULL OR lc.jurisdiction = filter_jurisdiction)
        AND (filter_source_type  IS NULL OR lc.source_type  = filter_source_type)
    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;

-- 6. Enable Row Level Security (RLS) — service key bypasses this
ALTER TABLE legal_chunks ENABLE ROW LEVEL SECURITY;

-- Allow service role full access
CREATE POLICY "service_role_all" ON legal_chunks
    FOR ALL
    TO service_role
    USING (true);

-- Allow anonymous read-only (for frontend search)
CREATE POLICY "anon_read" ON legal_chunks
    FOR SELECT
    TO anon
    USING (true);

-- Done!
SELECT 'Attorney.AI schema created successfully.' AS status;
