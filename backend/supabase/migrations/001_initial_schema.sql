-- Applied via Supabase MCP on 2026-04-29
-- Run: supabase db push  OR  paste into Supabase SQL editor

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE documents (
    document_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        TEXT NOT NULL,
    sha256_hash     TEXT NOT NULL UNIQUE,
    metadata        JSONB NOT NULL DEFAULT '{}',
    doc_metadata    JSONB,
    page_count      INTEGER,
    parser_provider TEXT,
    parser_warnings JSONB NOT NULL DEFAULT '[]',
    storage_path    TEXT NOT NULL,
    corpus_version  INTEGER NOT NULL DEFAULT 1,
    tenant_id       UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_documents_tenant     ON documents (tenant_id);
CREATE INDEX idx_documents_sha256     ON documents (sha256_hash);
CREATE INDEX idx_documents_created_at ON documents (created_at DESC);

CREATE TABLE ingestions (
    ingestion_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id   UUID REFERENCES documents(document_id) ON DELETE SET NULL,
    filename      TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending','processing','completed','failed')),
    stage         TEXT,
    error_code    TEXT,
    error_message TEXT,
    stats         JSONB NOT NULL DEFAULT '{}',
    quality_report JSONB,
    tenant_id     UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at    TIMESTAMPTZ,
    finished_at   TIMESTAMPTZ
);

CREATE INDEX idx_ingestions_document ON ingestions (document_id);
CREATE INDEX idx_ingestions_status   ON ingestions (status);
CREATE INDEX idx_ingestions_tenant   ON ingestions (tenant_id);

CREATE TABLE parent_chunks (
    chunk_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id   UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    text          TEXT NOT NULL,
    section_path  TEXT,
    token_count   INTEGER,
    tenant_id     UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_parent_chunks_document ON parent_chunks (document_id);

CREATE TABLE child_chunks (
    chunk_id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_chunk_id         UUID NOT NULL REFERENCES parent_chunks(chunk_id) ON DELETE CASCADE,
    document_id             UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index             INTEGER NOT NULL,
    text                    TEXT NOT NULL,
    contextualized_text     TEXT,
    section_type            TEXT,
    recommendation_strength TEXT,
    entities                JSONB NOT NULL DEFAULT '{}',
    qdrant_point_id         UUID,
    token_count             INTEGER,
    tenant_id               UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_child_chunks_parent   ON child_chunks (parent_chunk_id);
CREATE INDEX idx_child_chunks_document ON child_chunks (document_id);

CREATE TABLE chat_sessions (
    session_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
    metadata    JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at  TIMESTAMPTZ
);

CREATE TABLE chat_messages (
    message_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user','assistant')),
    content     TEXT NOT NULL,
    citations   JSONB NOT NULL DEFAULT '[]',
    abstained   BOOLEAN NOT NULL DEFAULT FALSE,
    confidence  NUMERIC(4,3),
    feedback    INTEGER CHECK (feedback IN (-1, 1)),
    tenant_id   UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_chat_messages_session ON chat_messages (session_id, created_at);

CREATE TABLE corpus_config (
    tenant_id       UUID PRIMARY KEY DEFAULT '00000000-0000-0000-0000-000000000000',
    corpus_version  INTEGER NOT NULL DEFAULT 1,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO corpus_config (tenant_id, corpus_version)
VALUES ('00000000-0000-0000-0000-000000000000', 1);

CREATE VIEW documents_with_status AS
SELECT
    d.*,
    i.ingestion_id     AS latest_ingestion_id,
    i.status           AS latest_status,
    i.stage            AS latest_stage,
    i.error_code       AS latest_error_code,
    i.error_message    AS latest_error_message,
    i.created_at       AS latest_created_at,
    i.finished_at      AS latest_finished_at
FROM documents d
LEFT JOIN LATERAL (
    SELECT * FROM ingestions
    WHERE document_id = d.document_id
    ORDER BY created_at DESC
    LIMIT 1
) i ON TRUE;

-- Migration 002
CREATE OR REPLACE FUNCTION increment_corpus_version(p_tenant_id UUID)
RETURNS void LANGUAGE plpgsql AS $$
BEGIN
  UPDATE corpus_config
     SET corpus_version = corpus_version + 1,
         updated_at     = NOW()
   WHERE tenant_id = p_tenant_id;
END;
$$;
