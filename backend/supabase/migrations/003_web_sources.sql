-- Applied via Supabase MCP on 2026-06-10
-- Web source support: crawled pages stored as documents with re-scrape scheduling.

ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'pdf';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_url TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS rescrape_interval_hours INTEGER;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS last_rescrape_at TIMESTAMPTZ;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS next_rescrape_at TIMESTAMPTZ;

-- One document per URL per tenant (enables in-place re-scrape lookups).
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_source_url
    ON documents (tenant_id, source_url) WHERE source_url IS NOT NULL;

-- Fast lookup for the scheduled re-scrape loop.
CREATE INDEX IF NOT EXISTS idx_documents_next_rescrape
    ON documents (next_rescrape_at) WHERE source_type = 'url';

-- Recreate the status view so d.* picks up the new columns
-- (CREATE OR REPLACE cannot add columns to a view's output, so DROP + CREATE).
DROP VIEW IF EXISTS documents_with_status;
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
