CREATE TABLE IF NOT EXISTS octa_jobs (
    id UUID PRIMARY KEY,
    owner_id UUID NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('video', 'image', 'research')),
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    provider_job_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_octa_jobs_owner_created ON octa_jobs (owner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_octa_jobs_status ON octa_jobs (status);
