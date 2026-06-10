export type IngestionMetadata = {
  category: string | null;
  reference: string | null;
  guideline_title: string | null;
  pdf_url: string | null;
};

export type IngestionCreated = {
  ingestion_id: string;
  document_id: string;
  status: string;
  stage: string;
  reused_existing_document: boolean;
};

export type IngestionStatus = {
  ingestion_id: string;
  document_id: string;
  filename: string;
  status: string;
  stage: string;
  error_code: string | null;
  error_message: string | null;
  stats: Record<string, unknown>;
  quality_report: Record<string, unknown> | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
};

export type IngestionListResponse = {
  items: IngestionStatus[];
  limit: number;
  offset: number;
};

export type CrawlRequest = {
  url: string;
  metadata: IngestionMetadata;
  force_rescrape: boolean;
  rescrape_interval_hours: number | null;
};

export type DocumentRecord = {
  document_id: string;
  filename: string;
  metadata: Record<string, unknown>;
  doc_metadata: Record<string, unknown> | null;
  page_count: number | null;
  parser_provider: string | null;
  parser_warnings: unknown[];
  storage_path: string;
  source_type: string;
  source_url: string | null;
  rescrape_interval_hours: number | null;
  last_rescrape_at: string | null;
  next_rescrape_at: string | null;
  created_at: string;
  latest_ingestion_id: string | null;
  latest_status: string | null;
  latest_stage: string | null;
  latest_error_code: string | null;
  latest_error_message: string | null;
  latest_created_at: string | null;
  latest_finished_at: string | null;
};

export type DocumentListResponse = {
  items: DocumentRecord[];
  limit: number;
  offset: number;
};

export type DocumentDeleteResponse = {
  document_id: string;
  filename: string;
  deleted: boolean;
};

export type RetrievalFilters = {
  doc_type?: string[];
  section_type?: string[];
  specialties?: string[];
  conditions_codes?: string[];
  geographic_scope?: string[];
  care_setting?: string[];
  has_dosing_tables?: boolean;
  has_red_flags?: boolean;
};

export type RetrievalProfile = "specific" | "narrow" | "broad";

export type RetrievedChunk = {
  chunk_id: string;
  parent_chunk_id: string;
  document_id: string;
  filename: string;
  section_path: string;
  section_type: string | null;
  recommendation_strength: string | null;
  entities: Record<string, unknown>;
  snippet: string;
  full_snippet: string;
  parent_text: string | null;
  doc_metadata: Record<string, unknown>;
  dense_score: number | null;
  fused_score: number;
  rerank_score: number | null;
};

export type RetrievalSearchResponse = {
  chunks: RetrievedChunk[];
  query_ms: number;
  corpus_version: number;
  cache_hit: boolean;
};

export type Citation = {
  chunk_id: string;
  source_index: number;  // Original [SOURCE n] number from LLM — used to sync answer badges with citation box
  filename: string;
  section_path: string;
  snippet: string;
  full_snippet?: string;  // Full ~1200-char excerpt the LLM saw — exposed via "Show full source" toggle
  doc_metadata: Record<string, unknown>;
};

export type ChatResponse = {
  answer: string;
  abstained: boolean;
  abstain_reason: string | null;
  confidence: number;
  citations: Citation[];
  session_id: string;
  message_id: string;
  retrieval_debug: Record<string, unknown> | null;
  follow_up_questions?: string[];
};

export type ChatMessage = {
  id: string;           // local React key (streaming ID or user ID)
  messageId?: string;   // backend UUID — set after response arrives, used for API calls
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  abstained?: boolean;
  abstainReason?: string | null;
  followUpQuestions?: string[];
};

export type ChatHistoryResponse = {
  session_id: string;
  messages: Array<{
    message_id: string;
    session_id: string;
    role: "user" | "assistant";
    content: string;
    citations: Citation[];
    abstained: boolean;
    confidence: number | null;
    created_at: string;
  }>;
};

export type ChatSessionResponse = {
  session_id: string;
  metadata: Record<string, unknown>;
  created_at: string;
  expires_at: string | null;
};
