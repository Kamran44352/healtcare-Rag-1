"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Globe,
  RefreshCcw,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  deleteDocument,
  getBulkCrawlStatus,
  getIngestion,
  listBulkCrawlBatches,
  listCrawls,
  submitBulkCrawl,
  submitCrawl,
} from "@/lib/api";
import type { CrawlBatchItemStatus, CrawlBatchStatus, CrawlBatchSummary, DocumentRecord } from "@/lib/types";

const INTERVAL_OPTIONS: { label: string; hours: number }[] = [
  { label: "Off (no auto re-scrape)", hours: 0 },
  { label: "Daily", hours: 24 },
  { label: "Weekly", hours: 168 },
  { label: "Monthly", hours: 720 },
];

// Persists the in-flight bulk-crawl batch id across page reloads/tab closes —
// the server owns the batch durably, this is just so the UI can find it again.
const ACTIVE_BATCH_STORAGE_KEY = "clintel:active_crawl_batch_id";

function statusVariant(status: string | null) {
  if (status === "completed") return "success" as const;
  if (status === "failed") return "destructive" as const;
  if (status === "processing") return "warning" as const;
  return "secondary" as const;
}

function stageLabel(stage: string | null) {
  const map: Record<string, string> = {
    queued: "Queued",
    scraping: "Scraping page…",
    quality_check: "Quality Check…",
    extracting_metadata: "Extracting Metadata…",
    chunking: "Chunking…",
    enriching_chunks: "Enriching Chunks…",
    contextualizing: "Contextualizing…",
    embedding: "Embedding…",
    indexing: "Indexing…",
    finalising: "Finalising…",
    completed: "Done",
    deduped: "Unchanged (skipped)",
  };
  return stage ? (map[stage] ?? stage) : "—";
}

function normalizeList<T>(payload: unknown): T[] {
  if (Array.isArray(payload)) return payload as T[];
  if (payload && typeof payload === "object" && Array.isArray((payload as { items?: unknown[] }).items))
    return (payload as { items: T[] }).items;
  return [];
}

function isTerminal(status: string | null) {
  return status === "completed" || status === "failed";
}

function fmt(dt: string | null) {
  return dt ? new Date(dt).toLocaleString() : "—";
}

function MetaRow({ label, value }: { label: string; value: string }) {
  if (!value || value === "null" || value === "unknown") return null;
  return (
    <div className="flex min-w-0 gap-3">
      <span className="w-32 shrink-0 text-xs text-muted-foreground">{label}</span>
      <span className="min-w-0 break-words text-xs font-medium text-foreground">{value}</span>
    </div>
  );
}

// ── Shared state hook (lets the form and list cards live in separate slots) ──

export type CrawlController = ReturnType<typeof useCrawls>;

export function useCrawls(baseUrl: string) {
  const [url, setUrl] = useState("");
  const [intervalHours, setIntervalHours] = useState<number>(24);
  const [submitting, setSubmitting] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [crawls, setCrawls] = useState<Record<string, DocumentRecord>>({});
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const [bulkUrls, setBulkUrls] = useState("");
  const [bulkSubmitting, setBulkSubmitting] = useState(false);
  const [bulkProgress, setBulkProgress] = useState<{ current: number; total: number; errors: string[] } | null>(null);
  const bulkPollRef = useRef<number | null>(null);

  const sortedCrawls = useMemo(
    () => Object.values(crawls).sort((a, b) => b.created_at.localeCompare(a.created_at)),
    [crawls]
  );

  const refresh = async () => {
    if (!baseUrl.trim()) return;
    setRefreshing(true);
    try {
      const payload = await listCrawls(baseUrl);
      const docs = normalizeList<DocumentRecord>(payload);
      const map: Record<string, DocumentRecord> = {};
      for (const d of docs) if (d.document_id) map[d.document_id] = d;
      setCrawls(map);

      const anyActive = docs.some((d) => !isTerminal(d.latest_status));
      if (anyActive && pollRef.current === null) {
        pollRef.current = window.setInterval(() => void refresh(), 3000);
      } else if (!anyActive && pollRef.current !== null) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    } catch (err) {
      toast.error(`Failed to load web sources: ${(err as Error).message}`);
    } finally {
      setRefreshing(false);
      setInitialLoading(false);
    }
  };

  const stopBulkPolling = () => {
    if (bulkPollRef.current !== null) {
      window.clearInterval(bulkPollRef.current);
      bulkPollRef.current = null;
    }
  };

  const clearActiveBatch = () => {
    try {
      window.localStorage.removeItem(ACTIVE_BATCH_STORAGE_KEY);
    } catch {
      // localStorage may be unavailable (private mode, etc.) — non-fatal.
    }
  };

  const pollBulkBatch = async (batchId: string) => {
    let status: CrawlBatchStatus;
    try {
      status = await getBulkCrawlStatus(baseUrl, batchId);
    } catch {
      // Batch no longer resolvable (deleted, wrong backend, etc.) — stop chasing it.
      stopBulkPolling();
      clearActiveBatch();
      setBulkSubmitting(false);
      setBulkProgress(null);
      return;
    }

    setBulkProgress({
      current: status.completed_count + status.failed_count,
      total: status.total_count,
      errors: status.items
        .filter((i) => i.status === "failed")
        .map((i) => `${i.url}: ${i.error_message ?? i.error_code ?? "failed"}`),
    });

    if (status.is_finished) {
      stopBulkPolling();
      clearActiveBatch();
      setBulkSubmitting(false);
      void refresh();
      if (status.failed_count === 0) {
        toast.success(`All ${status.total_count} URL${status.total_count > 1 ? "s" : ""} indexed successfully`);
      } else if (status.completed_count > 0) {
        toast.warning(`${status.completed_count}/${status.total_count} URLs indexed — ${status.failed_count} failed`);
      } else {
        toast.error(`All ${status.total_count} URLs failed to index`);
      }
      setBulkProgress(null);
    }
  };

  const startBulkPolling = (batchId: string) => {
    setBulkSubmitting(true);
    stopBulkPolling();
    void pollBulkBatch(batchId);
    bulkPollRef.current = window.setInterval(() => void pollBulkBatch(batchId), 3000);
  };

  useEffect(() => {
    void refresh();

    try {
      const activeBatchId = window.localStorage.getItem(ACTIVE_BATCH_STORAGE_KEY);
      if (activeBatchId) startBulkPolling(activeBatchId);
    } catch {
      // localStorage may be unavailable — resume just won't happen this session.
    }

    return () => {
      if (pollRef.current !== null) window.clearInterval(pollRef.current);
      pollRef.current = null;
      stopBulkPolling();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseUrl]);

  const trackJob = async (ingestionId: string) => {
    try {
      const status = await getIngestion(baseUrl, ingestionId);
      if (!isTerminal(status.status)) {
        window.setTimeout(() => void trackJob(ingestionId), 3000);
      } else {
        void refresh();
      }
    } catch {
      void refresh();
    }
  };

  const submit = async () => {
    if (!url.trim() || submitting) return;
    setSubmitting(true);
    try {
      const created = await submitCrawl(
        baseUrl,
        url.trim(),
        { category: null, reference: null, guideline_title: null, pdf_url: null },
        false,
        intervalHours
      );
      toast.success(
        created.reused_existing_document
          ? "Page unchanged since last scrape (skipped)"
          : "Crawl started"
      );
      setUrl("");
      void refresh();
      if (!isTerminal(created.status)) void trackJob(created.ingestion_id);
    } catch (err) {
      toast.error(`Crawl failed: ${(err as Error).message}`);
    } finally {
      setSubmitting(false);
    }
  };

  const onRescrape = async (doc: DocumentRecord) => {
    if (!doc.source_url) return;
    setBusyId(doc.document_id);
    try {
      const created = await submitCrawl(
        baseUrl,
        doc.source_url,
        { category: null, reference: null, guideline_title: null, pdf_url: null },
        true, // force — user explicitly asked to re-scrape
        doc.rescrape_interval_hours
      );
      toast.success(
        created.reused_existing_document ? "Page unchanged (no re-index needed)" : "Re-scrape started"
      );
      void refresh();
      if (!isTerminal(created.status)) void trackJob(created.ingestion_id);
    } catch (err) {
      toast.error(`Re-scrape failed: ${(err as Error).message}`);
    } finally {
      setBusyId(null);
    }
  };

  const submitBulk = async () => {
    const urls = bulkUrls
      .split(/[\n,]+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    if (!urls.length || bulkSubmitting) return;

    setBulkSubmitting(true);
    try {
      // One request durably persists the whole list server-side — the browser
      // can close immediately after this resolves; the server owns the rest.
      const created = await submitBulkCrawl(
        baseUrl,
        urls,
        { category: null, reference: null, guideline_title: null, pdf_url: null },
        intervalHours
      );
      try {
        window.localStorage.setItem(ACTIVE_BATCH_STORAGE_KEY, created.batch_id);
      } catch {
        // Non-fatal — the batch still processes server-side, just won't resume
        // in the UI after a reload.
      }
      const parts = [`${created.accepted_count} URL${created.accepted_count === 1 ? "" : "s"} queued`];
      if (created.duplicate_count > 0) parts.push(`${created.duplicate_count} duplicate(s) skipped`);
      toast.success(parts.join(", "));
      setBulkUrls("");
      startBulkPolling(created.batch_id);
    } catch (err) {
      toast.error(`Bulk crawl submission failed: ${(err as Error).message}`);
      setBulkSubmitting(false);
    }
  };

  const onDelete = async (doc: DocumentRecord) => {
    if (!window.confirm(`Delete "${doc.filename}"?\n\nThis removes the page, all chunks, and Qdrant vectors.`)) return;
    setBusyId(doc.document_id);
    try {
      await deleteDocument(baseUrl, doc.document_id);
      setCrawls((prev) => {
        const n = { ...prev };
        delete n[doc.document_id];
        return n;
      });
      if (expandedId === doc.document_id) setExpandedId(null);
      toast.success("Web source deleted");
    } catch (err) {
      toast.error(`Delete failed: ${(err as Error).message}`);
    } finally {
      setBusyId(null);
    }
  };

  return {
    url, setUrl, intervalHours, setIntervalHours, submitting, submit,
    refreshing, initialLoading, refresh,
    sortedCrawls, busyId, expandedId, setExpandedId, onRescrape, onDelete,
    bulkUrls, setBulkUrls, bulkSubmitting, bulkProgress, submitBulk,
  };
}

// ── Form card (place below "Upload Document") ───────────────────────────────

export function CrawlForm({ crawl }: { crawl: CrawlController }) {
  const {
    url, setUrl, intervalHours, setIntervalHours, submitting, submit,
    bulkUrls, setBulkUrls, bulkSubmitting, bulkProgress, submitBulk,
  } = crawl;

  const [mode, setMode] = useState<"single" | "bulk">("single");

  return (
    <Card className="border-primary/20">
      <CardHeader className="border-b border-border/50 pb-4">
        <CardTitle className="text-base font-semibold text-primary">Crawl Website</CardTitle>
        <CardDescription>
          Scrape web pages — content is chunked, embedded, and indexed just like an uploaded PDF.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-5 space-y-4">
        {/* Mode toggle */}
        <div className="flex gap-1 rounded-lg bg-secondary/60 p-1 w-fit">
          {(["single", "bulk"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                mode === m
                  ? "bg-background shadow text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {m === "single" ? "Single URL" : "Multiple URLs"}
            </button>
          ))}
        </div>

        {mode === "single" ? (
          <>
            <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
              <div className="flex-1 space-y-1.5">
                <Label htmlFor="crawl-url">
                  Page URL <span className="text-destructive">*</span>
                </Label>
                <Input
                  id="crawl-url"
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://www.nice.org.uk/guidance/ng136/chapter/recommendations"
                  className="font-mono text-sm"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="crawl-interval">Auto re-scrape</Label>
                <select
                  id="crawl-interval"
                  value={intervalHours}
                  onChange={(e) => setIntervalHours(Number(e.target.value))}
                  className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm sm:w-48"
                >
                  {INTERVAL_OPTIONS.map((o) => (
                    <option key={o.hours} value={o.hours}>{o.label}</option>
                  ))}
                </select>
              </div>
            </div>
            <Button
              onClick={() => void submit()}
              disabled={!url.trim() || submitting}
              className="bg-accent hover:bg-accent/90 text-white"
            >
              <Globe className="h-4 w-4" />
              {submitting ? "Crawling…" : "Crawl Page"}
            </Button>
          </>
        ) : (
          <>
            <div className="flex flex-col gap-4 sm:flex-row sm:items-start">
              <div className="flex-1 space-y-1.5">
                <Label htmlFor="crawl-bulk-urls">
                  URLs <span className="text-destructive">*</span>
                  <span className="ml-1 text-xs font-normal text-muted-foreground">
                    (comma or newline separated)
                  </span>
                </Label>
                <textarea
                  id="crawl-bulk-urls"
                  value={bulkUrls}
                  onChange={(e) => setBulkUrls(e.target.value)}
                  disabled={bulkSubmitting}
                  placeholder={
                    "https://example.com/page1,\nhttps://example.com/page2,\nhttps://example.com/page3"
                  }
                  rows={5}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring resize-y disabled:opacity-50"
                />
                <p className="text-xs text-muted-foreground">
                  {bulkUrls.trim()
                    ? `${bulkUrls.split(/[\n,]+/).map((s) => s.trim()).filter(Boolean).length} URL(s) detected`
                    : "Each URL is scraped and indexed one by one."}
                </p>
              </div>
              <div className="space-y-1.5 sm:w-48">
                <Label htmlFor="crawl-bulk-interval">Auto re-scrape</Label>
                <select
                  id="crawl-bulk-interval"
                  value={intervalHours}
                  onChange={(e) => setIntervalHours(Number(e.target.value))}
                  className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                >
                  {INTERVAL_OPTIONS.map((o) => (
                    <option key={o.hours} value={o.hours}>{o.label}</option>
                  ))}
                </select>
              </div>
            </div>

            {bulkProgress && (
              <div className="rounded-md border border-border/50 bg-secondary/30 px-4 py-3 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-foreground">
                    Processing {bulkProgress.current} of {bulkProgress.total}…
                  </span>
                  <span className="text-muted-foreground">
                    {bulkProgress.errors.length > 0 && `${bulkProgress.errors.length} failed`}
                  </span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-border overflow-hidden">
                  <div
                    className="h-full rounded-full bg-accent transition-all duration-300"
                    style={{ width: `${(bulkProgress.current / bulkProgress.total) * 100}%` }}
                  />
                </div>
              </div>
            )}

            <Button
              onClick={() => void submitBulk()}
              disabled={
                !bulkUrls.trim() ||
                bulkSubmitting ||
                bulkUrls.split(/[\n,]+/).map((s) => s.trim()).filter(Boolean).length === 0
              }
              className="bg-accent hover:bg-accent/90 text-white"
            >
              <Globe className="h-4 w-4" />
              {bulkSubmitting
                ? bulkProgress
                  ? `Crawling ${bulkProgress.current}/${bulkProgress.total}…`
                  : "Starting…"
                : "Crawl All URLs"}
            </Button>
          </>
        )}
      </CardContent>
    </Card>
  );
}

// ── List card (place below "Documents Library") ─────────────────────────────

export function CrawlList({ crawl }: { crawl: CrawlController }) {
  const {
    refreshing, initialLoading, refresh,
    sortedCrawls, busyId, expandedId, setExpandedId, onRescrape, onDelete,
  } = crawl;

  return (
    <Card className="border-primary/20">
      <CardHeader className="border-b border-border/50 pb-4">
        <div className="flex items-center justify-between gap-4">
          <div>
            <CardTitle className="text-base font-semibold text-primary">
              Website Crawls
              {sortedCrawls.length > 0 && (
                <span className="ml-2 rounded-full bg-primary/10 px-2 py-0.5 text-xs font-normal text-primary">
                  {sortedCrawls.length}
                </span>
              )}
            </CardTitle>
            <CardDescription>Crawled pages with live status and re-scrape scheduling.</CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={() => void refresh()} disabled={refreshing}>
            <RefreshCcw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
            {refreshing ? "Refreshing…" : "Refresh"}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {!initialLoading && !sortedCrawls.length ? (
          <div className="flex flex-col items-center gap-2 py-16 text-center">
            <Globe className="h-10 w-10 text-muted-foreground/40" />
            <p className="text-sm font-medium text-muted-foreground">No crawled pages yet.</p>
            <p className="text-xs text-muted-foreground">Crawl a URL above to start.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-sm">
              <thead>
                <tr className="border-b border-border/60 bg-primary/5 text-left">
                  <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wide text-primary">Page</th>
                  <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-primary">Status</th>
                  <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-primary">Stage</th>
                  <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-primary">Next re-scrape</th>
                  <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-primary">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/40">
                {sortedCrawls.map((doc) => {
                  const isExpanded = expandedId === doc.document_id;
                  const status = doc.latest_status;
                  return (
                    <>
                      <tr
                        key={doc.document_id}
                        className={`cursor-pointer transition-colors hover:bg-secondary/40 ${isExpanded ? "bg-secondary/50" : ""}`}
                        onClick={() => setExpandedId((p) => (p === doc.document_id ? null : doc.document_id))}
                      >
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2.5">
                            {isExpanded ? (
                              <ChevronDown className="h-3.5 w-3.5 shrink-0 text-primary" />
                            ) : (
                              <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                            )}
                            <Globe className="h-4 w-4 shrink-0 text-muted-foreground" />
                            <span className="max-w-[260px] truncate font-medium" title={doc.filename}>
                              {doc.filename}
                            </span>
                          </div>
                        </td>
                        <td className="px-3 py-3">
                          <Badge variant={statusVariant(status)} className="text-xs">
                            {status ?? "unknown"}
                          </Badge>
                        </td>
                        <td className="px-3 py-3 text-xs text-foreground/80">{stageLabel(doc.latest_stage)}</td>
                        <td className="px-3 py-3 text-xs text-muted-foreground whitespace-nowrap">
                          {doc.next_rescrape_at ? new Date(doc.next_rescrape_at).toLocaleDateString() : "Off"}
                        </td>
                        <td className="px-3 py-3" onClick={(e) => e.stopPropagation()}>
                          <div className="flex gap-2">
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-7 px-2"
                              onClick={() => void onRescrape(doc)}
                              disabled={busyId === doc.document_id || !isTerminal(status)}
                            >
                              <RefreshCcw className="h-3.5 w-3.5" />
                              Re-scrape
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-7 border-destructive/30 px-2 text-destructive hover:bg-destructive/5 hover:border-destructive/60"
                              onClick={() => void onDelete(doc)}
                              disabled={busyId === doc.document_id}
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </Button>
                          </div>
                        </td>
                      </tr>

                      {isExpanded && (
                        <tr key={`${doc.document_id}-detail`} className="bg-secondary/20">
                          <td colSpan={5} className="px-4 py-5">
                            <div className="grid gap-5 sm:grid-cols-2">
                              <div className="space-y-2">
                                <p className="text-xs font-semibold uppercase tracking-wide text-primary">Source</p>
                                <MetaRow label="Title" value={doc.filename} />
                                <div className="flex min-w-0 gap-3">
                                  <span className="w-32 shrink-0 text-xs text-muted-foreground">URL</span>
                                  {doc.source_url ? (
                                    <a
                                      href={doc.source_url}
                                      target="_blank"
                                      rel="noreferrer"
                                      className="min-w-0 break-all text-xs font-medium text-accent underline"
                                    >
                                      {doc.source_url}
                                    </a>
                                  ) : (
                                    <span className="text-xs">—</span>
                                  )}
                                </div>
                                <MetaRow label="Status" value={doc.latest_status ?? "—"} />
                                <MetaRow label="Stage" value={stageLabel(doc.latest_stage)} />
                                {doc.latest_error_message && (
                                  <div>
                                    <p className="mb-0.5 text-xs text-muted-foreground">Error</p>
                                    <p className="text-xs text-destructive">
                                      [{doc.latest_error_code}] {doc.latest_error_message}
                                    </p>
                                  </div>
                                )}
                                <MetaRow label="Document ID" value={doc.document_id} />
                              </div>
                              <div className="space-y-2">
                                <p className="text-xs font-semibold uppercase tracking-wide text-primary">Re-scrape schedule</p>
                                <MetaRow
                                  label="Cadence"
                                  value={
                                    doc.rescrape_interval_hours && doc.rescrape_interval_hours > 0
                                      ? `Every ${doc.rescrape_interval_hours}h`
                                      : "Off"
                                  }
                                />
                                <MetaRow label="Last scraped" value={fmt(doc.last_rescrape_at)} />
                                <MetaRow label="Next scrape" value={fmt(doc.next_rescrape_at)} />
                                <MetaRow label="First crawled" value={fmt(doc.created_at)} />
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ── Bulk import history (past "Multiple URLs" batches) ───────────────────────

function bulkStatusVariant(status: CrawlBatchSummary) {
  if (!status.is_finished) return "warning" as const;
  if (status.failed_count > 0 && status.completed_count === 0) return "destructive" as const;
  if (status.failed_count > 0) return "warning" as const;
  return "success" as const;
}

function bulkStatusLabel(status: CrawlBatchSummary) {
  if (!status.is_finished) return `Processing ${status.completed_count + status.failed_count}/${status.total_count}`;
  if (status.failed_count === 0) return "Completed";
  if (status.completed_count === 0) return "Failed";
  return "Completed with errors";
}

export type BulkBatchHistoryController = ReturnType<typeof useBulkBatchHistory>;

export function useBulkBatchHistory(baseUrl: string) {
  const [batches, setBatches] = useState<CrawlBatchSummary[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [items, setItems] = useState<Record<string, CrawlBatchItemStatus[]>>({});
  const [itemsLoading, setItemsLoading] = useState<string | null>(null);

  const refresh = async () => {
    if (!baseUrl.trim()) return;
    setRefreshing(true);
    try {
      const list = await listBulkCrawlBatches(baseUrl);
      setBatches(Array.isArray(list) ? list : []);
    } catch (err) {
      toast.error(`Failed to load bulk import history: ${(err as Error).message}`);
    } finally {
      setRefreshing(false);
      setInitialLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseUrl]);

  const toggleExpand = async (batchId: string) => {
    if (expandedId === batchId) {
      setExpandedId(null);
      return;
    }
    setExpandedId(batchId);
    if (items[batchId]) return; // already loaded — lazy-load once per batch
    setItemsLoading(batchId);
    try {
      const status = await getBulkCrawlStatus(baseUrl, batchId);
      setItems((prev) => ({ ...prev, [batchId]: status.items }));
    } catch (err) {
      toast.error(`Failed to load batch detail: ${(err as Error).message}`);
    } finally {
      setItemsLoading(null);
    }
  };

  return { batches, refreshing, initialLoading, refresh, expandedId, toggleExpand, items, itemsLoading };
}

export function BulkBatchHistory({ history }: { history: BulkBatchHistoryController }) {
  const { batches, refreshing, initialLoading, refresh, expandedId, toggleExpand, items, itemsLoading } = history;

  return (
    <Card className="border-primary/20">
      <CardHeader className="border-b border-border/50 pb-4">
        <div className="flex items-center justify-between gap-4">
          <div>
            <CardTitle className="text-base font-semibold text-primary">
              Bulk Import History
              {batches.length > 0 && (
                <span className="ml-2 rounded-full bg-primary/10 px-2 py-0.5 text-xs font-normal text-primary">
                  {batches.length}
                </span>
              )}
            </CardTitle>
            <CardDescription>Past "Multiple URLs" submissions and their outcomes.</CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={() => void refresh()} disabled={refreshing}>
            <RefreshCcw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
            {refreshing ? "Refreshing…" : "Refresh"}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {!initialLoading && !batches.length ? (
          <div className="flex flex-col items-center gap-2 py-16 text-center">
            <Globe className="h-10 w-10 text-muted-foreground/40" />
            <p className="text-sm font-medium text-muted-foreground">No bulk imports yet.</p>
            <p className="text-xs text-muted-foreground">Paste a list of URLs above under "Multiple URLs" to start.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-sm">
              <thead>
                <tr className="border-b border-border/60 bg-primary/5 text-left">
                  <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wide text-primary">Submitted</th>
                  <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-primary">Status</th>
                  <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-primary">Total</th>
                  <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-primary">Completed</th>
                  <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-primary">Failed</th>
                  <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-primary">Duplicates</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/40">
                {batches.map((batch) => {
                  const isExpanded = expandedId === batch.batch_id;
                  return (
                    <>
                      <tr
                        key={batch.batch_id}
                        className={`cursor-pointer transition-colors hover:bg-secondary/40 ${isExpanded ? "bg-secondary/50" : ""}`}
                        onClick={() => void toggleExpand(batch.batch_id)}
                      >
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2.5">
                            {isExpanded ? (
                              <ChevronDown className="h-3.5 w-3.5 shrink-0 text-primary" />
                            ) : (
                              <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                            )}
                            <span className="text-xs font-medium">{fmt(batch.created_at)}</span>
                          </div>
                        </td>
                        <td className="px-3 py-3">
                          <Badge variant={bulkStatusVariant(batch)} className="text-xs">
                            {bulkStatusLabel(batch)}
                          </Badge>
                        </td>
                        <td className="px-3 py-3 text-xs text-foreground/80">{batch.total_count}</td>
                        <td className="px-3 py-3 text-xs text-foreground/80">{batch.completed_count}</td>
                        <td className="px-3 py-3 text-xs text-foreground/80">{batch.failed_count}</td>
                        <td className="px-3 py-3 text-xs text-foreground/80">{batch.duplicate_count}</td>
                      </tr>

                      {isExpanded && (
                        <tr key={`${batch.batch_id}-detail`} className="bg-secondary/20">
                          <td colSpan={6} className="px-4 py-5">
                            {itemsLoading === batch.batch_id ? (
                              <p className="text-xs text-muted-foreground">Loading URLs…</p>
                            ) : (
                              <div className="space-y-2">
                                {(items[batch.batch_id] ?? []).map((item) => (
                                  <div key={item.item_id} className="flex min-w-0 items-start gap-3 border-b border-border/30 pb-2 last:border-0">
                                    <Badge variant={statusVariant(item.status)} className="mt-0.5 shrink-0 text-xs">
                                      {item.status}
                                    </Badge>
                                    <div className="min-w-0 flex-1">
                                      <p className="break-all text-xs font-medium text-foreground">{item.url}</p>
                                      {item.error_message && (
                                        <p className="text-xs text-destructive">
                                          [{item.error_code}] {item.error_message}
                                          {item.max_attempts > 1 && ` (attempt ${item.attempt_count}/${item.max_attempts})`}
                                        </p>
                                      )}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}
                          </td>
                        </tr>
                      )}
                    </>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
