"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock3,
  FileText,
  FileUp,
  RefreshCcw,
  Trash2,
  XCircle,
} from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { deleteDocument, getIngestion, listDocuments, listIngestions, submitIngestion } from "@/lib/api";
import type { IngestionStatus } from "@/lib/types";

type IngestionPanelProps = { baseUrl: string };

type DocMeta = {
  title?: string;
  document_type?: string;
  issuing_body?: string;
  issuing_body_authority?: string;
  reference_id?: string;
  publication_date?: string;
  geographic_scope?: string[];
  specialties?: string[];
  conditions?: { label: string; code?: string }[];
  drugs?: { label: string; code?: string }[];
  care_setting?: string[];
  evidence_grading_system?: string;
  has_dosing_tables?: boolean;
  has_red_flags?: boolean;
  has_referral_criteria?: boolean;
  summary?: string;
};

type DocumentItem = {
  document_id: string;
  filename: string;
  metadata: Record<string, unknown>;
  doc_metadata: DocMeta | null;
  page_count: number | null;
  parser_provider: string | null;
  storage_path: string;
  created_at: string;
  latest_ingestion_id: string | null;
  latest_status: string | null;
  latest_stage: string | null;
  latest_finished_at: string | null;
};

type JobItem = IngestionStatus & { updatedAtText: string };

function statusVariant(status: string) {
  if (status === "completed") return "success" as const;
  if (status === "failed") return "destructive" as const;
  if (status === "processing") return "warning" as const;
  return "secondary" as const;
}

function stageLabel(stage: string | null) {
  const map: Record<string, string> = {
    queued: "Queued",
    parsing: "Parsing PDF…",
    quality_check: "Quality Check…",
    extracting_metadata: "Extracting Metadata…",
    chunking: "Chunking…",
    enriching_chunks: "Enriching Chunks…",
    contextualizing: "Contextualizing…",
    embedding: "Embedding…",
    indexing: "Indexing…",
    finalising: "Finalising…",
    completed: "Done",
    deduped: "Duplicate (skipped)",
  };
  return stage ? (map[stage] ?? stage) : "—";
}

function normalizeList<T>(payload: unknown): T[] {
  if (Array.isArray(payload)) return payload as T[];
  if (payload && typeof payload === "object" && Array.isArray((payload as { items?: unknown[] }).items))
    return (payload as { items: T[] }).items;
  return [];
}

function isTerminal(status: string) {
  return status === "completed" || status === "failed";
}

function Pill({ text }: { text: string }) {
  return (
    <span className="inline-block rounded-full bg-primary/10 px-2 py-0.5 text-xs text-primary">
      {text}
    </span>
  );
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

function DocSkeletonRows() {
  return (
    <>
      {[180, 140, 220].map((w, i) => (
        <tr key={i} className="border-b border-border/40">
          <td className="px-4 py-3"><div className="h-4 animate-pulse rounded bg-muted/60" style={{ width: w }} /></td>
          <td className="px-3 py-3"><div className="h-4 w-24 animate-pulse rounded bg-muted/60" /></td>
          <td className="px-3 py-3"><div className="h-4 w-20 animate-pulse rounded bg-muted/60" /></td>
          <td className="px-3 py-3"><div className="h-5 w-20 animate-pulse rounded-full bg-muted/60" /></td>
          <td className="px-3 py-3"><div className="h-4 w-16 animate-pulse rounded bg-muted/60" /></td>
          <td className="px-3 py-3"><div className="h-7 w-16 animate-pulse rounded-md bg-muted/60" /></td>
        </tr>
      ))}
    </>
  );
}

function JobSkeletonRows() {
  return (
    <>
      {[200, 160].map((w, i) => (
        <tr key={i} className="border-b border-border/40">
          <td className="px-4 py-3"><div className="h-4 animate-pulse rounded bg-muted/60" style={{ width: w }} /></td>
          <td className="px-3 py-3"><div className="h-4 w-16 animate-pulse rounded bg-muted/60" /></td>
          <td className="px-3 py-3"><div className="h-5 w-20 animate-pulse rounded-full bg-muted/60" /></td>
          <td className="px-3 py-3"><div className="h-4 w-24 animate-pulse rounded bg-muted/60" /></td>
          <td className="px-3 py-3"><div className="h-4 w-16 animate-pulse rounded bg-muted/60" /></td>
        </tr>
      ))}
    </>
  );
}

export function IngestionPanel({ baseUrl }: IngestionPanelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [pdfUrl, setPdfUrl] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Record<string, JobItem>>({});
  const [documents, setDocuments] = useState<Record<string, DocumentItem>>({});
  const [expandedDocId, setExpandedDocId] = useState<string | null>(null);
  const [expandedJobId, setExpandedJobId] = useState<string | null>(null);
  const pollersRef = useRef<Map<string, number>>(new Map());

  const sortedDocs = useMemo(
    () => Object.values(documents).sort((a, b) => b.created_at.localeCompare(a.created_at)),
    [documents]
  );
  const sortedJobs = useMemo(
    () =>
      Object.values(jobs).sort(
        (a, b) =>
          new Date(b.finished_at || b.started_at || b.created_at || 0).getTime() -
          new Date(a.finished_at || a.started_at || a.created_at || 0).getTime()
      ),
    [jobs]
  );

  const stopPolling = (jobId: string) => {
    const t = pollersRef.current.get(jobId);
    if (t) { window.clearInterval(t); pollersRef.current.delete(jobId); }
  };

  const pollStatus = async (jobId: string) => {
    try {
      const payload = await getIngestion(baseUrl, jobId);
      setJobs((prev) => ({ ...prev, [jobId]: { ...payload, updatedAtText: new Date().toLocaleTimeString() } }));
      if (isTerminal(payload.status)) {
        stopPolling(jobId);
        void refresh();
        toast.info(`Job ${jobId.slice(0, 8)}… ${payload.status}`);
      }
    } catch {
      stopPolling(jobId);
    }
  };

  const startPolling = (jobId: string) => {
    if (pollersRef.current.has(jobId)) return;
    void pollStatus(jobId);
    const id = window.setInterval(() => void pollStatus(jobId), 3000);
    pollersRef.current.set(jobId, id);
  };

  const refresh = async () => {
    if (!baseUrl.trim()) return;
    setRefreshing(true);
    try {
      const [jobsPayload, docsPayload] = await Promise.all([
        listIngestions(baseUrl),
        listDocuments(baseUrl),
      ]);
      const ingestions = normalizeList<IngestionStatus>(jobsPayload);
      const docs = normalizeList<DocumentItem>(docsPayload);

      const hydratedJobs: Record<string, JobItem> = {};
      for (const i of ingestions) {
        hydratedJobs[i.ingestion_id] = {
          ...i,
          updatedAtText: new Date(i.finished_at || i.started_at || i.created_at || Date.now()).toLocaleTimeString(),
        };
      }

      const hydratedDocs: Record<string, DocumentItem> = {};
      for (const d of docs) {
        if (d.document_id) hydratedDocs[d.document_id] = d;
      }

      setJobs(hydratedJobs);
      setDocuments(hydratedDocs);

      for (const jobId of pollersRef.current.keys()) {
        const live = hydratedJobs[jobId];
        if (!live || isTerminal(live.status)) stopPolling(jobId);
      }
      for (const i of ingestions) {
        if (!isTerminal(i.status)) startPolling(i.ingestion_id);
      }
    } catch (err) {
      toast.error(`Failed to load admin data: ${(err as Error).message}`);
    } finally {
      setRefreshing(false);
      setInitialLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
    return () => { for (const t of pollersRef.current.values()) window.clearInterval(t); };
  }, [baseUrl]);

  const submit = async () => {
    if (!file || submitting) return;
    setSubmitting(true);
    try {
      const created = await submitIngestion(baseUrl, file, {
        category: null,
        reference: null,
        guideline_title: null,
        pdf_url: pdfUrl.trim() || null,
      });
      setJobs((prev) => ({
        ...prev,
        [created.ingestion_id]: {
          ingestion_id: created.ingestion_id,
          document_id: created.document_id,
          filename: file.name,
          status: created.status,
          stage: created.stage,
          error_code: null,
          error_message: null,
          stats: {},
          quality_report: null,
          created_at: new Date().toISOString(),
          started_at: null,
          finished_at: null,
          updatedAtText: new Date().toLocaleTimeString(),
        },
      }));
      setExpandedJobId(created.ingestion_id);
      if (!isTerminal(created.status)) startPolling(created.ingestion_id);
      void refresh();
      toast.success(created.reused_existing_document ? "Document already indexed (skipped re-ingestion)" : "Ingestion started");
      setFile(null);
      setPdfUrl("");
    } catch (err) {
      toast.error(`Upload failed: ${(err as Error).message}`);
    } finally {
      setSubmitting(false);
    }
  };

  const onDelete = async (doc: DocumentItem) => {
    if (!doc.document_id) return;
    if (!window.confirm(`Delete "${doc.filename}"?\n\nThis removes the PDF, all chunks, and Qdrant vectors.`)) return;
    setDeletingId(doc.document_id);
    try {
      await deleteDocument(baseUrl, doc.document_id);
      setDocuments((prev) => { const n = { ...prev }; delete n[doc.document_id]; return n; });
      if (expandedDocId === doc.document_id) setExpandedDocId(null);
      toast.success("Document deleted");
      void refresh();
    } catch (err) {
      toast.error(`Delete failed: ${(err as Error).message}`);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="space-y-6">

      {/* ── Upload ─────────────────────────────────────────────────── */}
      <Card className="border-primary/20">
        <CardHeader className="border-b border-border/50 pb-4">
          <CardTitle className="text-base font-semibold text-primary">Upload Document</CardTitle>
          <CardDescription>
            Upload any healthcare PDF — title, type, conditions, drugs, and specialties are extracted automatically.
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-5">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
            <div className="flex-1 space-y-1.5">
              <Label htmlFor="upload-file">PDF File <span className="text-destructive">*</span></Label>
              <Input
                id="upload-file"
                type="file"
                accept=".pdf,application/pdf"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                className="cursor-pointer"
              />
              {file && <p className="truncate text-xs text-muted-foreground">{file.name}</p>}
            </div>
            <div className="flex-1 space-y-1.5">
              <Label htmlFor="upload-url">
                Source URL{" "}
                <span className="text-xs font-normal text-muted-foreground">(optional — used in citations)</span>
              </Label>
              <Input
                id="upload-url"
                type="url"
                value={pdfUrl}
                onChange={(e) => setPdfUrl(e.target.value)}
                placeholder="https://www.nice.org.uk/guidance/ng136"
                className="font-mono text-sm"
              />
            </div>
          </div>
          <div className="mt-4">
            <Button
              onClick={submit}
              disabled={!file || submitting}
              className="bg-accent hover:bg-accent/90 text-white"
            >
              <FileUp className="h-4 w-4" />
              {submitting ? "Uploading…" : "Submit for Ingestion"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* ── Documents Library ─────────────────────────────────────── */}
      <Card className="border-primary/20">
        <CardHeader className="border-b border-border/50 pb-4">
          <div className="flex items-center justify-between gap-4">
            <div>
              <CardTitle className="text-base font-semibold text-primary">
                Documents Library
                {sortedDocs.length > 0 && (
                  <span className="ml-2 rounded-full bg-primary/10 px-2 py-0.5 text-xs font-normal text-primary">
                    {sortedDocs.length}
                  </span>
                )}
              </CardTitle>
              <CardDescription>All indexed documents with auto-extracted clinical metadata.</CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={() => void refresh()} disabled={refreshing}>
              <RefreshCcw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
              {refreshing ? "Refreshing…" : "Refresh"}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {!initialLoading && !sortedDocs.length ? (
            <div className="flex flex-col items-center gap-2 py-16 text-center">
              <FileText className="h-10 w-10 text-muted-foreground/40" />
              <p className="text-sm font-medium text-muted-foreground">No documents yet.</p>
              <p className="text-xs text-muted-foreground">Upload a PDF above to start.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] text-sm">
                <thead>
                  <tr className="border-b border-border/60 bg-primary/5 text-left">
                    <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wide text-primary">File</th>
                    <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-primary">Extracted Title</th>
                    <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-primary">Type</th>
                    <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-primary">Status</th>
                    <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-primary">Uploaded</th>
                    <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-primary">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/40">
                  {initialLoading && <DocSkeletonRows />}
                  {sortedDocs.map((doc) => {
                    const isExpanded = expandedDocId === doc.document_id;
                    const status = doc.latest_status || "unknown";
                    const dm = doc.doc_metadata;
                    return (
                      <>
                        <tr
                          key={doc.document_id}
                          className={`cursor-pointer transition-colors hover:bg-secondary/40 ${isExpanded ? "bg-secondary/50" : ""}`}
                          onClick={() => setExpandedDocId((p) => (p === doc.document_id ? null : doc.document_id))}
                        >
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-2.5">
                              {isExpanded
                                ? <ChevronDown className="h-3.5 w-3.5 shrink-0 text-primary" />
                                : <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />}
                              <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
                              <span className="max-w-[200px] truncate font-medium" title={doc.filename}>
                                {doc.filename}
                              </span>
                            </div>
                          </td>
                          <td className="px-3 py-3 text-xs text-foreground/80 max-w-[200px]">
                            {dm?.title
                              ? <span className="truncate block max-w-[200px]" title={dm.title}>{dm.title}</span>
                              : <span className="text-muted-foreground italic">Extracting…</span>}
                          </td>
                          <td className="px-3 py-3">
                            {dm?.document_type
                              ? <Pill text={dm.document_type.replace(/_/g, " ")} />
                              : <span className="text-xs text-muted-foreground">—</span>}
                          </td>
                          <td className="px-3 py-3">
                            <Badge variant={statusVariant(status)} className="text-xs">
                              {status}
                            </Badge>
                          </td>
                          <td className="px-3 py-3 text-xs text-muted-foreground whitespace-nowrap">
                            {new Date(doc.created_at).toLocaleDateString()}
                          </td>
                          <td className="px-3 py-3" onClick={(e) => e.stopPropagation()}>
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-7 border-destructive/30 px-2 text-destructive hover:bg-destructive/5 hover:border-destructive/60"
                              onClick={() => void onDelete(doc)}
                              disabled={deletingId === doc.document_id}
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                              {deletingId === doc.document_id ? "Deleting…" : "Delete"}
                            </Button>
                          </td>
                        </tr>

                        {isExpanded && (
                          <tr key={`${doc.document_id}-detail`} className="bg-secondary/20">
                            <td colSpan={6} className="px-4 py-5">
                              <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">

                                {/* Document info */}
                                <div className="space-y-2">
                                  <p className="text-xs font-semibold uppercase tracking-wide text-primary">Document</p>
                                  <MetaRow label="Filename" value={doc.filename} />
                                  <MetaRow label="Pages" value={String(doc.page_count ?? "—")} />
                                  <MetaRow label="Parser" value={doc.parser_provider ?? "—"} />
                                  <MetaRow label="Status" value={status} />
                                  <MetaRow label="Stage" value={stageLabel(doc.latest_stage)} />
                                  <MetaRow label="Uploaded" value={new Date(doc.created_at).toLocaleString()} />
                                  {doc.latest_finished_at && (
                                    <MetaRow label="Finished" value={new Date(doc.latest_finished_at).toLocaleString()} />
                                  )}
                                  <MetaRow label="Document ID" value={doc.document_id} />
                                </div>

                                {/* Auto-extracted clinical metadata */}
                                <div className="space-y-2">
                                  <p className="text-xs font-semibold uppercase tracking-wide text-primary">Auto-extracted Metadata</p>
                                  {!dm ? (
                                    <p className="text-xs text-muted-foreground italic">Not yet extracted — ingestion may still be running.</p>
                                  ) : (
                                    <>
                                      <MetaRow label="Title" value={dm.title ?? "—"} />
                                      <MetaRow label="Type" value={dm.document_type?.replace(/_/g, " ") ?? "—"} />
                                      <MetaRow label="Issuing body" value={dm.issuing_body ?? "—"} />
                                      <MetaRow label="Authority" value={dm.issuing_body_authority ?? "—"} />
                                      <MetaRow label="Reference ID" value={dm.reference_id ?? "—"} />
                                      <MetaRow label="Published" value={dm.publication_date ?? "—"} />
                                      <MetaRow label="Scope" value={(dm.geographic_scope ?? []).join(", ") || "—"} />
                                      <MetaRow label="Evidence system" value={dm.evidence_grading_system ?? "—"} />
                                      {dm.summary && (
                                        <div className="pt-1">
                                          <p className="text-xs text-muted-foreground mb-0.5">Summary</p>
                                          <p className="text-xs text-foreground leading-relaxed">{dm.summary}</p>
                                        </div>
                                      )}
                                    </>
                                  )}
                                </div>

                                {/* Clinical scope */}
                                <div className="space-y-2">
                                  <p className="text-xs font-semibold uppercase tracking-wide text-primary">Clinical Scope</p>
                                  {!dm ? (
                                    <p className="text-xs text-muted-foreground italic">—</p>
                                  ) : (
                                    <>
                                      {(dm.specialties ?? []).length > 0 && (
                                        <div>
                                          <p className="text-xs text-muted-foreground mb-1">Specialties</p>
                                          <div className="flex flex-wrap gap-1">
                                            {dm.specialties!.map((s) => <Pill key={s} text={s} />)}
                                          </div>
                                        </div>
                                      )}
                                      {(dm.conditions ?? []).length > 0 && (
                                        <div>
                                          <p className="text-xs text-muted-foreground mb-1">Conditions ({dm.conditions!.length})</p>
                                          <div className="flex flex-wrap gap-1">
                                            {dm.conditions!.slice(0, 6).map((c) => (
                                              <Pill key={c.label} text={c.code ? `${c.label} [${c.code}]` : c.label} />
                                            ))}
                                            {dm.conditions!.length > 6 && (
                                              <span className="text-xs text-muted-foreground">+{dm.conditions!.length - 6} more</span>
                                            )}
                                          </div>
                                        </div>
                                      )}
                                      {(dm.drugs ?? []).length > 0 && (
                                        <div>
                                          <p className="text-xs text-muted-foreground mb-1">Drugs ({dm.drugs!.length})</p>
                                          <div className="flex flex-wrap gap-1">
                                            {dm.drugs!.slice(0, 6).map((d) => (
                                              <Pill key={d.label} text={d.label} />
                                            ))}
                                            {dm.drugs!.length > 6 && (
                                              <span className="text-xs text-muted-foreground">+{dm.drugs!.length - 6} more</span>
                                            )}
                                          </div>
                                        </div>
                                      )}
                                      {(dm.care_setting ?? []).length > 0 && (
                                        <MetaRow label="Care setting" value={(dm.care_setting ?? []).join(", ")} />
                                      )}
                                      <div className="flex gap-3 pt-1 flex-wrap">
                                        {dm.has_dosing_tables && <Pill text="Dosing tables" />}
                                        {dm.has_red_flags && <Pill text="Red flags" />}
                                        {dm.has_referral_criteria && <Pill text="Referral criteria" />}
                                      </div>
                                    </>
                                  )}
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

      {/* ── Ingestion Jobs ────────────────────────────────────────── */}
      <Card className="border-primary/20">
        <CardHeader className="border-b border-border/50 pb-4">
          <div className="flex items-center justify-between gap-4">
            <div>
              <CardTitle className="text-base font-semibold text-primary">
                Ingestion Jobs
                {sortedJobs.length > 0 && (
                  <span className="ml-2 rounded-full bg-primary/10 px-2 py-0.5 text-xs font-normal text-primary">
                    {sortedJobs.length}
                  </span>
                )}
              </CardTitle>
              <CardDescription>Live pipeline status. Click a row to expand details.</CardDescription>
            </div>
            {sortedJobs.some((j) => !isTerminal(j.status)) && (
              <Badge variant="warning" className="animate-pulse gap-1 shrink-0">
                <Clock3 className="h-3 w-3" /> Processing
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {!initialLoading && !sortedJobs.length ? (
            <div className="flex flex-col items-center gap-2 py-16 text-center">
              <Clock3 className="h-10 w-10 text-muted-foreground/40" />
              <p className="text-sm font-medium text-muted-foreground">No jobs yet.</p>
              <p className="text-xs text-muted-foreground">Upload a PDF to start an ingestion job.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[560px] text-sm">
                <thead>
                  <tr className="border-b border-border/60 bg-primary/5 text-left">
                    <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wide text-primary">File</th>
                    <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-primary">Job ID</th>
                    <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-primary">Status</th>
                    <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-primary">Stage</th>
                    <th className="px-3 py-3 text-xs font-semibold uppercase tracking-wide text-primary">Updated</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/40">
                  {initialLoading && <JobSkeletonRows />}
                  {sortedJobs.map((job) => {
                    const isExpanded = expandedJobId === job.ingestion_id;
                    const isActive = !isTerminal(job.status);
                    return (
                      <>
                        <tr
                          key={job.ingestion_id}
                          className={`cursor-pointer transition-colors hover:bg-secondary/40 ${isExpanded ? "bg-secondary/50" : ""}`}
                          onClick={() => setExpandedJobId((p) => (p === job.ingestion_id ? null : job.ingestion_id))}
                        >
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-2.5">
                              {isExpanded
                                ? <ChevronDown className="h-3.5 w-3.5 shrink-0 text-primary" />
                                : <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />}
                              {job.status === "completed"
                                ? <CheckCircle2 className="h-4 w-4 shrink-0 text-green-600" />
                                : job.status === "failed"
                                ? <XCircle className="h-4 w-4 shrink-0 text-destructive" />
                                : <Clock3 className={`h-4 w-4 shrink-0 text-accent ${isActive ? "animate-pulse" : ""}`} />}
                              <span className="max-w-[220px] truncate font-medium" title={job.filename}>
                                {job.filename}
                              </span>
                            </div>
                          </td>
                          <td className="px-3 py-3 font-mono text-xs text-muted-foreground">
                            {job.ingestion_id.slice(0, 8)}…
                          </td>
                          <td className="px-3 py-3">
                            <Badge variant={statusVariant(job.status)} className="text-xs">{job.status}</Badge>
                          </td>
                          <td className="px-3 py-3 text-xs text-foreground/80">{stageLabel(job.stage)}</td>
                          <td className="px-3 py-3 text-xs text-muted-foreground whitespace-nowrap">
                            {job.updatedAtText}
                          </td>
                        </tr>

                        {isExpanded && (
                          <tr key={`${job.ingestion_id}-detail`} className="bg-secondary/20">
                            <td colSpan={5} className="px-4 py-4">
                              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                                <div className="space-y-2">
                                  <p className="text-xs font-semibold uppercase tracking-wide text-primary">Job Info</p>
                                  <MetaRow label="Job ID" value={job.ingestion_id} />
                                  <MetaRow label="Document ID" value={String(job.document_id ?? "—")} />
                                  <MetaRow label="Status" value={job.status} />
                                  <MetaRow label="Stage" value={stageLabel(job.stage)} />
                                  {job.error_message && (
                                    <div>
                                      <p className="text-xs text-muted-foreground mb-0.5">Error</p>
                                      <p className="text-xs text-destructive">[{job.error_code}] {job.error_message}</p>
                                    </div>
                                  )}
                                  <MetaRow label="Created" value={job.created_at ? new Date(job.created_at).toLocaleString() : "—"} />
                                  <MetaRow label="Started" value={job.started_at ? new Date(job.started_at).toLocaleString() : "—"} />
                                  <MetaRow label="Finished" value={job.finished_at ? new Date(job.finished_at).toLocaleString() : "—"} />
                                </div>

                                {Object.keys(job.stats ?? {}).length > 0 && (
                                  <div className="space-y-2">
                                    <p className="text-xs font-semibold uppercase tracking-wide text-primary">Stats</p>
                                    {Object.entries(job.stats).map(([k, v]) => (
                                      <MetaRow key={k} label={k.replace(/_/g, " ")} value={String(v ?? "—")} />
                                    ))}
                                  </div>
                                )}

                                {job.quality_report && Object.keys(job.quality_report).length > 0 && (
                                  <div className="space-y-2">
                                    <p className="text-xs font-semibold uppercase tracking-wide text-primary">Quality Report</p>
                                    {Object.entries(job.quality_report).map(([k, v]) => (
                                      <MetaRow key={k} label={k.replace(/_/g, " ")} value={String(v ?? "—")} />
                                    ))}
                                  </div>
                                )}
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
    </div>
  );
}
