"use client";

import { useMemo, useState } from "react";
import { Database, Search, SlidersHorizontal } from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { searchRetrieval } from "@/lib/api";
import type { RetrievedChunk, RetrievalFilters, RetrievalProfile, RetrievalSearchResponse } from "@/lib/types";

type RetrievalPanelProps = {
  baseUrl: string;
};

function parseCsv(value: string) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function buildFilters(input: {
  docType: string;
  sectionType: string;
  specialties: string;
  conditions: string;
  geographicScope: string;
  careSetting: string;
  hasDosingTables: string;
  hasRedFlags: string;
}): RetrievalFilters | undefined {
  const filters: RetrievalFilters = {};

  const docType = parseCsv(input.docType);
  const sectionType = parseCsv(input.sectionType);
  const specialties = parseCsv(input.specialties);
  const conditions = parseCsv(input.conditions);
  const geographicScope = parseCsv(input.geographicScope);
  const careSetting = parseCsv(input.careSetting);

  if (docType.length) filters.doc_type = docType;
  if (sectionType.length) filters.section_type = sectionType;
  if (specialties.length) filters.specialties = specialties;
  if (conditions.length) filters.conditions_codes = conditions;
  if (geographicScope.length) filters.geographic_scope = geographicScope;
  if (careSetting.length) filters.care_setting = careSetting;
  if (input.hasDosingTables === "yes") filters.has_dosing_tables = true;
  if (input.hasDosingTables === "no") filters.has_dosing_tables = false;
  if (input.hasRedFlags === "yes") filters.has_red_flags = true;
  if (input.hasRedFlags === "no") filters.has_red_flags = false;

  return Object.keys(filters).length ? filters : undefined;
}

function scoreBadge(value: number | null | undefined, label: string) {
  if (value === null || value === undefined) return null;
  return (
    <Badge variant="secondary" className="font-mono text-[11px]">
      {label} {value.toFixed(3)}
    </Badge>
  );
}

function textValue(value: unknown) {
  return typeof value === "string" ? value : "";
}

function SourceCard({
  chunk,
  index,
}: {
  chunk: RetrievedChunk;
  index: number;
}) {
  const [expandedChild, setExpandedChild] = useState(false);
  const [expandedParent, setExpandedParent] = useState(false);

  const metadata = (chunk.doc_metadata ?? {}) as Record<string, unknown>;
  const guidelineTitle = textValue(metadata.guideline_title) || chunk.filename;
  const reference = textValue(metadata.reference);
  const pdfUrl = textValue(metadata.pdf_url);

  let sectionDisplay = chunk.section_path || "";
  if (sectionDisplay) {
    const lower = sectionDisplay.toLowerCase();
    const prefixWithRef = reference ? `${guidelineTitle} (${reference})` : guidelineTitle;
    if (lower.startsWith(prefixWithRef.toLowerCase())) {
      sectionDisplay = sectionDisplay.slice(prefixWithRef.length).replace(/^\s*>\s*/, "").trim();
    } else if (lower.startsWith(guidelineTitle.toLowerCase())) {
      sectionDisplay = sectionDisplay
        .slice(guidelineTitle.length)
        .replace(/^\s*\(.*?\)\s*>\s*/, "")
        .replace(/^\s*>\s*/, "")
        .trim();
    }
  }

  const childPreview = chunk.snippet?.trim() || chunk.full_snippet?.slice(0, 300) || "";
  const childFull = chunk.full_snippet?.trim() || childPreview;
  const parentPreview = chunk.parent_text?.trim().slice(0, 420) || "";
  const parentFull = chunk.parent_text?.trim() || "";
  const hasLongChild = childFull.length > childPreview.length;
  const hasParent = Boolean(parentFull);
  const hasLongParent = parentFull.length > parentPreview.length;

  return (
    <div className="overflow-hidden rounded-xl border border-primary/15 bg-white shadow-sm">
      <div className="border-b border-primary/10 bg-primary/5 px-4 py-2">
        <div className="flex flex-wrap items-center gap-2">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-primary/60">
            Source {index + 1}
          </p>
          {chunk.section_type ? <Badge variant="secondary">{chunk.section_type}</Badge> : null}
          {chunk.recommendation_strength ? (
            <Badge variant="secondary">{chunk.recommendation_strength}</Badge>
          ) : null}
          {scoreBadge(chunk.rerank_score, "rerank")}
          {scoreBadge(chunk.fused_score, "rrf")}
          {scoreBadge(chunk.dense_score, "dense")}
        </div>
      </div>

      <div className="space-y-3 px-4 py-3">
        <blockquote className="whitespace-pre-line border-l-[3px] border-accent pl-3 text-sm italic leading-relaxed text-foreground/85">
          &ldquo;{expandedChild ? childFull : childPreview}&rdquo;
        </blockquote>

        {hasLongChild ? (
          <button
            type="button"
            onClick={() => setExpandedChild((prev) => !prev)}
            className="text-xs font-medium text-primary/80 hover:text-primary hover:underline"
          >
            {expandedChild ? "Show less child context" : "Show full child context"}
          </button>
        ) : null}

        <div>
          <p className="text-xs font-bold text-foreground">
            {guidelineTitle}
            {reference ? ` (${reference})` : ""}
          </p>
          {sectionDisplay ? <p className="text-xs text-muted-foreground">&rsaquo; {sectionDisplay}</p> : null}
        </div>

        {hasParent ? (
          <div className="rounded-lg border border-border/70 bg-secondary/20 px-3 py-3">
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-primary/70">
              Parent Context
            </p>
            <p className="whitespace-pre-line text-xs leading-relaxed text-foreground/85">
              {expandedParent || !hasLongParent ? parentFull : `${parentPreview}...`}
            </p>
            {hasLongParent ? (
              <button
                type="button"
                onClick={() => setExpandedParent((prev) => !prev)}
                className="mt-2 text-xs font-medium text-primary/80 hover:text-primary hover:underline"
              >
                {expandedParent ? "Collapse parent context" : "Expand parent context"}
              </button>
            ) : null}
          </div>
        ) : null}

        <div className="flex flex-wrap gap-2">
          {textValue(metadata.doc_type) ? <Badge variant="outline">{textValue(metadata.doc_type)}</Badge> : null}
          {Array.isArray(metadata.geographic_scope) && metadata.geographic_scope.length ? (
            <Badge variant="outline">{(metadata.geographic_scope as string[]).join(", ")}</Badge>
          ) : null}
          {Array.isArray(metadata.care_setting) && metadata.care_setting.length ? (
            <Badge variant="outline">{(metadata.care_setting as string[]).join(", ")}</Badge>
          ) : null}
        </div>

        {pdfUrl ? (
          <div className="flex justify-end pt-0.5">
            <a
              href={pdfUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-md border border-accent/30 bg-accent/5 px-2.5 py-1 text-xs font-medium text-accent transition-colors hover:bg-accent/10"
            >
              View guideline
            </a>
          </div>
        ) : null}
      </div>
    </div>
  );
}

export function RetrievalPanel({ baseUrl }: RetrievalPanelProps) {
  const [query, setQuery] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [profile, setProfile] = useState<RetrievalProfile>("specific");
  const [topK, setTopK] = useState(10);
  const [docType, setDocType] = useState("");
  const [sectionType, setSectionType] = useState("");
  const [specialties, setSpecialties] = useState("");
  const [conditions, setConditions] = useState("");
  const [geographicScope, setGeographicScope] = useState("");
  const [careSetting, setCareSetting] = useState("");
  const [hasDosingTables, setHasDosingTables] = useState("any");
  const [hasRedFlags, setHasRedFlags] = useState("any");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RetrievalSearchResponse | null>(null);

  const activeFilterCount = useMemo(() => {
    const filters = buildFilters({
      docType,
      sectionType,
      specialties,
      conditions,
      geographicScope,
      careSetting,
      hasDosingTables,
      hasRedFlags,
    });
    return Object.keys(filters ?? {}).length;
  }, [
    careSetting,
    conditions,
    docType,
    geographicScope,
    hasDosingTables,
    hasRedFlags,
    sectionType,
    specialties,
  ]);

  const advancedFilters = showAdvanced
    ? buildFilters({
        docType,
        sectionType,
        specialties,
        conditions,
        geographicScope,
        careSetting,
        hasDosingTables,
        hasRedFlags,
      })
    : undefined;

  const runSearch = async () => {
    if (!query.trim() || loading) return;
    setLoading(true);
    try {
      const response = await searchRetrieval(baseUrl, {
        query: query.trim(),
        profile,
        top_k: topK,
        filters: advancedFilters,
        expand_parents: true,
      });
      setResult(response);
    } catch (error) {
      toast.error(`Retrieval failed: ${(error as Error).message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card className="border-primary/20">
        <CardHeader className="border-b border-border/50 pb-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <CardTitle className="text-base font-semibold text-primary">Retrieval Playground</CardTitle>
              <CardDescription>
                Default flow is just question in, sources out. Advanced filters are optional.
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="gap-1">
                <Database className="h-3.5 w-3.5" />
                {showAdvanced ? `Advanced on${activeFilterCount ? ` (${activeFilterCount} filters)` : ""}` : "Auto mode"}
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-5 pt-5">
          <div className="space-y-1.5">
            <Label htmlFor="retrieval-query">Search query</Label>
            <Textarea
              id="retrieval-query"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Ask a question and review the retrieved sources..."
              className="min-h-[96px]"
            />
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={() => void runSearch()} disabled={loading || !query.trim()} className="gap-2">
              <Search className="h-4 w-4" />
              {loading ? "Searching..." : "Run retrieval"}
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={() => setShowAdvanced((prev) => !prev)}
              className="gap-2"
            >
              <SlidersHorizontal className="h-4 w-4" />
              {showAdvanced ? "Hide advanced options" : "Show advanced options"}
            </Button>
            {result ? (
              <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                <Badge variant="secondary">{result.chunks.length} chunks</Badge>
                <Badge variant="secondary">{result.query_ms} ms</Badge>
                <Badge variant="secondary">corpus v{result.corpus_version}</Badge>
                <Badge variant={result.cache_hit ? "success" : "secondary"}>
                  cache {result.cache_hit ? "hit" : "miss"}
                </Badge>
              </div>
            ) : null}
          </div>

          {showAdvanced ? (
            <div className="grid gap-4 rounded-xl border border-border/70 bg-secondary/15 p-4 md:grid-cols-2 xl:grid-cols-4">
              <div className="space-y-1.5">
                <Label htmlFor="retrieval-profile">Profile</Label>
                <select
                  id="retrieval-profile"
                  value={profile}
                  onChange={(event) => setProfile(event.target.value as RetrievalProfile)}
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                >
                  <option value="specific">specific</option>
                  <option value="narrow">narrow</option>
                  <option value="broad">broad</option>
                </select>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="retrieval-topk">Top K</Label>
                <Input
                  id="retrieval-topk"
                  type="number"
                  min={1}
                  max={20}
                  value={topK}
                  onChange={(event) => setTopK(Number(event.target.value) || 1)}
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="retrieval-doc-type">Doc type</Label>
                <Input
                  id="retrieval-doc-type"
                  value={docType}
                  onChange={(event) => setDocType(event.target.value)}
                  placeholder="clinical_guideline"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="retrieval-section-type">Section type</Label>
                <Input
                  id="retrieval-section-type"
                  value={sectionType}
                  onChange={(event) => setSectionType(event.target.value)}
                  placeholder="recommendation,dosing-table"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="retrieval-specialties">Specialties</Label>
                <Input
                  id="retrieval-specialties"
                  value={specialties}
                  onChange={(event) => setSpecialties(event.target.value)}
                  placeholder="nephrology,geriatrics"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="retrieval-conditions">Condition codes</Label>
                <Input
                  id="retrieval-conditions"
                  value={conditions}
                  onChange={(event) => setConditions(event.target.value)}
                  placeholder="N18.3,38341003"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="retrieval-geo">Geographic scope</Label>
                <Input
                  id="retrieval-geo"
                  value={geographicScope}
                  onChange={(event) => setGeographicScope(event.target.value)}
                  placeholder="GB"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="retrieval-care-setting">Care setting</Label>
                <Input
                  id="retrieval-care-setting"
                  value={careSetting}
                  onChange={(event) => setCareSetting(event.target.value)}
                  placeholder="primary,outpatient"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="retrieval-dosing">Has dosing tables</Label>
                <select
                  id="retrieval-dosing"
                  value={hasDosingTables}
                  onChange={(event) => setHasDosingTables(event.target.value)}
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                >
                  <option value="any">any</option>
                  <option value="yes">yes</option>
                  <option value="no">no</option>
                </select>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="retrieval-red-flags">Has red flags</Label>
                <select
                  id="retrieval-red-flags"
                  value={hasRedFlags}
                  onChange={(event) => setHasRedFlags(event.target.value)}
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                >
                  <option value="any">any</option>
                  <option value="yes">yes</option>
                  <option value="no">no</option>
                </select>
              </div>
            </div>
          ) : null}
        </CardContent>
      </Card>

      {result ? (
        <Card className="border-primary/20">
          <CardHeader className="border-b border-border/50 pb-4">
            <CardTitle className="text-base font-semibold text-primary">Retrieved Sources</CardTitle>
            <CardDescription>
              Same source-first presentation we can carry forward into the chat experience.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 pt-5">
            {result.chunks.length ? (
              result.chunks.map((chunk, index) => (
                <SourceCard key={chunk.chunk_id} chunk={chunk} index={index} />
              ))
            ) : (
              <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-border/70 py-14 text-center">
                <Database className="h-10 w-10 text-muted-foreground/40" />
                <p className="text-sm font-medium text-muted-foreground">No chunks matched this query.</p>
                <p className="text-xs text-muted-foreground">Try relaxing filters or switching to the broad profile.</p>
              </div>
            )}
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
