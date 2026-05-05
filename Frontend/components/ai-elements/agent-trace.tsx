"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect } from "react";
import {
  Search,
  Filter,
  BookOpen,
  BarChart3,
  RefreshCw,
  PenTool,
  ShieldCheck,
  CheckCircle2,
  Loader2,
  AlertCircle,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

export type TraceStep = {
  stage: string;
  label: string;
  description: string;
  status: "done" | "error" | "running";
  ms: number;
  [key: string]: unknown;
};

const STAGE_ICONS: Record<string, React.ElementType> = {
  classify_intent: Search,
  extract_filters: Filter,
  retrieve: BookOpen,
  grade_chunks: BarChart3,
  rewrite_query: RefreshCw,
  generate_answer: PenTool,
  verify_grounding: ShieldCheck,
};

const STAGE_COLORS: Record<string, string> = {
  classify_intent: "text-blue-500",
  extract_filters: "text-violet-500",
  retrieve: "text-emerald-500",
  grade_chunks: "text-amber-500",
  rewrite_query: "text-orange-500",
  generate_answer: "text-sky-500",
  verify_grounding: "text-green-600",
};

type AgentTraceProps = {
  steps: TraceStep[];
  isRunning: boolean;
  currentStage?: string;
};

export function AgentTrace({ steps, isRunning, currentStage }: AgentTraceProps) {
  const [expanded, setExpanded] = useState(true);

  // Auto-collapse when agent finishes
  useEffect(() => {
    if (!isRunning) {
      setExpanded(false);
    } else {
      setExpanded(true);
    }
  }, [isRunning]);

  if (steps.length === 0 && !isRunning) return null;

  return (
    <div className="rounded-xl border border-primary/20 bg-gradient-to-br from-primary/[0.03] to-transparent overflow-hidden shadow-sm">
      {/* Header */}
      <button
        type="button"
        onClick={() => setExpanded((p) => !p)}
        className="flex w-full items-center gap-2.5 px-4 py-2.5 text-left transition-colors hover:bg-primary/5"
      >
        {isRunning ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
        ) : (
          <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
        )}
        <span className="flex-1 text-[11px] font-semibold uppercase tracking-wider text-primary/70">
          {isRunning ? "Agent is thinking..." : `Completed in ${steps.reduce((a, s) => a + (s.ms || 0), 0)}ms`}
        </span>
        <span className="text-[10px] text-muted-foreground">
          {steps.length} step{steps.length !== 1 ? "s" : ""}
        </span>
        {expanded ? (
          <ChevronUp className="h-3.5 w-3.5 text-muted-foreground" />
        ) : (
          <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
        )}
      </button>

      {/* Steps */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="border-t border-primary/10 px-4 py-2.5 space-y-1">
              {steps.map((step, idx) => {
                const Icon = STAGE_ICONS[step.stage] || Search;
                const colorClass = STAGE_COLORS[step.stage] || "text-primary";

                return (
                  <motion.div
                    key={`${step.stage}-${idx}`}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.2, delay: 0.05 }}
                    className="flex items-start gap-2.5 py-1"
                  >
                    {/* Icon */}
                    <div className="mt-0.5 flex-shrink-0">
                      {step.status === "error" ? (
                        <AlertCircle className="h-3.5 w-3.5 text-red-500" />
                      ) : (
                        <Icon className={`h-3.5 w-3.5 ${colorClass}`} />
                      )}
                    </div>

                    {/* Content */}
                    <div className="min-w-0 flex-1">
                      <div className="flex items-baseline gap-2">
                        <span className="text-xs font-medium text-foreground/90 truncate">
                          {step.label}
                        </span>
                        {step.ms > 0 && (
                          <span className="flex-shrink-0 text-[10px] text-muted-foreground">
                            {step.ms}ms
                          </span>
                        )}
                      </div>

                      {/* Detail line */}
                      <StepDetail step={step} />
                    </div>

                    {/* Status dot */}
                    <div className="mt-1.5 flex-shrink-0">
                      {step.status === "done" && (
                        <div className="h-1.5 w-1.5 rounded-full bg-green-400" />
                      )}
                      {step.status === "error" && (
                        <div className="h-1.5 w-1.5 rounded-full bg-red-400" />
                      )}
                      {step.status === "running" && (
                        <Loader2 className="h-3 w-3 animate-spin text-primary/60" />
                      )}
                    </div>
                  </motion.div>
                );
              })}

              {/* Active step indicator */}
              {isRunning && currentStage && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex items-center gap-2.5 py-1"
                >
                  <Loader2 className="h-3.5 w-3.5 animate-spin text-primary/50" />
                  <span className="text-xs text-primary/60 italic">
                    {currentStage}
                  </span>
                </motion.div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/** Renders a compact detail line under each step based on step data */
function StepDetail({ step }: { step: TraceStep }) {
  const details: string[] = [];

  if (step.stage === "classify_intent" && step.intent) {
    details.push(`Intent: ${step.intent}`);
    if (step.confidence) details.push(`(${Math.round(Number(step.confidence) * 100)}%)`);
  }
  if (step.stage === "extract_filters") {
    if (step.filters && step.filters !== "none") details.push(String(step.filters));
    if (step.retrieval_query) details.push(`"${String(step.retrieval_query).slice(0, 60)}…"`);
  }
  if (step.stage === "retrieve") {
    if (step.chunk_count !== undefined) details.push(`${step.chunk_count} chunks found`);
    if (step.retrieval_ms) details.push(`(${step.retrieval_ms}ms search)`);
  }
  if (step.stage === "grade_chunks") {
    if (step.coverage_score !== undefined) details.push(`Coverage: ${Math.round(Number(step.coverage_score) * 100)}%`);
    if (step.reason) details.push(String(step.reason).slice(0, 80));
  }
  if (step.stage === "rewrite_query") {
    if (step.rewritten_query) details.push(`"${String(step.rewritten_query).slice(0, 70)}…"`);
    if (step.retry_number) details.push(`(retry ${step.retry_number})`);
  }
  if (step.stage === "generate_answer") {
    if (step.confidence !== undefined) details.push(`Confidence: ${Math.round(Number(step.confidence) * 100)}%`);
    if (step.sources_used) details.push(`${step.sources_used} sources cited`);
  }
  if (step.stage === "verify_grounding") {
    if (step.verdict) details.push(`Verdict: ${step.verdict}`);
    if (step.ungrounded_count && Number(step.ungrounded_count) > 0) {
      details.push(`${step.ungrounded_count} ungrounded`);
    }
    if (step.summary) details.push(String(step.summary).slice(0, 60));
  }
  if (step.status === "error" && step.error) {
    details.push(`Error: ${String(step.error).slice(0, 80)}`);
  }

  if (details.length === 0) return null;

  return (
    <p className="text-[10px] leading-tight text-muted-foreground/80 truncate">
      {details.join(" · ")}
    </p>
  );
}
