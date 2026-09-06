"use client";

import { useEffect, useRef, useState } from "react";
import { ThumbsDown, ThumbsUp } from "lucide-react";
import { toast } from "sonner";

import { PromptInput } from "@/components/ai-elements/prompt-input";
import { Response } from "@/components/ai-elements/response";
import { Thinking } from "@/components/ai-elements/thinking";
import { Message } from "@/components/ai-elements/message";
import { AgentTrace, type TraceStep } from "@/components/ai-elements/agent-trace";
import { cn } from "@/lib/utils";
import { chatQuery, chatQueryStream, submitFeedback } from "@/lib/api";
import type { ChatMessage, ChatResponse, Citation } from "@/lib/types";

type ChatPanelProps = {
  baseUrl: string;
};

type Rating = "up" | "down" | null;

export function ChatPanel({ baseUrl }: ChatPanelProps) {
  const [sessionId, setSessionId] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [ratings, setRatings] = useState<Record<string, Rating>>({});
  const [expandedCitations, setExpandedCitations] = useState<Set<string>>(new Set());

  // Agent trace state
  const [traceSteps, setTraceSteps] = useState<TraceStep[]>([]);
  const [traceRunning, setTraceRunning] = useState(false);
  const [currentTraceName, setCurrentTraceName] = useState<string>("");

  const toggleCitationExpanded = (chunkId: string) => {
    setExpandedCitations((prev) => {
      const next = new Set(prev);
      if (next.has(chunkId)) next.delete(chunkId);
      else next.add(chunkId);
      return next;
    });
  };

  const stripMarkdown = (text: string) => (text || "").replace(/\*\*|__|\\*|_|~~|`/g, "").trim();
  const liveIdRef = useRef<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const liveMessageRef = useRef<HTMLDivElement | null>(null);
  const didScrollToAnswerRef = useRef(false);

  useEffect(() => {
    if (!messages.length) return;
    const last = messages[messages.length - 1];
    if (last.role === "user") {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
      didScrollToAnswerRef.current = false;
      return;
    }
    if (last.role === "assistant" && last.id === liveIdRef.current) {
      if (!last.content) {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
        return;
      }
      if (!didScrollToAnswerRef.current) {
        didScrollToAnswerRef.current = true;
        liveMessageRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }
  }, [messages]);

  const formatStreamingMarkdown = (input: string) => String(input || "").replace(/\r\n/g, "\n");

  const upsertLiveAssistant = (
    content: string,
    extra?: {
      messageId?: string;
      citations?: Citation[];
      abstained?: boolean;
      abstainReason?: string | null;
      followUpQuestions?: string[];
    }
  ) => {
    const liveId = liveIdRef.current;
    if (!liveId) return;
    setMessages((prev) =>
      prev.map((item) =>
        item.id === liveId
          ? {
            ...item,
            content,
            messageId: extra?.messageId ?? item.messageId,
            citations: extra?.citations ?? item.citations,
            abstained: extra?.abstained ?? item.abstained,
            abstainReason: extra?.abstainReason ?? item.abstainReason,
            followUpQuestions: extra?.followUpQuestions ?? item.followUpQuestions,
          }
          : item
      )
    );
  };

  const startLiveAssistant = () => {
    const liveId = `live-${Date.now()}`;
    liveIdRef.current = liveId;
    didScrollToAnswerRef.current = false;
    setMessages((prev) => [...prev, { id: liveId, role: "assistant", content: "" }]);
  };

  // ── Next stage names for the "running" indicator ──
  const NEXT_STAGE: Record<string, string> = {
    classify_intent: "Extracting clinical filters...",
    extract_filters: "Searching guidelines...",
    retrieve: "Evaluating relevance...",
    grade_chunks: "Generating answer...",
    rewrite_query: "Re-searching guidelines...",
    generate_answer: "Verifying citations...",
    verify_grounding: "Finalizing...",
  };

  const sendMessage = async (overrideQuestion?: string) => {
    const text = (overrideQuestion ?? question).trim();
    if (!text || loading) return;
    setQuestion("");
    setLoading(true);

    // Reset trace state
    setTraceSteps([]);
    setTraceRunning(true);
    setCurrentTraceName("Analyzing question...");

    setMessages((prev) => [...prev, { id: `u-${Date.now()}`, role: "user", content: text }]);
    startLiveAssistant();

    const payload = { session_id: sessionId.trim() || null, question: text };
    let streamedContent = "";
    let hasStreamedDelta = false;
    let scheduledRender: number | null = null;

    const flushStreamRender = () => {
      if (scheduledRender !== null && typeof window !== "undefined") {
        window.cancelAnimationFrame(scheduledRender);
        scheduledRender = null;
      }
      upsertLiveAssistant(formatStreamingMarkdown(streamedContent));
    };

    const scheduleStreamRender = () => {
      if (scheduledRender !== null || typeof window === "undefined") return;
      scheduledRender = window.requestAnimationFrame(() => {
        scheduledRender = null;
        upsertLiveAssistant(formatStreamingMarkdown(streamedContent));
      });
    };

    try {
      const response: ChatResponse = await chatQueryStream(baseUrl, payload, {
        onStepStarted: (stage, data) => {
          console.log("onStepStarted", stage);
          const step: TraceStep = {
            stage,
            label: String(data.label || stage),
            description: String(data.description || ""),
            status: "running",
            ms: 0,
          };
          setTraceSteps((prev) => [...prev, step]);
        },
        onStage: (stage, data) => {
          console.log("onStage", stage, data);
          const completedStep: TraceStep = {
            stage,
            label: String(data.label || stage),
            description: String(data.description || ""),
            status: (data.status as "done" | "error") || "done",
            ms: Number(data.ms || 0),
            ...data,
          };
          setTraceSteps((prev) => {
            let idx = -1;
            for (let i = prev.length - 1; i >= 0; i--) {
              if (prev[i].stage === stage && prev[i].status === "running") {
                idx = i;
                break;
              }
            }
            if (idx >= 0) {
              const next = [...prev];
              next[idx] = completedStep;
              return next;
            }
            return [...prev, completedStep];
          });
          setCurrentTraceName(NEXT_STAGE[stage] || "Processing...");

          if (stage === "verify_grounding") {
            setTraceRunning(false);
            setCurrentTraceName("");
          }
        },
        onAnswerDelta: (delta) => {
          hasStreamedDelta = true;
          streamedContent += delta;
          scheduleStreamRender();
        },
        onCitationsReady: (citations) => {
          if (hasStreamedDelta) flushStreamRender();
          const sourceIndexToPosition = new Map<number, number>(
            citations.map((c, i) => [c.source_index, i + 1])
          );
          streamedContent = streamedContent.replace(/\[SOURCE\s+(\d+)\]/gi, (match, n) => {
            const original = parseInt(n, 10);
            const position = sourceIndexToPosition.get(original);
            return position !== undefined ? `[SOURCE ${position}]` : match;
          });
          upsertLiveAssistant(formatStreamingMarkdown(streamedContent), { citations });
        },
        onFollowUpQuestionsReady: (questions) => {
          upsertLiveAssistant(formatStreamingMarkdown(streamedContent), { followUpQuestions: questions });
        },
      });
      if (hasStreamedDelta) flushStreamRender();
      setSessionId(response.session_id);
      setTraceRunning(false);
      if (!hasStreamedDelta) {
        upsertLiveAssistant(formatStreamingMarkdown(response.answer));
      }
      upsertLiveAssistant(formatStreamingMarkdown(streamedContent || response.answer), {
        messageId: response.message_id,
        citations: response.citations,
        abstained: response.abstained,
        abstainReason: response.abstain_reason,
        followUpQuestions: response.follow_up_questions ?? [],
      });
    } catch (streamError) {
      setTraceRunning(false);
      if (hasStreamedDelta) {
        flushStreamRender();
        upsertLiveAssistant(`${formatStreamingMarkdown(streamedContent)}\n\n_Stream interrupted before final event._`);
        toast.error(`Streaming interrupted: ${(streamError as Error).message}`);
        return;
      }
      try {
        const response: ChatResponse = await chatQuery(baseUrl, payload);
        setSessionId(response.session_id);
        upsertLiveAssistant(formatStreamingMarkdown(response.answer), {
          messageId: response.message_id,
          citations: response.citations,
          abstained: response.abstained,
          abstainReason: response.abstain_reason,
          followUpQuestions: response.follow_up_questions ?? [],
        });
      } catch (httpError) {
        upsertLiveAssistant(`Request failed: ${(httpError as Error).message}`);
        toast.error(`Chat failed: ${(httpError as Error).message}`);
      }
    } finally {
      if (scheduledRender !== null && typeof window !== "undefined") {
        window.cancelAnimationFrame(scheduledRender);
      }
      liveIdRef.current = null;
      setLoading(false);
      setTraceRunning(false);
    }
  };

  const setRating = (localId: string, backendMessageId: string | undefined, rating: "up" | "down") => {
    const newRating = ratings[localId] === rating ? null : rating;
    setRatings((prev) => ({ ...prev, [localId]: newRating }));
    if (backendMessageId) {
      submitFeedback(baseUrl, backendMessageId, newRating).catch((err) =>
        console.warn("Feedback not saved:", err)
      );
    }
  };

  const disclaimers = (
    <div className="flex flex-col items-center gap-1 pt-2 text-center">
      <p className="text-[11px] font-medium text-amber-600">⚠ No patient identifiable details please</p>
      <p className="text-[11px] text-muted-foreground">ClinTel can make mistakes — always verify from NICE guidelines</p>
    </div>
  );

  if (!messages.length) {
    return (
      <div className="flex min-h-0 flex-1 flex-col items-center justify-center px-4 sm:px-6">
        <div className="w-full max-w-2xl space-y-6">
          <div className="space-y-3 text-center">
            <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-primary/70 shadow-sm">
              <span className="text-2xl font-bold text-white">C</span>
            </div>
            <h2 className="text-2xl font-semibold text-primary">NICE guidelines at your fingertips</h2>
          </div>
          <PromptInput
            value={question}
            onChange={setQuestion}
            onSubmit={() => sendMessage()}
            disabled={loading}
            placeholder="What anti-sickness medication can I prescribe in pregnancy?"
          />
          {disclaimers}
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Scrollable messages area */}
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl space-y-6 px-4 py-6 sm:px-6">
          {messages.map((message) => {
            const isLiveStreaming = loading && message.id === liveIdRef.current;
            const hasCitations = !!message.citations?.length;
            const answerStillStreaming = isLiveStreaming && !hasCitations;
            const showThreeBoxLayout =
              message.role === "assistant" && !!message.content && (!isLiveStreaming || hasCitations);

            // User messages
            if (message.role === "user") {
              return (
                <div key={message.id}>
                  <Message role="user">
                    <Response markdown={message.content} />
                  </Message>
                </div>
              );
            }

            // Streaming assistant message — show trace + thinking
            if (!showThreeBoxLayout) {
              return (
                <div key={message.id} ref={message.id === liveIdRef.current ? liveMessageRef : undefined}>
                  <Message role="assistant">
                    {/* Agent Trace — the trace panel has its own "Agent is thinking..." header */}
                    {isLiveStreaming && (traceSteps.length > 0 || traceRunning) && (
                      <div className="mb-3">
                        <AgentTrace
                          steps={traceSteps}
                          isRunning={traceRunning}
                          currentStage={currentTraceName}
                        />
                      </div>
                    )}
                    {/* Show content once answer starts streaming */}
                    {message.content ? (
                      <Response
                        markdown={message.content}
                        streaming={answerStillStreaming}
                      />
                    ) : (
                      /* Only show the basic "Thinking..." if no trace steps yet (before first SSE event) */
                      isLiveStreaming && traceSteps.length === 0 && !traceRunning ? (
                        <Thinking label="Connecting..." />
                      ) : null
                    )}
                  </Message>
                </div>
              );
            }

            // 3-box layout — completed messages
            const currentRating = ratings[message.id] ?? null;

            return (
              <div key={message.id} className="flex w-full justify-start gap-3" ref={message.id === liveIdRef.current ? liveMessageRef : undefined}>
                {/* Avatar */}
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-primary to-primary/70 shadow-sm">
                  <span className="text-xs font-bold text-white">C</span>
                </div>

                {/* Three boxes */}
                <div className="min-w-0 flex-1 space-y-3" style={{ maxWidth: "calc(100% - 44px)" }}>

                  {/* Agent Trace (collapsed after completion) */}
                  {traceSteps.length > 0 && message.id === messages.filter(m => m.role === "assistant").pop()?.id && (
                    <AgentTrace
                      steps={traceSteps}
                      isRunning={false}
                      currentStage=""
                    />
                  )}

                  {/* Box 1: Answer */}
                  <div className="rounded-2xl rounded-tl-sm border-2 border-primary/50 bg-primary/15 px-5 py-4 shadow-sm">
                    <Response markdown={message.content} streaming={answerStillStreaming} />
                    {message.abstained && (
                      <p className="mt-3 rounded-lg border border-accent/30 bg-accent/5 p-2.5 text-xs text-foreground">
                        <span className="font-semibold text-accent">Note:</span>{" "}
                        {message.abstainReason || "Insufficient evidence to answer with confidence."}
                      </p>
                    )}
                  </div>

                  {/* Box 2: Guidelines */}
                  {!!message.citations?.length && (
                    <div className="rounded-xl border-2 border-primary/50 bg-white px-5 py-4 shadow-sm">
                      <p className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-primary/70">
                        What the Guidelines say
                      </p>
                      <div className="space-y-3">
                        {message.citations.map((citation, idx) => {
                          const meta = citation.doc_metadata as Record<string, string | undefined> | undefined;
                          const pdfUrl = meta?.pdf_url;
                          const guidelineTitle = meta?.guideline_title || citation.filename;
                          const reference = meta?.reference;

                          let sectionDisplay = citation.section_path || "";
                          if (sectionDisplay) {
                            const lower = sectionDisplay.toLowerCase();
                            const prefixWithRef = reference
                              ? `${guidelineTitle} (${reference})`
                              : guidelineTitle;
                            if (lower.startsWith(prefixWithRef.toLowerCase())) {
                              sectionDisplay = sectionDisplay.slice(prefixWithRef.length).replace(/^\s*>\s*/, "").trim();
                            } else if (lower.startsWith(guidelineTitle.toLowerCase())) {
                              sectionDisplay = sectionDisplay.slice(guidelineTitle.length).replace(/^\s*\(.*?\)\s*>\s*/, "").replace(/^\s*>\s*/, "").trim();
                            }
                          }

                          return (
                            <div
                              key={citation.chunk_id}
                              className="rounded-xl border border-primary/15 bg-white shadow-sm overflow-hidden"
                            >
                              <div className="border-b border-primary/10 bg-primary/5 px-4 py-2">
                                <p className="text-[10px] font-semibold uppercase tracking-widest text-primary/60">
                                  Source {idx + 1}
                                </p>
                              </div>
                              <div className="space-y-2 px-4 py-3">
                                {(() => {
                                  const isExpanded = expandedCitations.has(citation.chunk_id);
                                  const previewText = stripMarkdown(citation.snippet);
                                  const fullText = stripMarkdown(citation.full_snippet || citation.snippet);
                                  const hasMore = fullText.length > previewText.length;
                                  return (
                                    <>
                                      <blockquote className="border-l-[3px] border-accent pl-3 text-sm italic leading-relaxed text-foreground/85 whitespace-pre-line">
                                        &ldquo;{isExpanded ? fullText : previewText}&rdquo;
                                      </blockquote>
                                      {hasMore && (
                                        <button
                                          type="button"
                                          onClick={() => toggleCitationExpanded(citation.chunk_id)}
                                          className="text-xs font-medium text-primary/80 hover:text-primary hover:underline"
                                        >
                                          {isExpanded ? "Show less ▲" : "Show full source ▼"}
                                        </button>
                                      )}
                                    </>
                                  );
                                })()}
                                <div>
                                  <p className="text-xs font-bold text-foreground">
                                    {guidelineTitle}{reference && ` (${reference})`}
                                  </p>
                                  {sectionDisplay && (
                                    <p className="text-xs text-muted-foreground">› {sectionDisplay}</p>
                                  )}
                                </div>
                                {pdfUrl && (
                                  <div className="flex justify-end pt-0.5">
                                    <a
                                      href={pdfUrl}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="rounded-md border border-accent/30 bg-accent/5 px-2.5 py-1 text-xs font-medium text-accent transition-colors hover:bg-accent/10"
                                      onClick={(e) => e.stopPropagation()}
                                    >
                                      View guideline →
                                    </a>
                                  </div>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Feedback bar */}
                  <div className="flex items-center gap-3 rounded-xl border border-border/70 bg-card px-4 py-3 shadow-sm">
                    <p className="flex-1 text-sm font-medium text-foreground/80">
                      Help us improve — was this helpful?
                    </p>
                    <button
                      type="button"
                      onClick={() => setRating(message.id, message.messageId, "up")}
                      className={cn(
                        "flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm font-medium transition-all duration-200 active:scale-95",
                        currentRating === "up"
                          ? "scale-105 border-primary/40 bg-primary/10 text-primary shadow-sm"
                          : "scale-100 border-border text-foreground/70 hover:border-primary/40 hover:bg-primary/8 hover:text-primary"
                      )}
                    >
                      <ThumbsUp className="h-4 w-4" />
                      Yes
                    </button>
                    <button
                      type="button"
                      onClick={() => setRating(message.id, message.messageId, "down")}
                      className={cn(
                        "flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm font-medium transition-all duration-200 active:scale-95",
                        currentRating === "down"
                          ? "scale-105 border-destructive/40 bg-destructive/10 text-destructive shadow-sm"
                          : "scale-100 border-border text-foreground/70 hover:border-destructive/40 hover:bg-destructive/5 hover:text-destructive"
                      )}
                    >
                      <ThumbsDown className="h-4 w-4" />
                      No
                    </button>
                  </div>

                  {/* Box 3: Follow-up questions — hidden for now.
                      The backend still generates them and they still arrive over SSE
                      (`follow_up_questions_ready`) into `message.followUpQuestions`,
                      so restoring this section is a pure UI change. */}
                </div>
              </div>
            );
          })}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Fixed input area */}
      <div className="shrink-0 border-t border-border/60 bg-white/80 px-4 py-4 backdrop-blur sm:px-6">
        <div className="mx-auto max-w-3xl">
          <PromptInput
            value={question}
            onChange={setQuestion}
            onSubmit={() => sendMessage()}
            disabled={loading}
            placeholder="Ask about healthcare guidelines..."
          />
          {disclaimers}
        </div>
      </div>
    </div>
  );
}
