import type {
  ChatHistoryResponse,
  ChatResponse,
  ChatSessionResponse,
  DocumentDeleteResponse,
  DocumentListResponse,
  IngestionListResponse,
  IngestionCreated,
  IngestionMetadata,
  IngestionStatus,
  RetrievalFilters,
  RetrievalProfile,
  RetrievalSearchResponse
} from "@/lib/types";

const DEFAULT_BASE_URL = "http://127.0.0.1:8000";

function normalizeBaseUrl(value: string | undefined) {
  return (value || DEFAULT_BASE_URL).trim().replace(/\/$/, "");
}

type RequestOptions = RequestInit & {
  baseUrl: string;
};

async function requestJson<T>(path: string, options: RequestOptions): Promise<T> {
  const { baseUrl, ...rest } = options;
  const response = await fetch(`${normalizeBaseUrl(baseUrl)}${path}`, rest);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(String(body?.detail || body?.message || response.statusText));
  }
  return body as T;
}

export async function checkHealth(baseUrl: string) {
  return requestJson<{ ok: boolean }>("/health", {
    baseUrl,
    method: "GET",
    cache: "no-store"
  });
}

export async function submitIngestion(
  baseUrl: string,
  file: File,
  metadata: IngestionMetadata
) {
  const form = new FormData();
  form.append("file", file);
  form.append("metadata", JSON.stringify(metadata));
  form.append("force_reingest", "false");
  return requestJson<IngestionCreated>("/v1/ingestions", {
    baseUrl,
    method: "POST",
    body: form
  });
}

export async function getIngestion(baseUrl: string, ingestionId: string) {
  return requestJson<IngestionStatus>(`/v1/ingestions/${ingestionId}`, {
    baseUrl,
    method: "GET",
    cache: "no-store"
  });
}

export async function listIngestions(baseUrl: string, limit = 200, offset = 0) {
  return requestJson<IngestionListResponse>(`/v1/ingestions?limit=${limit}&offset=${offset}`, {
    baseUrl,
    method: "GET",
    cache: "no-store"
  });
}

export async function listDocuments(baseUrl: string, limit = 200, offset = 0) {
  return requestJson<DocumentListResponse>(`/v1/documents?limit=${limit}&offset=${offset}`, {
    baseUrl,
    method: "GET",
    cache: "no-store"
  });
}

export async function deleteDocument(baseUrl: string, documentId: string) {
  return requestJson<DocumentDeleteResponse>(`/v1/documents/${documentId}`, {
    baseUrl,
    method: "DELETE"
  });
}

export async function createSession(baseUrl: string) {
  return requestJson<ChatSessionResponse>("/v1/chat/sessions", {
    baseUrl,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ metadata: {} })
  });
}

export async function getSessionMessages(baseUrl: string, sessionId: string) {
  return requestJson<ChatHistoryResponse>(`/v1/chat/sessions/${sessionId}/messages?limit=200&offset=0`, {
    baseUrl,
    method: "GET",
    cache: "no-store"
  });
}

export async function submitFeedback(
  baseUrl: string,
  messageId: string,
  rating: "up" | "down" | null
): Promise<void> {
  await requestJson<{ message_id: string; rating: string | null }>(
    `/v1/chat/messages/${messageId}/feedback`,
    {
      baseUrl,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rating }),
    }
  );
}

export async function chatQuery(
  baseUrl: string,
  payload: {
    session_id: string | null;
    question: string;
    filters?: RetrievalFilters;
    top_k?: number;
    include_debug?: boolean;
  }
) {
  return requestJson<ChatResponse>("/v1/chat/query", {
    baseUrl,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function searchRetrieval(
  baseUrl: string,
  payload: {
    query: string;
    filters?: RetrievalFilters;
    profile?: RetrievalProfile;
    top_k?: number;
    expand_parents?: boolean;
  }
) {
  return requestJson<RetrievalSearchResponse>("/v1/retrieval/search", {
    baseUrl,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

type StreamHandlers = {
  onStage?: (stage: string, data: Record<string, unknown>) => void;
  onAnswerDelta?: (delta: string) => void;
  onCitationsReady?: (citations: import("@/lib/types").Citation[]) => void;
  onFollowUpQuestionsReady?: (questions: string[]) => void;
};

function parseSseFrame(frame: string) {
  const lines = frame.split("\n");
  let eventName = "message";
  const dataLines: string[] = [];
  for (const line of lines) {
    if (line.startsWith("event:")) eventName = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  let payload: Record<string, unknown> = {};
  try {
    payload = dataLines.length ? JSON.parse(dataLines.join("\n")) : {};
  } catch {
    payload = {};
  }
  return { eventName, payload };
}

export async function chatQueryStream(
  baseUrl: string,
  payload: {
    session_id: string | null;
    question: string;
    filters?: RetrievalFilters;
    top_k?: number;
    include_debug?: boolean;
  },
  handlers: StreamHandlers
) {
  const response = await fetch(`${normalizeBaseUrl(baseUrl)}/v1/chat/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok || !response.body) {
    const text = await response.text().catch(() => "");
    throw new Error(text || response.statusText);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let finalResponse: ChatResponse | null = null;

  const processFrame = (frame: string) => {
    if (!frame) return;
    const { eventName, payload: eventPayload } = parseSseFrame(frame);
    if (eventName === "stage") {
      const stageName = String(eventPayload.stage || "stage");
      const data = { ...eventPayload };
      delete data.stage;
      handlers.onStage?.(stageName, data);
    } else if (eventName === "answer_delta") {
      handlers.onAnswerDelta?.(String(eventPayload.delta || ""));
    } else if (eventName === "citations_ready") {
      handlers.onCitationsReady?.(eventPayload.citations as import("@/lib/types").Citation[]);
    } else if (eventName === "follow_up_questions_ready") {
      handlers.onFollowUpQuestionsReady?.(eventPayload.follow_up_questions as string[]);
    } else if (eventName === "final_response") {
      finalResponse = eventPayload as unknown as ChatResponse;
    } else if (eventName === "error") {
      throw new Error(String(eventPayload.message || "Streaming failed"));
    } else if (eventName === "done") {
      handlers.onStage?.("done", {});
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      const tail = buffer.trim();
      if (tail) processFrame(tail);
      break;
    }
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let boundary = buffer.indexOf("\n\n");

    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");
      processFrame(frame);
    }
  }

  if (!finalResponse) {
    throw new Error("Streaming ended before final response.");
  }
  return finalResponse;
}
