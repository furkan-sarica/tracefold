import type { Citation, DocumentItem, EvaluationResult, RuntimeStatus, TranslationStatus } from "./types";

const API_BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function errorFrom(response: Response): Promise<ApiError> {
  try {
    const body = await response.json();
    const detail = body.detail || body.message;
    const message = typeof detail === "string" ? detail : detail?.message;
    return new ApiError(response.status, message || `HTTP ${response.status}`);
  } catch {
    return new ApiError(response.status, `HTTP ${response.status}`);
  }
}

async function json<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) throw await errorFrom(response);
  return response.json() as Promise<T>;
}

export async function getRuntime(): Promise<RuntimeStatus> {
  const raw = await json<Record<string, unknown>>("/api/runtime");
  const chat = raw.chat_model as Record<string, unknown> | string | undefined;
  const embedding = raw.embedding_model as Record<string, unknown> | string | undefined;
  const translation = raw.translation && typeof raw.translation === "object"
    ? raw.translation as TranslationStatus
    : undefined;
  return {
    ...raw,
    chat_model: typeof chat === "object" ? String(chat.alias || chat.variant_id || "—") : chat,
    embedding_model: typeof embedding === "object" ? String(embedding.alias || embedding.variant_id || "—") : embedding,
    chat_variant: typeof chat === "object" && chat.variant_id ? String(chat.variant_id) : undefined,
    embedding_variant: typeof embedding === "object" && embedding.variant_id ? String(embedding.variant_id) : undefined,
    execution_provider: typeof chat === "object" && chat.execution_provider
      ? String(chat.execution_provider)
      : typeof embedding === "object" && embedding.execution_provider
        ? String(embedding.execution_provider)
        : undefined,
    device: typeof chat === "object" ? String(chat.device || "—") : undefined,
    translation,
    message: String(raw.message || raw.error || ""),
  } as RuntimeStatus;
}

export async function getDocuments(): Promise<{ documents: DocumentItem[]; chunks?: number }> {
  const raw = await json<{ documents?: Array<DocumentItem | string>; chunks?: number }>("/api/documents");
  const documents = (raw.documents || []).map((doc, index) =>
    typeof doc === "string"
      ? { id: `${doc}-${index}`, filename: doc, status: "ready" }
      : {
          ...doc,
          id: String(doc.id ?? `document-${index}`),
          filename: doc.filename || (doc as DocumentItem & { name?: string }).name || "—",
          chunks: doc.chunks ?? (doc as DocumentItem & { chunk_count?: number }).chunk_count,
          size: doc.size ?? (doc as DocumentItem & { size_bytes?: number }).size_bytes,
        },
  );
  return { documents, chunks: raw.chunks };
}

export async function uploadDocument(file: File): Promise<Record<string, unknown>> {
  const data = new FormData();
  data.append("file", file);
  return json<Record<string, unknown>>("/api/documents", { method: "POST", body: data });
}

export async function deleteDocument(id: string): Promise<void> {
  await json(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export async function getJob(id: string): Promise<Record<string, unknown>> {
  return json(`/api/jobs/${encodeURIComponent(id)}`);
}

function normalizeEvaluation(raw: Record<string, unknown>): EvaluationResult {
  const metrics = (raw.metrics || {}) as Record<string, unknown>;
  const cases = Array.isArray(raw.cases) ? raw.cases as Array<Record<string, unknown>> : [];
  const optionalNumber = (value: unknown) => value === undefined || value === null || Number.isNaN(Number(value)) ? undefined : Number(value);
  const passed = cases.filter((item) => item.passed === true).length;
  return {
    ...raw,
    total_cases: optionalNumber(metrics.total_cases ?? raw.total_cases),
    passed_cases: optionalNumber(metrics.passed_cases ?? raw.passed_cases) ?? (cases.length ? passed : undefined),
    hit_at_5: optionalNumber(metrics.hit_at_k ?? raw.hit_at_5),
    source_recall_at_5: optionalNumber(metrics.source_recall_at_k ?? raw.source_recall_at_5),
    citation_accuracy: optionalNumber(metrics.citation_accuracy ?? raw.citation_accuracy),
    required_fact_recall: optionalNumber(metrics.required_fact_recall ?? raw.required_fact_recall),
    refusal_accuracy: optionalNumber(metrics.refusal_accuracy ?? raw.refusal_accuracy),
    retrieval_p95_ms: optionalNumber(metrics.retrieval_p95_ms ?? raw.retrieval_p95_ms),
    cases: cases as EvaluationResult["cases"],
  } as EvaluationResult;
}

export async function getLatestEvaluation(): Promise<EvaluationResult | null> {
  try {
    return normalizeEvaluation(await json<Record<string, unknown>>("/api/evaluations/latest"));
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return null;
    throw error;
  }
}

export async function runEvaluation(): Promise<EvaluationResult> {
  const raw = await json<Record<string, unknown>>("/api/evaluations/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ top_k: 5 }),
  });
  return normalizeEvaluation(raw);
}

type StreamHandlers = {
  onToken: (token: string) => void;
  onCitations: (citations: Citation[]) => void;
  onRetrieval?: () => void;
  onDone?: (payload: Record<string, unknown>) => void;
};

function normalizeCitation(value: Record<string, unknown>, index: number): Citation {
  const rawScore = value.score ?? value.similarity;
  const rawPage = value.page ?? value.page_start;
  const rawSection = value.section;
  return {
    id: String(value.id ?? value.chunk_id ?? `citation-${index}`),
    index: Number(value.index ?? index + 1),
    source: String(value.source ?? value.filename ?? value.document ?? "—"),
    excerpt: String(value.excerpt ?? value.text ?? value.content ?? "—"),
    page: typeof rawPage === "number" || typeof rawPage === "string" ? rawPage : undefined,
    section: typeof rawSection === "string" && rawSection.trim() ? rawSection : undefined,
    score: typeof rawScore === "number" ? rawScore : Number(rawScore) || undefined,
    chunk_id: value.chunk_id as string | undefined,
  };
}

export async function streamChat(
  question: string,
  topK: number,
  locale: "en" | "tr",
  handlers: StreamHandlers,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question, top_k: topK, locale }),
  });
  if (!response.ok) throw await errorFrom(response);

  const contentType = response.headers.get("content-type") || "";
  if (!response.body || contentType.includes("application/json")) {
    const payload = (await response.json()) as Record<string, unknown>;
    handlers.onToken(String(payload.answer ?? payload.message ?? ""));
    const sources = Array.isArray(payload.citations) ? payload.citations : payload.sources;
    if (Array.isArray(sources)) handlers.onCitations(sources.map((item, i) => normalizeCitation(item as Record<string, unknown>, i)));
    handlers.onDone?.(payload);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const consume = (block: string) => {
    const lines = block.split(/\r?\n/);
    let event = "message";
    const data: string[] = [];
    for (const line of lines) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
    }
    if (!data.length && block.trim().startsWith("{")) data.push(block.trim());
    if (!data.length) return;
    const joined = data.join("\n");
    let payload: unknown = joined;
    try { payload = JSON.parse(joined); } catch { /* token may be plain text */ }
    const object = typeof payload === "object" && payload !== null ? payload as Record<string, unknown> : {};
    const type = String(object.type ?? event);
    if (type === "token" || type === "message") handlers.onToken(String(object.token ?? object.text ?? object.delta ?? object.content ?? payload));
    if (type === "retrieval") handlers.onRetrieval?.();
    if (type === "citations" || type === "sources") {
      const items = (object.citations ?? object.sources ?? []) as Array<Record<string, unknown>>;
      handlers.onCitations(items.map(normalizeCitation));
    }
    if (type === "done") handlers.onDone?.(object);
    if (type === "error") throw new Error(String(object.message ?? object.detail ?? "Generation failed"));
  };

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() || "";
    blocks.forEach(consume);
    if (done) break;
  }
  if (buffer.trim()) consume(buffer);
}
