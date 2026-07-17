export type View = "ask" | "library" | "setup" | "quality";

export interface RuntimeStatus {
  status?: string;
  ready?: boolean;
  available?: boolean;
  runtime?: string;
  model?: string;
  chat_model?: string;
  embedding_model?: string;
  chat_variant?: string;
  embedding_variant?: string;
  execution_provider?: string;
  device?: string;
  message?: string;
  documents_indexed?: number;
  chunks_indexed?: number;
}

export interface DocumentItem {
  id: string;
  filename: string;
  status?: string;
  chunks?: number;
  size?: number;
  created_at?: string;
  error?: string;
}

export interface Citation {
  id: string;
  index: number;
  source: string;
  excerpt: string;
  page?: number | string;
  section?: string;
  score?: number;
  chunk_id?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  status?: "streaming" | "done" | "error" | "refusal";
}

export interface EvaluationResult {
  id?: string;
  status?: string;
  created_at?: string;
  total_cases?: number;
  passed_cases?: number;
  hit_at_5?: number;
  source_recall_at_5?: number;
  citation_accuracy?: number;
  required_fact_recall?: number;
  refusal_accuracy?: number;
  retrieval_p95_ms?: number;
  cases?: Array<{ question?: string; status?: string; passed?: boolean; answerable?: boolean }>;
  message?: string;
}

export interface UploadTask {
  id: string;
  filename: string;
  status: "queued" | "uploading" | "indexing" | "ready" | "error";
  progress: number;
  message?: string;
}
