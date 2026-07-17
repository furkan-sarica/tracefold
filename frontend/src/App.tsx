import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  BookOpen,
  Check,
  ChevronRight,
  CircleAlert,
  CircleCheck,
  Database,
  File,
  FileSearch,
  Gauge,
  HardDrive,
  Library,
  LoaderCircle,
  Menu,
  MessageSquareText,
  PanelRightClose,
  Play,
  Plus,
  Search,
  Send,
  Server,
  ShieldCheck,
  Sparkles,
  Trash2,
  UploadCloud,
  X,
} from "lucide-react";
import {
  deleteDocument,
  getDocuments,
  getJob,
  getLatestEvaluation,
  getRuntime,
  runEvaluation,
  streamChat,
  uploadDocument,
} from "./api";
import type {
  ChatMessage,
  Citation,
  DocumentItem,
  EvaluationResult,
  RuntimeStatus,
  UploadTask,
  View,
} from "./types";

const navItems: Array<{ id: View; label: string; icon: typeof Search }> = [
  { id: "ask", label: "Ask", icon: MessageSquareText },
  { id: "library", label: "Library", icon: Library },
  { id: "setup", label: "Setup", icon: Server },
  { id: "quality", label: "Quality", icon: Gauge },
];

const starterQuestions = [
  "What should I know before the first day?",
  "Which equipment do participants need to bring?",
  "What is the emergency contact procedure?",
];

const uid = () => Math.random().toString(36).slice(2, 10);

function runtimeReady(runtime: RuntimeStatus | null) {
  if (!runtime) return false;
  if (runtime.ready !== undefined) return runtime.ready === true;
  return ["ready", "healthy", "online"].includes(String(runtime.status || "").toLowerCase());
}

function runtimeAvailable(runtime: RuntimeStatus | null) {
  return runtimeReady(runtime) || runtime?.available === true;
}

function statusLabel(runtime: RuntimeStatus | null, loading: boolean) {
  if (loading) return "Checking runtime";
  if (!runtime) return "Runtime unavailable";
  if (runtimeReady(runtime)) return "Foundry Local ready";
  if (runtimeAvailable(runtime)) return "Foundry Local available";
  return runtime.message || "Setup required";
}

function formatBytes(value?: number) {
  if (!value) return "—";
  const units = ["B", "KB", "MB", "GB"];
  const rank = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / 1024 ** rank).toFixed(rank ? 1 : 0)} ${units[rank]}`;
}

function formatPercent(value?: number) {
  if (value === undefined || value === null) return "—";
  const normalized = value <= 1 ? value * 100 : value;
  return `${Math.round(normalized)}%`;
}

function formatDate(value?: string) {
  if (!value) return "Not recorded";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : new Intl.DateTimeFormat("en", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(date);
}

function formatExcerpt(value: string) {
  return value.replace(/\*\*(.*?)\*\*/g, "$1").replace(/`([^`]+)`/g, "$1");
}

function Brand() {
  return (
    <button className="brand" onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })} aria-label="Tracefold home">
      <span className="brand-mark" aria-hidden="true"><span /></span>
      <span className="brand-copy"><strong>Tracefold</strong><small>Answers you can trace.</small></span>
    </button>
  );
}

function EmptyState({ icon: Icon, title, copy, action }: { icon: typeof Search; title: string; copy: string; action?: React.ReactNode }) {
  return (
    <div className="empty-state">
      <span className="empty-icon"><Icon size={21} strokeWidth={1.7} /></span>
      <h3>{title}</h3>
      <p>{copy}</p>
      {action}
    </div>
  );
}

function App() {
  const [view, setView] = useState<View>("ask");
  const [mobileNav, setMobileNav] = useState(false);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [runtimeLoading, setRuntimeLoading] = useState(true);
  const [runtimeError, setRuntimeError] = useState("");
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [documentError, setDocumentError] = useState("");
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [uploads, setUploads] = useState<UploadTask[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [asking, setAsking] = useState(false);
  const [askError, setAskError] = useState("");
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationResult | null>(null);
  const [evaluationLoading, setEvaluationLoading] = useState(true);
  const [evaluationRunning, setEvaluationRunning] = useState(false);
  const [evaluationError, setEvaluationError] = useState("");
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const askInput = useRef<HTMLTextAreaElement>(null);
  const transcriptEnd = useRef<HTMLDivElement>(null);
  const evidenceCloseButton = useRef<HTMLButtonElement>(null);
  const evidenceTrigger = useRef<HTMLElement | null>(null);

  const openCitation = useCallback((citation: Citation) => {
    evidenceTrigger.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setSelectedCitation(citation);
  }, []);

  const closeCitation = useCallback(() => {
    setSelectedCitation(null);
    window.requestAnimationFrame(() => evidenceTrigger.current?.focus());
  }, []);

  const refreshRuntime = useCallback(async () => {
    setRuntimeLoading(true);
    setRuntimeError("");
    try { setRuntime(await getRuntime()); }
    catch (error) { setRuntime(null); setRuntimeError(error instanceof Error ? error.message : "Runtime check failed"); }
    finally { setRuntimeLoading(false); }
  }, []);

  const refreshDocuments = useCallback(async () => {
    setDocumentsLoading(true);
    setDocumentError("");
    try { setDocuments((await getDocuments()).documents); }
    catch (error) { setDocumentError(error instanceof Error ? error.message : "Could not load documents"); }
    finally { setDocumentsLoading(false); }
  }, []);

  const refreshEvaluation = useCallback(async () => {
    setEvaluationLoading(true);
    setEvaluationError("");
    try { setEvaluation(await getLatestEvaluation()); }
    catch (error) { setEvaluationError(error instanceof Error ? error.message : "Could not load results"); }
    finally { setEvaluationLoading(false); }
  }, []);

  useEffect(() => { void Promise.all([refreshRuntime(), refreshDocuments(), refreshEvaluation()]); }, [refreshRuntime, refreshDocuments, refreshEvaluation]);
  useEffect(() => { transcriptEnd.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }); }, [messages]);
  useEffect(() => {
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (selectedCitation) closeCitation();
        setMobileNav(false);
      }
      if (event.key === "Tab" && selectedCitation) {
        event.preventDefault();
        evidenceCloseButton.current?.focus();
      }
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [closeCitation, selectedCitation]);

  const navigate = (next: View) => {
    setView(next);
    setMobileNav(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const pollJob = async (jobId: string, taskId: string) => {
    for (let attempt = 0; attempt < 120; attempt += 1) {
      await new Promise((resolve) => window.setTimeout(resolve, 900));
      const job = await getJob(jobId);
      const status = String(job.status || "indexing").toLowerCase();
      const rawProgress = Number(job.progress ?? Math.min(96, 15 + attempt * 3));
      const progress = rawProgress <= 1 ? rawProgress * 100 : rawProgress;
      setUploads((current) => current.map((item) => item.id === taskId ? { ...item, status: status === "failed" ? "error" : status === "ready" || status === "completed" ? "ready" : "indexing", progress, message: String(job.message || "") } : item));
      if (["ready", "completed", "failed", "error"].includes(status)) {
        if (["ready", "completed"].includes(status)) await refreshDocuments();
        return;
      }
    }
    throw new Error("Indexing is taking longer than expected. You can safely refresh the library.");
  };

  const handleFiles = async (files: File[]) => {
    const supported = files.filter((file) => /\.(pdf|docx|txt|md)$/i.test(file.name));
    if (!supported.length) {
      setDocumentError("Choose PDF, DOCX, TXT, or Markdown files.");
      return;
    }
    setDocumentError(files.length !== supported.length ? "Unsupported files were skipped." : "");
    const tasks = supported.map<UploadTask>((file) => ({ id: uid(), filename: file.name, status: "queued", progress: 0 }));
    setUploads((current) => [...tasks, ...current]);

    for (let index = 0; index < supported.length; index += 1) {
      const file = supported[index];
      const task = tasks[index];
      setUploads((current) => current.map((item) => item.id === task.id ? { ...item, status: "uploading", progress: 8 } : item));
      try {
        const response = await uploadDocument(file);
        const jobId = String(response.job_id ?? response.id ?? "");
        if (jobId && String(response.status || "").toLowerCase() !== "ready") {
          setUploads((current) => current.map((item) => item.id === task.id ? { ...item, status: "indexing", progress: 18 } : item));
          await pollJob(jobId, task.id);
        } else {
          setUploads((current) => current.map((item) => item.id === task.id ? { ...item, status: "ready", progress: 100 } : item));
          await refreshDocuments();
        }
      } catch (error) {
        setUploads((current) => current.map((item) => item.id === task.id ? { ...item, status: "error", progress: 100, message: error instanceof Error ? error.message : "Upload failed" } : item));
      }
    }
    void refreshRuntime();
  };

  const removeDocument = async (document: DocumentItem) => {
    if (!window.confirm(`Remove “${document.filename}” from the local library?`)) return;
    try { await deleteDocument(document.id); await refreshDocuments(); }
    catch (error) { setDocumentError(error instanceof Error ? error.message : "Could not remove document"); }
  };

  const ask = async (override?: string) => {
    const clean = (override ?? question).trim();
    if (!clean || asking) return;
    const userMessage: ChatMessage = { id: uid(), role: "user", content: clean };
    const answerId = uid();
    setMessages((current) => [...current, userMessage, { id: answerId, role: "assistant", content: "", status: "streaming" }]);
    setQuestion("");
    setAskError("");
    setAsking(true);
    let content = "";
    let citations: Citation[] = [];
    try {
      await streamChat(clean, topK, {
        onRetrieval: () => setMessages((current) => current.map((message) => message.id === answerId && !message.content ? { ...message, content: "Searching your library…" } : message)),
        onToken: (token) => {
          if (content === "" && token !== "Searching your library…") content = token;
          else content += token;
          setMessages((current) => current.map((message) => message.id === answerId ? { ...message, content } : message));
        },
        onCitations: (items) => {
          citations = items;
          setMessages((current) => current.map((message) => message.id === answerId ? { ...message, citations: items } : message));
        },
      });
      const refusal = /not enough (information|evidence)|cannot answer|couldn.?t find|not found in/i.test(content);
      setMessages((current) => current.map((message) => message.id === answerId ? { ...message, content: content || "I couldn’t find enough evidence in your library to answer that.", citations, status: refusal || !content ? "refusal" : "done" } : message));
    } catch (error) {
      const text = error instanceof Error ? error.message : "The answer could not be generated.";
      setAskError(text);
      setMessages((current) => current.map((message) => message.id === answerId ? { ...message, content: text, status: "error" } : message));
    } finally {
      setAsking(false);
      window.setTimeout(() => askInput.current?.focus(), 50);
    }
  };

  const startEvaluation = async () => {
    setEvaluationRunning(true);
    setEvaluationError("");
    try { setEvaluation(await runEvaluation()); }
    catch (error) { setEvaluationError(error instanceof Error ? error.message : "Evaluation failed"); }
    finally { setEvaluationRunning(false); }
  };

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to content</a>
      <header className="site-header">
        <div className="header-inner">
          <Brand />
          <nav className="desktop-nav" aria-label="Primary navigation">
            {navItems.map((item) => <button key={item.id} className={view === item.id ? "active" : ""} onClick={() => navigate(item.id)}>{item.label}</button>)}
          </nav>
          <div className="header-meta">
            <button className={`runtime-chip ${runtimeAvailable(runtime) ? "ready" : ""}`} onClick={() => navigate("setup")}>
              <span className="status-dot" />
              {runtimeLoading ? "Checking" : runtimeReady(runtime) ? "Local & ready" : runtimeAvailable(runtime) ? "Local available" : "Setup needed"}
            </button>
            <button className="mobile-menu" onClick={() => setMobileNav(!mobileNav)} aria-expanded={mobileNav} aria-label="Open navigation"><Menu size={21} /></button>
          </div>
        </div>
        {mobileNav && <nav className="mobile-nav" aria-label="Mobile navigation">{navItems.map((item) => { const Icon = item.icon; return <button key={item.id} className={view === item.id ? "active" : ""} onClick={() => navigate(item.id)}><Icon size={18} />{item.label}</button>; })}</nav>}
      </header>

      <main id="main-content">
        {view === "ask" && (
          <section className="ask-page">
            <div className="hero-copy">
              <div className="eyebrow"><ShieldCheck size={14} /> Private by design · Grounded by evidence</div>
              <h1>Ask your documents.<br /><em>Inspect the proof.</em></h1>
              <p>Tracefold finds the most relevant passages in your local library, then answers with citations you can open and verify.</p>
            </div>

            <div className="ask-workspace">
              {messages.length === 0 ? (
                <div className="ask-empty">
                  <div className="trace-orbit" aria-hidden="true"><span /><i /><b /></div>
                  <h2>What do you want to trace?</h2>
                  <p>{documents.length ? `${documents.length} local document${documents.length === 1 ? " is" : "s are"} ready to search.` : "Add documents to your library, then ask a grounded question."}</p>
                  <div className="suggestions">
                    {starterQuestions.map((item) => <button key={item} onClick={() => void ask(item)} disabled={!documents.length || asking}>{item}<ArrowRight size={15} /></button>)}
                  </div>
                </div>
              ) : (
                <div className="transcript" aria-live="polite">
                  {messages.map((message) => (
                    <article key={message.id} className={`message ${message.role} ${message.status || ""}`}>
                      <div className="message-label">{message.role === "user" ? "You" : "Tracefold"}</div>
                      <div className="message-body">
                        {message.status === "streaming" && !message.content ? <span className="thinking"><i /><i /><i /><span className="sr-only">Generating answer</span></span> : <p>{message.content}</p>}
                        {message.status === "refusal" && <div className="refusal-note"><CircleAlert size={16} /> Tracefold won’t guess when the evidence is insufficient.</div>}
                        {!!message.citations?.length && <div className="citation-row" aria-label="Sources">{message.citations.map((citation) => <button key={citation.id} onClick={() => openCitation(citation)}><span>{citation.index}</span>{citation.source}</button>)}</div>}
                      </div>
                    </article>
                  ))}
                  <div ref={transcriptEnd} />
                </div>
              )}

              <div className="composer-wrap">
                {askError && <div className="inline-error"><CircleAlert size={15} />{askError}</div>}
                <div className="composer">
                  <textarea ref={askInput} value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void ask(); } }} placeholder={documents.length ? "Ask a question grounded in your library…" : "Add a document to begin…"} disabled={asking || !documents.length} rows={1} aria-label="Ask your library" />
                  <button className="send-button" onClick={() => void ask()} disabled={!question.trim() || asking || !documents.length} aria-label="Send question">{asking ? <LoaderCircle className="spin" size={19} /> : <Send size={18} />}</button>
                </div>
                <div className="composer-meta">
                  <button onClick={() => navigate("library")}><BookOpen size={14} />{documents.length ? `${documents.length} document${documents.length === 1 ? "" : "s"}` : "Add documents"}</button>
                  <label>Evidence depth <select value={topK} onChange={(event) => setTopK(Number(event.target.value))} aria-label="Evidence depth">{[3, 4, 5, 6, 8].map((value) => <option key={value} value={value}>{value} passages</option>)}</select></label>
                  <span><HardDrive size={13} /> Stays on this device</span>
                </div>
              </div>
            </div>

            <div className="principles-strip">
              <div><Search size={18} /><span><strong>Retrieve</strong><small>Find the closest passages</small></span></div>
              <ChevronRight size={16} className="principle-arrow" />
              <div><Sparkles size={18} /><span><strong>Answer</strong><small>Use only retrieved context</small></span></div>
              <ChevronRight size={16} className="principle-arrow" />
              <div><FileSearch size={18} /><span><strong>Verify</strong><small>Open the exact evidence</small></span></div>
            </div>
          </section>
        )}

        {view === "library" && (
          <section className="page-section">
            <div className="page-heading"><div><span className="section-kicker">Local knowledge</span><h1>Your library</h1><p>Every file is parsed, embedded, and stored on this device.</p></div><button className="primary-button" onClick={() => fileInput.current?.click()}><Plus size={17} />Add documents</button></div>
            <input ref={fileInput} type="file" multiple accept=".pdf,.docx,.txt,.md" hidden onChange={(event) => { void handleFiles(Array.from(event.target.files || [])); event.target.value = ""; }} />
            <div className={`dropzone ${dragging ? "dragging" : ""}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); void handleFiles(Array.from(event.dataTransfer.files)); }} onClick={() => fileInput.current?.click()} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") fileInput.current?.click(); }} role="button" tabIndex={0}>
              <span className="drop-icon"><UploadCloud size={24} /></span><div><strong>Drop your source material here</strong><small>PDF, DOCX, TXT, or Markdown · multiple files supported</small></div><span className="browse-label">Browse files</span>
            </div>
            {documentError && <div className="banner error"><CircleAlert size={17} /><span>{documentError}</span><button onClick={() => setDocumentError("")} aria-label="Dismiss"><X size={16} /></button></div>}
            {!!uploads.length && <section className="panel upload-panel"><div className="panel-heading"><div><span className="section-kicker">Ingestion queue</span><h2>Processing</h2></div><button className="text-button" onClick={() => setUploads((current) => current.filter((item) => !["ready", "error"].includes(item.status)))}>Clear completed</button></div><div className="upload-list">{uploads.map((task) => <div className="upload-item" key={task.id}><span className={`file-icon ${task.status}`}><File size={18} /></span><div><strong>{task.filename}</strong><small>{task.message || ({ queued: "Waiting", uploading: "Uploading securely", indexing: "Creating searchable passages", ready: "Ready to search", error: "Could not be indexed" }[task.status])}</small><span className="progress"><i style={{ width: `${task.progress}%` }} /></span></div><span className={`task-status ${task.status}`}>{task.status === "ready" ? <Check size={15} /> : task.status === "error" ? <CircleAlert size={15} /> : <LoaderCircle className={task.status !== "queued" ? "spin" : ""} size={15} />}{task.status}</span></div>)}</div></section>}
            <section className="panel library-panel">
              <div className="panel-heading"><div><span className="section-kicker">Searchable sources</span><h2>Indexed documents</h2></div><span className="count-pill">{documents.length} docs · {documents.reduce((total, document) => total + (document.chunks || 0), 0)} passages</span></div>
              {documentsLoading ? <div className="skeleton-list"><i /><i /><i /></div> : documentError && !documents.length ? <EmptyState icon={CircleAlert} title="Library unavailable" copy={documentError} action={<button className="secondary-button" onClick={() => void refreshDocuments()}>Try again</button>} /> : !documents.length ? <EmptyState icon={Library} title="Your library is empty" copy="Add 5–10 source documents to create a useful, focused knowledge base." action={<button className="secondary-button" onClick={() => fileInput.current?.click()}><Plus size={16} />Add your first document</button>} /> : <div className="document-list">{documents.map((document) => <article key={document.id}><span className="document-icon"><File size={20} /></span><div className="document-name"><strong>{document.filename}</strong><span>{document.chunks !== undefined ? `${document.chunks} passages` : "Indexed"}{document.size ? ` · ${formatBytes(document.size)}` : ""}</span></div><span className={`document-state ${document.status === "error" ? "error" : ""}`}><i />{document.status === "error" ? "Issue" : "Searchable"}</span><time>{formatDate(document.created_at)}</time><button className="icon-button danger" onClick={() => void removeDocument(document)} aria-label={`Remove ${document.filename}`}><Trash2 size={17} /></button></article>)}</div>}
            </section>
          </section>
        )}

        {view === "setup" && (
          <section className="page-section narrow">
            <div className="page-heading"><div><span className="section-kicker">System readiness</span><h1>Local setup</h1><p>Tracefold keeps inference and retrieval on your machine.</p></div><button className="secondary-button" onClick={() => void refreshRuntime()} disabled={runtimeLoading}>{runtimeLoading ? <LoaderCircle className="spin" size={17} /> : <Search size={17} />}Check again</button></div>
            <section className={`runtime-hero ${runtimeAvailable(runtime) ? "ready" : "attention"}`}>
              <span className="runtime-symbol">{runtimeAvailable(runtime) ? <CircleCheck size={29} /> : <CircleAlert size={29} />}</span>
              <div><span className="section-kicker">Current status</span><h2>{statusLabel(runtime, runtimeLoading)}</h2><p>{runtimeReady(runtime) ? "Models are loaded and your library is ready for private, offline questions." : runtimeAvailable(runtime) ? "The local models are available and will load on demand when you index or ask." : runtimeError || runtime?.message || "Start Foundry Local and make sure both required models are installed."}</p></div>
              <span className="local-badge"><ShieldCheck size={15} />On-device</span>
            </section>
            <div className="setup-grid">
              <article className="panel config-card"><div className="config-icon"><Server size={21} /></div><span>Runtime</span><strong>{runtime?.runtime || runtime?.status || "Not detected"}</strong><small>{[runtime?.device || "Local machine", runtime?.execution_provider].filter(Boolean).join(" · ")}</small></article>
              <article className="panel config-card"><div className="config-icon"><Sparkles size={21} /></div><span>Chat model</span><strong>{runtime?.chat_model || runtime?.model || "Not reported"}</strong><small>{runtime?.chat_variant || "Grounded answer generation"}</small></article>
              <article className="panel config-card"><div className="config-icon"><Database size={21} /></div><span>Embedding model</span><strong>{runtime?.embedding_model || "Not reported"}</strong><small>{runtime?.embedding_variant || "Semantic document search"}</small></article>
            </div>
            {!runtimeAvailable(runtime) && <section className="panel setup-steps"><div className="panel-heading"><div><span className="section-kicker">Readiness checklist</span><h2>Before your first question</h2></div></div><ol><li><span>1</span><div><strong>Start Foundry Local</strong><p>Launch the local runtime on this machine.</p></div></li><li><span>2</span><div><strong>Make the models available</strong><p>Install the chat and embedding models defined by the project.</p></div></li><li><span>3</span><div><strong>Run the check again</strong><p>Tracefold will confirm the models before accepting questions.</p></div></li></ol></section>}
            <section className="privacy-note"><ShieldCheck size={20} /><div><strong>Nothing leaves this machine</strong><p>Documents, vector embeddings, retrieved evidence, and generated answers remain in the local workspace.</p></div></section>
          </section>
        )}

        {view === "quality" && (
          <section className="page-section">
            <div className="page-heading"><div><span className="section-kicker">Evidence, measured</span><h1>Quality report</h1><p>Test retrieval, citation fidelity, and safe refusal against the project evaluation set.</p></div><button className="primary-button" onClick={() => void startEvaluation()} disabled={evaluationRunning || !runtimeAvailable(runtime)}>{evaluationRunning ? <LoaderCircle className="spin" size={17} /> : <Play size={16} fill="currentColor" />}{evaluationRunning ? "Running evaluation" : "Run evaluation"}</button></div>
            {evaluationError && <div className="banner error"><CircleAlert size={17} />{evaluationError}</div>}
            {evaluationLoading ? <div className="quality-grid"><div className="metric-skeleton" /><div className="metric-skeleton" /><div className="metric-skeleton" /><div className="metric-skeleton" /></div> : !evaluation ? <section className="panel"><EmptyState icon={Gauge} title="No evaluation run yet" copy="Run the built-in cases to see honest retrieval and grounding results. Tracefold will never fill this page with sample scores." action={<button className="secondary-button" onClick={() => void startEvaluation()} disabled={!runtimeAvailable(runtime)}><Play size={15} />Run the first evaluation</button>} /></section> : <>
              <div className="quality-summary"><div><span className={`quality-seal ${(evaluation.passed_cases ?? 0) === (evaluation.total_cases ?? -1) ? "pass" : ""}`}><Check size={24} /></span><div><span className="section-kicker">Latest run · {formatDate(evaluation.created_at)}</span><h2>{evaluation.passed_cases ?? "—"} of {evaluation.total_cases ?? "—"} cases passed</h2></div></div><span className="run-status">{evaluation.status || "Complete"}</span></div>
              <div className="quality-grid">
                <article className="metric-card"><span>Retrieval hit@5</span><strong>{formatPercent(evaluation.hit_at_5)}</strong><small>Answerable cases with evidence found</small></article>
                <article className="metric-card"><span>Citation validity</span><strong>{formatPercent(evaluation.citation_accuracy)}</strong><small>Answers with resolvable evidence IDs</small></article>
                <article className="metric-card"><span>Safe refusal</span><strong>{formatPercent(evaluation.refusal_accuracy)}</strong><small>Unanswerable cases correctly declined</small></article>
                <article className="metric-card"><span>Source coverage</span><strong>{formatPercent(evaluation.source_recall_at_5)}</strong><small>Expected sources represented in evidence</small></article>
              </div>
              <section className="panel latency-panel"><div className="panel-heading"><div><span className="section-kicker">Measured evidence</span><h2>Run details</h2></div></div><div className="latency-row"><div><span>Retrieval p95</span><strong>{evaluation.retrieval_p95_ms !== undefined ? `${Math.round(evaluation.retrieval_p95_ms)} ms` : "—"}</strong></div><div><span>Required fact recall</span><strong>{formatPercent(evaluation.required_fact_recall)}</strong></div></div></section>
            </>}
          </section>
        )}
      </main>

      <footer><Brand /><p>Private local retrieval. Verifiable answers.</p><span>Built for Foundry Local</span></footer>

      {selectedCitation && (
        <div className="drawer-layer" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) closeCitation(); }}>
          <aside className="evidence-drawer" role="dialog" aria-modal="true" aria-labelledby="evidence-title">
            <div className="drawer-header">
              <div><span className="section-kicker">Retrieved evidence</span><h2 id="evidence-title">Source [{selectedCitation.index}]</h2></div>
              <button ref={evidenceCloseButton} className="icon-button" autoFocus onClick={closeCitation} aria-label="Close evidence"><PanelRightClose size={20} /></button>
            </div>
            <div className="source-card"><span className="document-icon"><File size={19} /></span><div><strong>{selectedCitation.source}</strong><span>{selectedCitation.page !== undefined ? `Page ${selectedCitation.page}` : selectedCitation.section || "Document passage"}</span></div></div>
            <div className="evidence-meta"><span>Similarity <strong>{selectedCitation.score !== undefined ? formatPercent(selectedCitation.score) : "Not reported"}</strong></span>{selectedCitation.chunk_id && <span>Chunk <strong>{selectedCitation.chunk_id}</strong></span>}</div>
            <blockquote><span>Exact excerpt</span><p>{formatExcerpt(selectedCitation.excerpt)}</p></blockquote>
            <div className="drawer-foot"><ShieldCheck size={16} /><p>This passage was retrieved from your local library and supplied to the model as evidence.</p></div>
          </aside>
        </div>
      )}
    </div>
  );
}

export default App;
