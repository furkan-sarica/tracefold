import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
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
  Languages,
  Library,
  LoaderCircle,
  Menu,
  MessageSquareText,
  Monitor,
  Moon,
  PanelRightClose,
  Play,
  Plus,
  Search,
  Send,
  Server,
  Settings2,
  ShieldCheck,
  Sparkles,
  Sun,
  Trash2,
  UploadCloud,
  Waves,
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
  ManualLanguageReview,
  RuntimeStatus,
  TranslationModelStatus,
  TranslationStatus,
  UploadTask,
  View,
} from "./types";
import { copyByLanguage } from "./copy";
import type { Language } from "./copy";

type ThemePreference = "system" | "light" | "dark";
type MotionPreference = "system" | "full" | "reduced";

interface Preferences {
  language: Language;
  theme: ThemePreference;
  motion: MotionPreference;
}

const STORAGE_KEY = "tracefold.preferences.v1";
const defaultPreferences: Preferences = { language: "en", theme: "system", motion: "system" };

function loadPreferences(): Preferences {
  try {
    const saved = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "{}") as Partial<Preferences>;
    return {
      language: saved.language === "tr" ? "tr" : "en",
      theme: ["system", "light", "dark"].includes(saved.theme || "") ? saved.theme as ThemePreference : "system",
      motion: ["system", "full", "reduced"].includes(saved.motion || "") ? saved.motion as MotionPreference : "system",
    };
  } catch {
    return defaultPreferences;
  }
}

function preferredScrollBehavior(): ScrollBehavior {
  return document.documentElement.dataset.motion === "reduced" ? "auto" : "smooth";
}

const uid = () => Math.random().toString(36).slice(2, 10);

function runtimeReady(runtime: RuntimeStatus | null) {
  if (!runtime) return false;
  if (runtime.ready !== undefined) return runtime.ready === true;
  return ["ready", "healthy", "online"].includes(String(runtime.status || "").toLowerCase());
}

function runtimeAvailable(runtime: RuntimeStatus | null) {
  return runtimeReady(runtime) || runtime?.available === true;
}

function statusLabel(runtime: RuntimeStatus | null, loading: boolean, copy: typeof copyByLanguage.en) {
  if (loading) return copy.setup.checkingRuntime;
  if (!runtime) return copy.setup.unavailable;
  if (runtimeReady(runtime)) return copy.setup.ready;
  if (runtimeAvailable(runtime)) return copy.setup.available;
  return copy.locale === "en-US" && runtime.message ? runtime.message : copy.setup.required;
}

function localizedError(error: unknown, fallback: string, language: Language) {
  return language === "en" && error instanceof Error && error.message ? error.message : fallback;
}

function formatBytes(value?: number) {
  if (!value) return "—";
  const units = ["B", "KB", "MB", "GB"];
  const rank = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / 1024 ** rank).toFixed(rank ? 1 : 0)} ${units[rank]}`;
}

function formatPercent(value?: number, locale = "en-US") {
  if (value === undefined || value === null) return "—";
  const normalized = value <= 1 ? value * 100 : value;
  const rounded = Math.round(normalized);
  return locale.toLowerCase().startsWith("tr") ? `%${rounded}` : `${rounded}%`;
}

function formatDate(value: string | undefined, locale: string, notRecorded: string) {
  if (!value) return notRecorded;
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : new Intl.DateTimeFormat(locale, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(date);
}

function formatDecimal(value: number, locale: string, digits = 2) {
  return new Intl.NumberFormat(locale, { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
}

function formatExcerpt(value: string) {
  return value.replace(/\*\*(.*?)\*\*/g, "$1").replace(/`([^`]+)`/g, "$1");
}

type EvaluationCaseDetail = {
  id?: string;
  locale?: Language | string;
  paired_case_id?: string;
  type?: string;
  category?: string;
  question?: string;
  answer?: string;
  answerable?: boolean;
  passed?: boolean;
  expected_sources?: string[];
  retrieved_sources?: string[];
  cited_sources?: string[];
  hit?: boolean | null;
  source_recall?: number | null;
  citation_precision?: number | null;
  citation_correct?: boolean | null;
  citation_sentence_coverage?: number | boolean | null;
  required_fact_recall?: number | null;
  protected_fact_grounded?: boolean | null;
  language_quality?: boolean | null;
  refusal_correct?: boolean | null;
  retrieval_ms?: number;
};

type EvaluationMetrics = {
  total_cases?: number;
  passed_cases?: number;
  hit_at_k?: number;
  source_recall_at_k?: number;
  citation_accuracy?: number;
  required_fact_recall?: number;
  refusal_accuracy?: number;
  retrieval_p95_ms?: number;
  language_quality_accuracy?: number;
  citation_sentence_coverage?: number;
  protected_fact_grounding_accuracy?: number;
  bilingual_pair_accuracy?: number;
  top_k?: number;
  retrieval_threshold?: number;
  locales?: string[];
  locale_breakdown?: Record<string, EvaluationMetrics>;
  run_context?: RunContext;
};

type CodeProvenance = {
  revision?: string | null;
  dirty?: boolean | null;
  source?: string;
};

type RunContext = {
  dataset_id?: string;
  runtime?: Record<string, unknown>;
  configuration?: Record<string, unknown>;
  documents?: number;
  chunks?: number;
  code_provenance?: CodeProvenance;
  translation?: TranslationStatus;
  manual_language_review?: ManualLanguageReview;
};

type QualityRun = Omit<EvaluationResult, "cases"> & {
  dataset_id?: string;
  metrics?: EvaluationMetrics;
  run_context?: RunContext;
  cases?: EvaluationCaseDetail[];
};

type CaseFilter = "all" | "passed" | "failed" | "en" | "tr";

function formatElapsed(seconds: number) {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
  const remainder = (seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${remainder}`;
}

function caseLocale(item: EvaluationCaseDetail): Language {
  const locale = String(item.locale || "").toLowerCase();
  if (locale === "tr" || locale.startsWith("tr-")) return "tr";
  return String(item.id || "").toUpperCase().includes("-TR") ? "tr" : "en";
}

function nestedLabel(value: unknown, fallback = "—") {
  if (typeof value === "string" || typeof value === "number") return String(value);
  if (value && typeof value === "object") {
    const item = value as Record<string, unknown>;
    return String(item.alias || item.variant_id || item.model || item.status || fallback);
  }
  return fallback;
}

function infrastructureLabel(value: unknown, fallback = "—") {
  const raw = nestedLabel(value, fallback);
  const normalized = raw.replace(/[\s_-]/g, "").toLowerCase();
  if (normalized === "foundrylocal") return "Foundry Local";
  if (normalized === "marianopus") return "Marian OPUS-MT";
  if (normalized === "webgpuexecutionprovider" || normalized === "webgpu") return "WebGPU";
  return raw;
}

function shortRevision(revision: string) {
  return revision.length > 10 ? revision.slice(0, 10) : revision;
}

function shortDigest(digest: string) {
  return digest.length > 12 ? `${digest.slice(0, 12)}…` : digest;
}

type TranslationModelView = {
  id: string;
  revision: string;
  cached: boolean;
  loaded: boolean;
};

function normalizeTranslationModel(value: TranslationModelStatus | string | undefined): TranslationModelView {
  if (typeof value === "string") return { id: value, revision: "", cached: false, loaded: false };
  return {
    id: String(value?.id || "").trim(),
    revision: String(value?.revision || "").trim(),
    cached: value?.cached === true,
    loaded: value?.loaded === true,
  };
}

function translationStatusVerified(value: TranslationStatus | undefined, requireLoadedSnapshot = false) {
  if (value?.runtime !== "marian_opus" || value.ready !== true || value.offline_only !== true || !String(value.glossary?.id || "").trim()) return false;
  if (!Number.isFinite(value.round_trip_threshold) || Number(value.round_trip_threshold) <= 0 || Number(value.round_trip_threshold) > 1) return false;
  const trToEn = normalizeTranslationModel(value.models?.tr_to_en);
  const enToTr = normalizeTranslationModel(value.models?.en_to_tr);
  const trToEnAvailable = requireLoadedSnapshot ? trToEn.cached && trToEn.loaded : trToEn.cached || trToEn.loaded;
  const enToTrAvailable = requireLoadedSnapshot ? enToTr.cached && enToTr.loaded : enToTr.cached || enToTr.loaded;
  return Boolean(trToEn.id && trToEn.revision && trToEnAvailable && enToTr.id && enToTr.revision && enToTrAvailable);
}

function manualLanguageReviewApproved(review: ManualLanguageReview | undefined) {
  const reviewedAt = String(review?.reviewed_at || "").trim();
  const reviewedAtValid = /^\d{4}-\d{2}-\d{2}T/.test(reviewedAt) && !Number.isNaN(Date.parse(reviewedAt));
  return review?.verified === true
    && String(review?.status || "").toLowerCase() === "approved"
    && String(review?.locale || "").toLowerCase() === "tr"
    && Boolean(String(review?.reviewer || "").trim())
    && Boolean(String(review?.reviewer_type || "").trim())
    && reviewedAtValid
    && review?.cases_reviewed === 12
    && /^[a-f0-9]{64}$/i.test(String(review?.answers_sha256 || ""))
    && Array.isArray(review?.criteria)
    && review.criteria.length >= 5
    && review.criteria.every((criterion) => Boolean(String(criterion).trim()));
}

function reviewerTypeLabel(value: string | undefined, copy: typeof copyByLanguage.en) {
  if (String(value || "").trim().toLowerCase() === "case-by-case release audit") return copy.quality.manualCaseByCaseReviewerType;
  return value || copy.notRecorded;
}

function trapFocus(event: KeyboardEvent, container: HTMLElement | null) {
  if (event.key !== "Tab" || !container) return;
  const focusable = Array.from(container.querySelectorAll<HTMLElement>(
    'button:not([disabled]), [href], input:not([disabled]):not([type="radio"]), input[type="radio"]:checked:not([disabled]), select:not([disabled]), textarea:not([disabled]), details > summary, [tabindex]:not([tabindex="-1"])',
  )).filter((element) => element.getClientRects().length > 0 && !element.hasAttribute("hidden") && element.getAttribute("aria-hidden") !== "true");
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

function SourceList({ values, empty }: { values?: string[]; empty: string }) {
  if (!values?.length) return <span className="case-empty-value">{empty}</span>;
  return <ul className="case-source-list">{values.map((value) => <li key={value}>{value}</li>)}</ul>;
}

function MetricCard({ label, value, description, target, met, informational = false }: {
  label: string;
  value: string;
  description: string;
  target: string;
  met?: boolean;
  informational?: boolean;
}) {
  return (
    <article className={`metric-card ${met === false ? "miss" : ""}`}>
      <div className="metric-label-row"><span>{label}</span><span className={`target-badge ${informational ? "info" : met ? "met" : "miss"}`}>{target}</span></div>
      <strong>{value}</strong>
      <small>{description}</small>
    </article>
  );
}

function TranslationCheckpointCard({ direction, model, labels }: {
  direction: string;
  model: TranslationModelView;
  labels: {
    checkpoint: string;
    notReported: string;
    loaded: string;
    cached: string;
    missing: string;
  };
}) {
  const ready = model.loaded || model.cached;
  const state = model.loaded ? labels.loaded : model.cached ? labels.cached : labels.missing;
  return (
    <article className="translation-checkpoint">
      <span className="translation-direction">{direction}</span>
      <strong title={model.id || undefined}>{model.id || labels.notReported}</strong>
      <div className="checkpoint-meta"><span>{labels.checkpoint}</span><code title={model.revision || undefined}>{model.revision ? shortRevision(model.revision) : labels.notReported}</code></div>
      <span className={`checkpoint-state ${ready ? "ready" : "missing"}`} aria-label={`${direction}: ${state}`}>{ready ? <CircleCheck size={14} aria-hidden="true" /> : <CircleAlert size={14} aria-hidden="true" />}{state}</span>
    </article>
  );
}

function Brand({ copy }: { copy: typeof copyByLanguage.en }) {
  return (
    <button className="brand" onClick={() => window.scrollTo({ top: 0, behavior: preferredScrollBehavior() })} aria-label={copy.homeAria}>
      <span className="brand-mark" aria-hidden="true"><span /></span>
      <span className="brand-copy"><strong>Tracefold</strong><small>{copy.brandTagline}</small></span>
    </button>
  );
}

function PreferenceGroup<T extends string>({ label, hint, name, value, options, onChange }: {
  label: string;
  hint: string;
  name: string;
  value: T;
  options: Array<{ value: T; label: string; icon?: React.ReactNode }>;
  onChange: (value: T) => void;
}) {
  return (
    <fieldset className="preference-group">
      <legend>{label}</legend>
      <p>{hint}</p>
      <div className="preference-options">
        {options.map((option) => (
          <label key={option.value}>
            <input type="radio" name={name} value={option.value} checked={value === option.value} onChange={() => onChange(option.value)} />
            <span>{option.icon}{option.label}{value === option.value && <Check size={13} aria-hidden="true" />}</span>
          </label>
        ))}
      </div>
    </fieldset>
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
  const [preferences, setPreferences] = useState<Preferences>(loadPreferences);
  const [preferencesOpen, setPreferencesOpen] = useState(false);
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
  const [evaluationStartedAt, setEvaluationStartedAt] = useState<number | null>(null);
  const [evaluationElapsed, setEvaluationElapsed] = useState(0);
  const [caseFilter, setCaseFilter] = useState<CaseFilter>("all");
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const askInput = useRef<HTMLTextAreaElement>(null);
  const transcriptEnd = useRef<HTMLDivElement>(null);
  const evidenceCloseButton = useRef<HTMLButtonElement>(null);
  const evidenceTrigger = useRef<HTMLElement | null>(null);
  const preferencesButton = useRef<HTMLButtonElement>(null);
  const mobileMenuButton = useRef<HTMLButtonElement>(null);
  const preferencesClose = useRef<HTMLButtonElement>(null);
  const preferencesDialog = useRef<HTMLElement>(null);
  const evidenceDialog = useRef<HTMLElement>(null);
  const mainContent = useRef<HTMLElement>(null);
  const askFocusTimer = useRef<number | null>(null);
  const copy = copyByLanguage[preferences.language];
  const navItems: Array<{ id: View; label: string; icon: typeof Search }> = [
    { id: "ask", label: copy.navigation.ask, icon: MessageSquareText },
    { id: "library", label: copy.navigation.library, icon: Library },
    { id: "setup", label: copy.navigation.setup, icon: Server },
    { id: "quality", label: copy.navigation.quality, icon: Gauge },
  ];
  const qualityRun = evaluation as QualityRun | null;
  const qualityMetrics: EvaluationMetrics = qualityRun?.metrics || {
    total_cases: qualityRun?.total_cases,
    passed_cases: qualityRun?.passed_cases,
    hit_at_k: qualityRun?.hit_at_5,
    source_recall_at_k: qualityRun?.source_recall_at_5,
    citation_accuracy: qualityRun?.citation_accuracy,
    required_fact_recall: qualityRun?.required_fact_recall,
    refusal_accuracy: qualityRun?.refusal_accuracy,
    retrieval_p95_ms: qualityRun?.retrieval_p95_ms,
  };
  const runContext = qualityMetrics.run_context || qualityRun?.run_context;
  const evaluationCases = qualityRun?.cases || [];
  const filteredCases = evaluationCases.filter((item) => {
    if (caseFilter === "passed") return item.passed === true;
    if (caseFilter === "failed") return item.passed !== true;
    if (caseFilter === "en" || caseFilter === "tr") return caseLocale(item) === caseFilter;
    return true;
  });
  const totalCases = qualityMetrics.total_cases ?? qualityRun?.total_cases;
  const passedCases = qualityMetrics.passed_cases ?? qualityRun?.passed_cases;
  const passRate = totalCases && passedCases !== undefined ? passedCases / totalCases : undefined;
  const hitAtFive = qualityMetrics.hit_at_k ?? qualityRun?.hit_at_5;
  const sourceRecall = qualityMetrics.source_recall_at_k ?? qualityRun?.source_recall_at_5;
  const citationAccuracy = qualityMetrics.citation_accuracy ?? qualityRun?.citation_accuracy;
  const refusalAccuracy = qualityMetrics.refusal_accuracy ?? qualityRun?.refusal_accuracy;
  const retrievalP95 = qualityMetrics.retrieval_p95_ms ?? qualityRun?.retrieval_p95_ms;
  const factRecall = qualityMetrics.required_fact_recall ?? qualityRun?.required_fact_recall;
  const runRuntime = runContext?.runtime || {};
  const runConfiguration = runContext?.configuration || {};
  const runTranslation = runContext?.translation;
  const runTranslationTrToEn = normalizeTranslationModel(runTranslation?.models?.tr_to_en);
  const runTranslationEnToTr = normalizeTranslationModel(runTranslation?.models?.en_to_tr);
  const translationProvenanceVerified = translationStatusVerified(runTranslation, true);
  const configuredTranslationThreshold = typeof runConfiguration.translation_round_trip_threshold === "number"
    ? runConfiguration.translation_round_trip_threshold
    : undefined;
  const translationThreshold = runTranslation?.round_trip_threshold ?? configuredTranslationThreshold;
  const rawQualityPolicy = runConfiguration.quality_policy;
  const qualityPolicy = rawQualityPolicy && typeof rawQualityPolicy === "object" ? rawQualityPolicy as Record<string, unknown> : {};
  const manualReviewLocales = Array.isArray(qualityPolicy.manual_language_review_required_for_release)
    ? qualityPolicy.manual_language_review_required_for_release.map(String)
    : [];
  const reportedLocales = [
    ...(Array.isArray(qualityMetrics.locales) ? qualityMetrics.locales : []),
    ...Object.keys(qualityMetrics.locale_breakdown || {}),
    ...evaluationCases.map(caseLocale),
  ];
  const evaluatesTurkish = reportedLocales.some((locale) => String(locale).toLowerCase().startsWith("tr"));
  const manualReviewRequired = evaluatesTurkish && manualReviewLocales.some((locale) => locale.toLowerCase() === "tr");
  const manualReview = runContext?.manual_language_review;
  const manualReviewIsApproved = manualLanguageReviewApproved(manualReview);
  const manualReviewGateMet = !manualReviewRequired || manualReviewIsApproved;
  const runtimeTranslation = runtime?.translation;
  const runtimeTranslationTrToEn = normalizeTranslationModel(runtimeTranslation?.models?.tr_to_en);
  const runtimeTranslationEnToTr = normalizeTranslationModel(runtimeTranslation?.models?.en_to_tr);
  const runtimeTranslationVerified = translationStatusVerified(runtimeTranslation);
  const codeProvenance = runContext?.code_provenance;
  const codeRevision = typeof codeProvenance?.revision === "string" ? codeProvenance.revision.trim() : "";
  const provenanceVerified = Boolean(codeRevision) && codeProvenance?.dirty === false;
  const automatedReleaseReady = passRate === 1 && provenanceVerified && translationProvenanceVerified;
  const releaseAccepted = automatedReleaseReady && manualReviewGateMet;
  const releaseStatus = !provenanceVerified
    ? copy.quality.releaseUnverified
    : !translationProvenanceVerified
      ? copy.quality.releaseTranslationUnverified
      : passRate !== 1
        ? copy.quality.releaseNeedsReview
        : !manualReviewGateMet
          ? copy.quality.releaseManualLanguageReviewRequired
          : copy.quality.releaseAccepted;
  const localeNames = Array.from(new Set(reportedLocales.map((locale) => String(locale).toLowerCase().startsWith("tr") ? "tr" : "en"))) as Language[];
  const target100 = formatPercent(1, copy.locale);
  const target90 = formatPercent(.9, copy.locale);

  useEffect(() => {
    const root = document.documentElement;
    const darkMode = window.matchMedia("(prefers-color-scheme: dark)");
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const apply = () => {
      const theme = preferences.theme === "system" ? (darkMode.matches ? "dark" : "light") : preferences.theme;
      const motion = preferences.motion === "system" ? (reducedMotion.matches ? "reduced" : "full") : preferences.motion;
      root.lang = preferences.language;
      root.dataset.theme = theme;
      root.dataset.motion = motion;
      root.style.colorScheme = theme;
      document.title = `Tracefold — ${copy.brandTagline}`;
      document.querySelector<HTMLMetaElement>('meta[name="theme-color"]')?.setAttribute("content", theme === "dark" ? "#0b1516" : "#0d1b1d");
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
    };
    apply();
    darkMode.addEventListener("change", apply);
    reducedMotion.addEventListener("change", apply);
    return () => {
      darkMode.removeEventListener("change", apply);
      reducedMotion.removeEventListener("change", apply);
    };
  }, [copy.brandTagline, preferences]);

  useEffect(() => {
    if (!preferencesOpen) return;
    window.requestAnimationFrame(() => preferencesClose.current?.focus());
  }, [preferencesOpen]);

  useEffect(() => {
    if (!evaluationRunning || evaluationStartedAt === null) return;
    const updateElapsed = () => setEvaluationElapsed(Math.max(0, Math.floor((Date.now() - evaluationStartedAt) / 1000)));
    updateElapsed();
    const timer = window.setInterval(updateElapsed, 500);
    return () => window.clearInterval(timer);
  }, [evaluationRunning, evaluationStartedAt]);

  useEffect(() => {
    if (!selectedCitation && !preferencesOpen) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = previousOverflow; };
  }, [preferencesOpen, selectedCitation]);

  useEffect(() => () => {
    if (askFocusTimer.current !== null) window.clearTimeout(askFocusTimer.current);
  }, []);

  const openCitation = useCallback((citation: Citation) => {
    if (askFocusTimer.current !== null) window.clearTimeout(askFocusTimer.current);
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
    catch (error) { setRuntime(null); setRuntimeError(localizedError(error, copy.setup.checkFailed, preferences.language)); }
    finally { setRuntimeLoading(false); }
  }, [copy.setup.checkFailed, preferences.language]);

  const refreshDocuments = useCallback(async () => {
    setDocumentsLoading(true);
    setDocumentError("");
    try { setDocuments((await getDocuments()).documents); }
    catch (error) { setDocumentError(localizedError(error, copy.library.loadFailed, preferences.language)); }
    finally { setDocumentsLoading(false); }
  }, [copy.library.loadFailed, preferences.language]);

  const refreshEvaluation = useCallback(async () => {
    setEvaluationLoading(true);
    setEvaluationError("");
    try { setEvaluation(await getLatestEvaluation()); }
    catch (error) { setEvaluationError(localizedError(error, copy.quality.loadFailed, preferences.language)); }
    finally { setEvaluationLoading(false); }
  }, [copy.quality.loadFailed, preferences.language]);

  useEffect(() => { void Promise.all([refreshRuntime(), refreshDocuments(), refreshEvaluation()]); }, [refreshRuntime, refreshDocuments, refreshEvaluation]);
  useEffect(() => { transcriptEnd.current?.scrollIntoView({ behavior: preferredScrollBehavior(), block: "nearest" }); }, [messages]);
  useEffect(() => {
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (selectedCitation) closeCitation();
        if (preferencesOpen) {
          setPreferencesOpen(false);
          window.requestAnimationFrame(() => preferencesButton.current?.focus());
        }
        if (mobileNav) {
          setMobileNav(false);
          window.requestAnimationFrame(() => mobileMenuButton.current?.focus());
        }
      }
      if (selectedCitation) trapFocus(event, evidenceDialog.current);
      else if (preferencesOpen) trapFocus(event, preferencesDialog.current);
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [closeCitation, mobileNav, preferencesOpen, selectedCitation]);

  const navigate = (next: View) => {
    setView(next);
    setMobileNav(false);
    window.scrollTo({ top: 0, behavior: preferredScrollBehavior() });
    window.requestAnimationFrame(() => mainContent.current?.focus({ preventScroll: true }));
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
    throw new Error(copy.library.longIndexing);
  };

  const handleFiles = async (files: File[]) => {
    const supported = files.filter((file) => /\.(pdf|docx|txt|md|markdown)$/i.test(file.name));
    if (!supported.length) {
      setDocumentError(copy.library.chooseFiles);
      return;
    }
    setDocumentError(files.length !== supported.length ? copy.library.unsupportedSkipped : "");
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
        setUploads((current) => current.map((item) => item.id === task.id ? { ...item, status: "error", progress: 100, message: localizedError(error, copy.library.uploadFailed, preferences.language) } : item));
      }
    }
    void refreshRuntime();
  };

  const removeDocument = async (document: DocumentItem) => {
    if (!window.confirm(copy.library.removeConfirm(document.filename))) return;
    try { await deleteDocument(document.id); await refreshDocuments(); }
    catch (error) { setDocumentError(localizedError(error, copy.library.removeFailed, preferences.language)); }
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
    let refused = false;
    try {
      await streamChat(clean, topK, preferences.language, {
        onRetrieval: () => undefined,
        onToken: (token) => {
          content += token;
          setMessages((current) => current.map((message) => message.id === answerId ? { ...message, content } : message));
        },
        onCitations: (items) => {
          citations = items;
          setMessages((current) => current.map((message) => message.id === answerId ? { ...message, citations: items } : message));
        },
        onDone: (payload) => {
          refused = payload.refused === true;
        },
      });
      setMessages((current) => current.map((message) => message.id === answerId ? { ...message, content: content || copy.ask.noEvidence, citations, status: refused || !content ? "refusal" : "done" } : message));
    } catch (error) {
      const text = localizedError(error, copy.errors.answer, preferences.language);
      setAskError(text);
      setMessages((current) => current.map((message) => message.id === answerId ? { ...message, content: text, status: "error" } : message));
    } finally {
      setAsking(false);
      askFocusTimer.current = window.setTimeout(() => {
        if (!document.querySelector('[role="dialog"][aria-modal="true"]')) askInput.current?.focus();
        askFocusTimer.current = null;
      }, 50);
    }
  };

  const startEvaluation = async () => {
    setEvaluationRunning(true);
    setEvaluationStartedAt(Date.now());
    setEvaluationElapsed(0);
    setEvaluationError("");
    try { setEvaluation(await runEvaluation()); }
    catch (error) { setEvaluationError(localizedError(error, copy.quality.evaluationFailed, preferences.language)); }
    finally {
      setEvaluationRunning(false);
      setEvaluationStartedAt(null);
    }
  };

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content" onClick={() => window.requestAnimationFrame(() => mainContent.current?.focus())}>{copy.skipToContent}</a>
      <header className="site-header">
        <div className="header-inner">
          <Brand copy={copy} />
          <nav className="desktop-nav" aria-label={copy.navigation.label}>
            {navItems.map((item) => <button key={item.id} className={view === item.id ? "active" : ""} aria-current={view === item.id ? "page" : undefined} onClick={() => navigate(item.id)}>{item.label}</button>)}
          </nav>
          <div className="header-meta">
            <button className={`runtime-chip ${runtimeAvailable(runtime) ? "ready" : ""}`} aria-current={view === "setup" ? "page" : undefined} onClick={() => navigate("setup")}>
              <span className="status-dot" aria-hidden="true" />
              {runtimeLoading ? copy.runtimeChip.checking : runtimeReady(runtime) ? copy.runtimeChip.ready : runtimeAvailable(runtime) ? copy.runtimeChip.available : copy.runtimeChip.needed}
              <span className="sr-only" role="status" aria-live="polite">{statusLabel(runtime, runtimeLoading, copy)}</span>
            </button>
            <div className="preferences-root">
              <button ref={preferencesButton} className="preferences-trigger" onClick={() => { setPreferencesOpen((open) => !open); setMobileNav(false); }} aria-expanded={preferencesOpen} aria-controls="tracefold-preferences" aria-label={copy.preferences.open}>
                <Settings2 size={17} aria-hidden="true" /><span>{preferences.language.toUpperCase()}</span>
              </button>
            </div>
            <button ref={mobileMenuButton} className="mobile-menu" onClick={() => { setMobileNav(!mobileNav); setPreferencesOpen(false); }} aria-expanded={mobileNav} aria-controls="tracefold-mobile-navigation" aria-label={mobileNav ? copy.navigation.close : copy.navigation.open}><Menu size={21} /></button>
          </div>
        </div>
        {mobileNav && <nav id="tracefold-mobile-navigation" className="mobile-nav" aria-label={copy.navigation.mobileLabel}>{navItems.map((item) => { const Icon = item.icon; return <button key={item.id} className={view === item.id ? "active" : ""} aria-current={view === item.id ? "page" : undefined} onClick={() => navigate(item.id)}><Icon size={18} aria-hidden="true" />{item.label}</button>; })}</nav>}
      </header>

      <main ref={mainContent} id="main-content" tabIndex={-1}>
        {view === "ask" && (
          <section className="ask-page">
            <div className="hero-copy">
              <div className="eyebrow"><ShieldCheck size={14} />{copy.ask.eyebrow}</div>
              <h1>{copy.ask.titleOne}<br /><em>{copy.ask.titleTwo}</em></h1>
              <p>{copy.ask.intro}</p>
            </div>

            <div className="ask-workspace">
              {messages.length === 0 ? (
                <div className="ask-empty">
                  <div className="trace-orbit" aria-hidden="true"><span /><i /><b /></div>
                  <h2>{copy.ask.emptyTitle}</h2>
                  <p>{documents.length ? copy.ask.documentsReady(documents.length) : copy.ask.noDocuments}</p>
                  <div className="suggestions">
                    {copy.ask.starterQuestions.map((item) => <button key={item} onClick={() => void ask(item)} disabled={!documents.length || asking}>{item}<ArrowRight size={15} /></button>)}
                  </div>
                </div>
              ) : (
                <div className="transcript" role="log" aria-live="polite" aria-relevant="additions text" aria-busy={asking}>
                  {messages.map((message) => (
                    <article key={message.id} className={`message ${message.role} ${message.status || ""}`} aria-busy={message.status === "streaming"}>
                      <div className="message-label">{message.role === "user" ? copy.ask.you : "Tracefold"}</div>
                      <div className="message-body">
                        {message.status === "streaming" && !message.content ? <span className="thinking"><i /><i /><i /><span className="sr-only">{copy.ask.generating}</span></span> : <p>{message.content}</p>}
                        {message.status === "refusal" && <div className="refusal-note"><CircleAlert size={16} />{copy.ask.refusal}</div>}
                        {!!message.citations?.length && <div className="citation-row" aria-label={copy.ask.sources}>{message.citations.map((citation) => <button key={citation.id} onClick={() => openCitation(citation)}><span>{citation.index}</span>{citation.source}</button>)}</div>}
                      </div>
                    </article>
                  ))}
                  <div ref={transcriptEnd} />
                </div>
              )}

              <div className="composer-wrap">
                {askError && <div className="inline-error" role="alert"><CircleAlert size={15} aria-hidden="true" />{askError}</div>}
                <div className="composer">
                  <textarea ref={askInput} value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void ask(); } }} placeholder={documents.length ? copy.ask.placeholder : copy.ask.emptyPlaceholder} disabled={asking || !documents.length} rows={1} aria-label={copy.ask.inputAria} aria-describedby="composer-privacy-status" />
                  <button className="send-button" onClick={() => void ask()} disabled={!question.trim() || asking || !documents.length} aria-label={copy.ask.sendAria}>{asking ? <LoaderCircle className="spin" size={19} /> : <Send size={18} />}</button>
                </div>
                <div className="composer-meta">
                  <button onClick={() => navigate("library")}><BookOpen size={14} />{documents.length ? copy.ask.documents(documents.length) : copy.ask.addDocuments}</button>
                  <label>{copy.ask.evidenceDepth} <select value={topK} onChange={(event) => setTopK(Number(event.target.value))} aria-label={copy.ask.evidenceDepthAria}>{[3, 4, 5, 6, 8].map((value) => <option key={value} value={value}>{copy.ask.passages(value)}</option>)}</select></label>
                  <span id="composer-privacy-status"><HardDrive size={13} aria-hidden="true" />{copy.ask.staysLocal}</span>
                </div>
              </div>
            </div>

            <div className="principles-strip">
              <div><Search size={18} /><span><strong>{copy.ask.retrieve}</strong><small>{copy.ask.retrieveCopy}</small></span></div>
              <ChevronRight size={16} className="principle-arrow" />
              <div><Sparkles size={18} /><span><strong>{copy.ask.answer}</strong><small>{copy.ask.answerCopy}</small></span></div>
              <ChevronRight size={16} className="principle-arrow" />
              <div><FileSearch size={18} /><span><strong>{copy.ask.verify}</strong><small>{copy.ask.verifyCopy}</small></span></div>
            </div>
          </section>
        )}

        {view === "library" && (
          <section className="page-section">
            <div className="page-heading"><div><span className="section-kicker">{copy.library.kicker}</span><h1>{copy.library.title}</h1><p>{copy.library.intro}</p></div><button className="primary-button" onClick={() => fileInput.current?.click()}><Plus size={17} />{copy.library.addDocuments}</button></div>
            <input ref={fileInput} type="file" multiple accept=".pdf,.docx,.txt,.md,.markdown" hidden onChange={(event) => { void handleFiles(Array.from(event.target.files || [])); event.target.value = ""; }} />
            <div className={`dropzone ${dragging ? "dragging" : ""}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); void handleFiles(Array.from(event.dataTransfer.files)); }} onClick={() => fileInput.current?.click()} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); fileInput.current?.click(); } }} role="button" tabIndex={0}>
              <span className="drop-icon"><UploadCloud size={24} /></span><div><strong>{copy.library.dropTitle}</strong><small>{copy.library.dropCopy}</small></div><span className="browse-label">{copy.library.browse}</span>
            </div>
            {documentError && <div className="banner error" role="alert"><CircleAlert size={17} aria-hidden="true" /><span>{documentError}</span><button className="icon-button" onClick={() => setDocumentError("")} aria-label={copy.library.dismiss}><X size={16} /></button></div>}
            {!!uploads.length && <section className="panel upload-panel" aria-live="polite"><div className="panel-heading"><div><span className="section-kicker">{copy.library.queueKicker}</span><h2>{copy.library.processing}</h2></div><button className="text-button" onClick={() => setUploads((current) => current.filter((item) => !["ready", "error"].includes(item.status)))}>{copy.library.clearCompleted}</button></div><div className="upload-list">{uploads.map((task) => <div className="upload-item" key={task.id}><span className={`file-icon ${task.status}`}><File size={18} /></span><div><strong>{task.filename}</strong><small>{preferences.language === "en" && task.message ? task.message : copy.library.taskCopy[task.status]}</small><span className="progress" role="progressbar" aria-label={`${task.filename}: ${copy.library.taskStatus[task.status]}`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(task.progress)}><i style={{ width: `${task.progress}%` }} /></span></div><span className={`task-status ${task.status}`}>{task.status === "ready" ? <Check size={15} /> : task.status === "error" ? <CircleAlert size={15} /> : <LoaderCircle className={task.status !== "queued" ? "spin" : ""} size={15} />}{copy.library.taskStatus[task.status]}</span></div>)}</div></section>}
            <section className="panel library-panel">
              <div className="panel-heading"><div><span className="section-kicker">{copy.library.searchableKicker}</span><h2>{copy.library.indexedTitle}</h2></div><span className="count-pill">{copy.library.count(documents.length, documents.reduce((total, document) => total + (document.chunks || 0), 0))}</span></div>
              {documentsLoading ? <div className="skeleton-list"><i /><i /><i /></div> : documentError && !documents.length ? <EmptyState icon={CircleAlert} title={copy.library.unavailable} copy={documentError} action={<button className="secondary-button" onClick={() => void refreshDocuments()}>{copy.library.tryAgain}</button>} /> : !documents.length ? <EmptyState icon={Library} title={copy.library.emptyTitle} copy={copy.library.emptyCopy} action={<button className="secondary-button" onClick={() => fileInput.current?.click()}><Plus size={16} />{copy.library.addFirst}</button>} /> : <div className="document-list">{documents.map((document) => <article key={document.id}><span className="document-icon"><File size={20} /></span><div className="document-name"><strong>{document.filename}</strong><span>{document.chunks !== undefined ? copy.library.passages(document.chunks) : copy.library.indexed}{document.size ? ` · ${formatBytes(document.size)}` : ""}</span></div><span className={`document-state ${document.status === "error" ? "error" : ""}`}><i />{document.status === "error" ? copy.library.issue : copy.library.searchable}</span><time>{formatDate(document.created_at, copy.locale, copy.notRecorded)}</time><button className="icon-button danger" onClick={() => void removeDocument(document)} aria-label={copy.library.remove(document.filename)}><Trash2 size={17} /></button></article>)}</div>}
            </section>
          </section>
        )}

        {view === "setup" && (
          <section className="page-section narrow">
            <div className="page-heading"><div><span className="section-kicker">{copy.setup.kicker}</span><h1>{copy.setup.title}</h1><p>{copy.setup.intro}</p></div><button className="secondary-button" onClick={() => void refreshRuntime()} disabled={runtimeLoading}>{runtimeLoading ? <LoaderCircle className="spin" size={17} /> : <Search size={17} />}{copy.setup.checkAgain}</button></div>
            <section className={`runtime-hero ${runtimeAvailable(runtime) ? "ready" : "attention"}`} role="status" aria-live="polite" aria-busy={runtimeLoading}>
              <span className="runtime-symbol">{runtimeAvailable(runtime) ? <CircleCheck size={29} /> : <CircleAlert size={29} />}</span>
              <div><span className="section-kicker">{copy.setup.currentStatus}</span><h2>{statusLabel(runtime, runtimeLoading, copy)}</h2><p>{runtimeReady(runtime) ? copy.setup.readyCopy : runtimeAvailable(runtime) ? copy.setup.availableCopy : runtimeError || (preferences.language === "en" ? runtime?.message : "") || copy.setup.missingCopy}</p></div>
              <span className="local-badge"><ShieldCheck size={15} />{copy.setup.onDevice}</span>
            </section>
            <div className="setup-grid">
              <article className="panel config-card"><div className="config-icon"><Server size={21} /></div><span>{copy.setup.runtime}</span><strong>{infrastructureLabel(runtime?.runtime || statusLabel(runtime, runtimeLoading, copy))}</strong><small>{[runtime?.device || copy.setup.localMachine, runtime?.execution_provider].filter(Boolean).map((value) => infrastructureLabel(value)).join(" · ")}</small></article>
              <article className="panel config-card"><div className="config-icon"><Sparkles size={21} /></div><span>{copy.setup.chatModel}</span><strong>{runtime?.chat_model || runtime?.model || copy.setup.notReported}</strong><small>{runtime?.chat_variant || copy.setup.groundedGeneration}</small></article>
              <article className="panel config-card"><div className="config-icon"><Database size={21} /></div><span>{copy.setup.embeddingModel}</span><strong>{runtime?.embedding_model || copy.setup.notReported}</strong><small>{runtime?.embedding_variant || copy.setup.semanticSearch}</small></article>
            </div>
            <section className="panel translation-panel" aria-labelledby="setup-translation-title">
              <div className="panel-heading translation-panel-heading">
                <div><span className="section-kicker">{copy.setup.translationKicker}</span><h2 id="setup-translation-title">{copy.setup.translationTitle}</h2><p>{copy.setup.translationIntro}</p></div>
                <span className={`translation-verification ${runtimeTranslationVerified ? "verified" : "unverified"}`}>{runtimeTranslationVerified ? <CircleCheck size={15} aria-hidden="true" /> : <CircleAlert size={15} aria-hidden="true" />}{runtimeTranslationVerified ? copy.setup.translationVerified : copy.setup.translationUnverified}</span>
              </div>
              <div className="translation-checkpoint-grid">
                <TranslationCheckpointCard direction={copy.setup.trToEn} model={runtimeTranslationTrToEn} labels={{ checkpoint: copy.setup.checkpoint, notReported: copy.setup.notReported, loaded: copy.setup.checkpointLoaded, cached: copy.setup.checkpointCached, missing: copy.setup.checkpointMissing }} />
                <TranslationCheckpointCard direction={copy.setup.enToTr} model={runtimeTranslationEnToTr} labels={{ checkpoint: copy.setup.checkpoint, notReported: copy.setup.notReported, loaded: copy.setup.checkpointLoaded, cached: copy.setup.checkpointCached, missing: copy.setup.checkpointMissing }} />
                <article className="translation-checkpoint glossary-checkpoint">
                  <span className="translation-direction">{copy.setup.glossaryVersion}</span>
                  <strong title={runtimeTranslation?.glossary?.id}>{runtimeTranslation?.glossary?.id || copy.setup.notReported}</strong>
                  <div className="checkpoint-meta"><span>{copy.setup.glossaryEntries}</span><code>{runtimeTranslation?.glossary?.entries ?? copy.setup.notReported}</code></div>
                  <span className={`checkpoint-state ${runtimeTranslation?.glossary?.id ? "ready" : "missing"}`}>{runtimeTranslation?.glossary?.id ? <CircleCheck size={14} aria-hidden="true" /> : <CircleAlert size={14} aria-hidden="true" />}{runtimeTranslation?.glossary?.id ? copy.setup.glossaryLoaded : copy.setup.checkpointMissing}</span>
                </article>
              </div>
              <div className="translation-runtime-line"><span>{copy.setup.translationRuntime}</span><strong>{infrastructureLabel(runtimeTranslation?.runtime, copy.setup.notReported)}</strong><small>{runtimeTranslation?.offline_only === true ? copy.setup.offlineOnly : copy.setup.offlineNotVerified}</small></div>
            </section>
            {!runtimeAvailable(runtime) && <section className="panel setup-steps"><div className="panel-heading"><div><span className="section-kicker">{copy.setup.checklist}</span><h2>{copy.setup.beforeFirst}</h2></div></div><ol><li><span>1</span><div><strong>{copy.setup.startFoundry}</strong><p>{copy.setup.startFoundryCopy}</p></div></li><li><span>2</span><div><strong>{copy.setup.makeModels}</strong><p>{copy.setup.makeModelsCopy}</p></div></li><li><span>3</span><div><strong>{copy.setup.runAgain}</strong><p>{copy.setup.runAgainCopy}</p></div></li></ol></section>}
            <section className="privacy-note"><ShieldCheck size={20} /><div><strong>{copy.setup.privateTitle}</strong><p>{copy.setup.privateCopy}</p></div></section>
          </section>
        )}

        {view === "quality" && (
          <section className="page-section" aria-labelledby="quality-title">
            <div className="page-heading">
              <div><span className="section-kicker">{copy.quality.kicker}</span><h1 id="quality-title">{copy.quality.title}</h1><p>{copy.quality.intro}</p></div>
              <button className="primary-button" onClick={() => void startEvaluation()} disabled={evaluationRunning || !runtimeAvailable(runtime)} aria-describedby={evaluationRunning ? "evaluation-live-copy" : undefined}>{evaluationRunning ? <LoaderCircle className="spin" size={17} aria-hidden="true" /> : <Play size={16} fill="currentColor" aria-hidden="true" />}{evaluationRunning ? copy.quality.running : copy.quality.run}</button>
            </div>

            {evaluationError && <div className="banner error" role="alert"><CircleAlert size={17} aria-hidden="true" />{evaluationError}</div>}

            {evaluationRunning && (
              <section className="evaluation-live" role="status" aria-live="polite" aria-atomic="true">
                <span className="live-symbol" aria-hidden="true"><LoaderCircle className="spin" size={23} /></span>
                <div><span className="section-kicker">{copy.quality.liveKicker}</span><h2>{copy.quality.liveTitle}</h2><p id="evaluation-live-copy">{copy.quality.liveCopy(formatElapsed(evaluationElapsed))}</p></div>
                <span className="live-time"><small>{copy.quality.liveStatus}</small><strong>{formatElapsed(evaluationElapsed)}</strong></span>
              </section>
            )}

            {evaluationLoading ? (
              <div className="quality-grid" aria-busy="true" aria-label={copy.quality.running}><div className="metric-skeleton" /><div className="metric-skeleton" /><div className="metric-skeleton" /><div className="metric-skeleton" /></div>
            ) : !evaluation ? (
              <section className="panel"><EmptyState icon={Gauge} title={copy.quality.noRun} copy={copy.quality.noRunCopy} action={<button className="secondary-button" onClick={() => void startEvaluation()} disabled={!runtimeAvailable(runtime)}><Play size={15} aria-hidden="true" />{copy.quality.firstRun}</button>} /></section>
            ) : <>
              {evaluationRunning && <p className="previous-run-label">{copy.quality.previousResults}</p>}
              <div className={`quality-summary ${releaseAccepted ? "accepted" : "unverified"}`} role="status" aria-label={releaseStatus}>
                <div><span className={`quality-seal ${releaseAccepted ? "pass" : "blocked"}`}>{releaseAccepted ? <Check size={24} aria-hidden="true" /> : <CircleAlert size={24} aria-hidden="true" />}</span><div><span className="section-kicker">{copy.quality.latestRun(formatDate(qualityRun?.created_at, copy.locale, copy.notRecorded))}</span><h2>{copy.quality.casesPassed(passedCases ?? "—", totalCases ?? "—")}</h2></div></div>
                <span className={`run-status ${releaseAccepted ? "accepted" : "blocked"}`}>{releaseStatus}</span>
              </div>

              <div className="quality-grid quality-grid-expanded">
                <MetricCard label={copy.quality.passRate} value={formatPercent(passRate, copy.locale)} description={copy.quality.passRateCopy} target={`${copy.quality.target}: ${copy.quality.targetExactly(target100)} · ${passRate === 1 ? copy.quality.targetMet : copy.quality.targetMissed}`} met={passRate === 1} />
                <MetricCard label={copy.quality.retrieval} value={formatPercent(hitAtFive, copy.locale)} description={copy.quality.retrievalCopy} target={`${copy.quality.target}: ${copy.quality.targetAtLeast(target90)} · ${(hitAtFive ?? -1) >= .9 ? copy.quality.targetMet : copy.quality.targetMissed}`} met={(hitAtFive ?? -1) >= .9} />
                <MetricCard label={copy.quality.coverage} value={formatPercent(sourceRecall, copy.locale)} description={copy.quality.coverageCopy} target={`${copy.quality.target}: ${copy.quality.targetExactly(target100)} · ${sourceRecall === 1 ? copy.quality.targetMet : copy.quality.targetMissed}`} met={sourceRecall === 1} />
                <MetricCard label={copy.quality.citation} value={formatPercent(citationAccuracy, copy.locale)} description={copy.quality.citationCopy} target={`${copy.quality.target}: ${copy.quality.targetExactly(target100)} · ${citationAccuracy === 1 ? copy.quality.targetMet : copy.quality.targetMissed}`} met={citationAccuracy === 1} />
                <MetricCard label={copy.quality.refusal} value={formatPercent(refusalAccuracy, copy.locale)} description={copy.quality.refusalCopy} target={`${copy.quality.target}: ${copy.quality.targetExactly(target100)} · ${refusalAccuracy === 1 ? copy.quality.targetMet : copy.quality.targetMissed}`} met={refusalAccuracy === 1} />
                {qualityMetrics.language_quality_accuracy !== undefined && <MetricCard label={copy.quality.languageQuality} value={formatPercent(qualityMetrics.language_quality_accuracy, copy.locale)} description={copy.quality.languageQualityCopy} target={`${copy.quality.target}: ${copy.quality.targetExactly(target100)} · ${qualityMetrics.language_quality_accuracy === 1 ? copy.quality.targetMet : copy.quality.targetMissed}`} met={qualityMetrics.language_quality_accuracy === 1} />}
              </div>

              <section className="panel latency-panel">
                <div className="panel-heading"><div><span className="section-kicker">{copy.quality.measured}</span><h2>{copy.quality.details}</h2></div></div>
                <div className="evidence-metrics">
                  <div><span>{copy.quality.p95}</span><strong>{retrievalP95 !== undefined ? `${Math.round(retrievalP95)} ms` : "—"}</strong><small className={retrievalP95 !== undefined && retrievalP95 < 500 ? "met" : "miss"}>{copy.quality.targetUnder("500 ms")}</small></div>
                  <div><span>{copy.quality.factRecall}</span><strong>{formatPercent(factRecall, copy.locale)}</strong><small className="info">{copy.quality.informational}</small></div>
                  {qualityMetrics.citation_sentence_coverage !== undefined && <div><span>{copy.quality.sentenceCoverage}</span><strong>{formatPercent(qualityMetrics.citation_sentence_coverage, copy.locale)}</strong><small className={qualityMetrics.citation_sentence_coverage === 1 ? "met" : "miss"}>{copy.quality.targetExactly(target100)}</small></div>}
                  {qualityMetrics.protected_fact_grounding_accuracy !== undefined && <div><span>{copy.quality.protectedFacts}</span><strong>{formatPercent(qualityMetrics.protected_fact_grounding_accuracy, copy.locale)}</strong><small className={qualityMetrics.protected_fact_grounding_accuracy === 1 ? "met" : "miss"}>{copy.quality.targetExactly(target100)}</small></div>}
                  {qualityMetrics.bilingual_pair_accuracy !== undefined && <div><span>{copy.quality.bilingualPairs}</span><strong>{formatPercent(qualityMetrics.bilingual_pair_accuracy, copy.locale)}</strong><small className={qualityMetrics.bilingual_pair_accuracy === 1 ? "met" : "miss"}>{copy.quality.targetExactly(target100)}</small></div>}
                </div>
              </section>

              {!!localeNames.length && (
                <section className="panel locale-panel" aria-labelledby="locale-results-title">
                  <div className="panel-heading"><div><span className="section-kicker">{copy.quality.localeKicker}</span><h2 id="locale-results-title">{copy.quality.localeTitle}</h2></div></div>
                  <div className="locale-results">
                    {localeNames.map((locale) => {
                      const recorded = qualityMetrics.locale_breakdown?.[locale] || {};
                      const localeCases = evaluationCases.filter((item) => caseLocale(item) === locale);
                      const localeTotal = recorded.total_cases ?? localeCases.length;
                      const localePassed = recorded.passed_cases ?? localeCases.filter((item) => item.passed).length;
                      const localeRate = localeTotal ? localePassed / localeTotal : undefined;
                      return <article key={locale}><span className="locale-code">{locale.toUpperCase()}</span><div><strong>{locale === "tr" ? copy.quality.turkishLocale : copy.quality.englishLocale}</strong><small>{copy.quality.casesPassed(localePassed, localeTotal)}</small></div><div className="locale-score"><span>{copy.quality.localeMetric}</span><strong>{formatPercent(localeRate, copy.locale)}</strong></div></article>;
                    })}
                  </div>
                </section>
              )}

              <section className="panel translation-panel quality-translation-panel" aria-labelledby="quality-translation-title">
                <div className="panel-heading translation-panel-heading">
                  <div><span className="section-kicker">{copy.quality.translationKicker}</span><h2 id="quality-translation-title">{copy.quality.translationTitle}</h2><p>{copy.quality.translationIntro}</p></div>
                  <span className={`translation-verification ${translationProvenanceVerified ? "verified" : "unverified"}`}>{translationProvenanceVerified ? <CircleCheck size={15} aria-hidden="true" /> : <CircleAlert size={15} aria-hidden="true" />}{translationProvenanceVerified ? copy.quality.translationVerified : copy.quality.translationUnverified}</span>
                </div>
                {!translationProvenanceVerified && <div className="translation-warning" role="alert"><CircleAlert size={19} aria-hidden="true" /><div><strong>{copy.quality.translationWarningTitle}</strong><p>{copy.quality.translationWarningCopy}</p></div></div>}
                <div className="translation-checkpoint-grid">
                  <TranslationCheckpointCard direction={copy.setup.trToEn} model={runTranslationTrToEn} labels={{ checkpoint: copy.setup.checkpoint, notReported: copy.setup.notReported, loaded: copy.setup.checkpointLoaded, cached: copy.setup.checkpointCached, missing: copy.setup.checkpointMissing }} />
                  <TranslationCheckpointCard direction={copy.setup.enToTr} model={runTranslationEnToTr} labels={{ checkpoint: copy.setup.checkpoint, notReported: copy.setup.notReported, loaded: copy.setup.checkpointLoaded, cached: copy.setup.checkpointCached, missing: copy.setup.checkpointMissing }} />
                  <article className="translation-checkpoint glossary-checkpoint">
                    <span className="translation-direction">{copy.quality.translationGlossary}</span>
                    <strong title={runTranslation?.glossary?.id}>{runTranslation?.glossary?.id || copy.notRecorded}</strong>
                    <div className="checkpoint-meta"><span>{copy.setup.glossaryEntries}</span><code>{runTranslation?.glossary?.entries ?? copy.notRecorded}</code></div>
                    <span className={`checkpoint-state ${runTranslation?.glossary?.id ? "ready" : "missing"}`}>{runTranslation?.glossary?.id ? <CircleCheck size={14} aria-hidden="true" /> : <CircleAlert size={14} aria-hidden="true" />}{runTranslation?.glossary?.id ? copy.setup.glossaryLoaded : copy.setup.checkpointMissing}</span>
                  </article>
                </div>
                <dl className="translation-gates">
                  <div><dt>{copy.quality.translationRuntime}</dt><dd>{infrastructureLabel(runTranslation?.runtime, copy.notRecorded)}</dd></div>
                  <div><dt>{copy.quality.roundTripGate}</dt><dd>{translationThreshold !== undefined ? copy.quality.roundTripGateValue(formatPercent(translationThreshold, copy.locale)) : copy.notRecorded}</dd></div>
                  <div><dt>{copy.quality.manualLanguageReviewGate}</dt><dd>{manualReviewRequired ? manualReviewIsApproved ? copy.quality.manualLanguageReviewApproved : copy.quality.manualLanguageReviewRequired(manualReviewLocales.map((locale) => locale.toLowerCase() === "tr" ? copy.quality.turkishLocale : locale.toLowerCase() === "en" ? copy.quality.englishLocale : locale).join(", ")) : copy.quality.manualLanguageReviewNotRequired}</dd></div>
                </dl>
                {(manualReviewRequired || manualReview) && <section className={`manual-review-record ${manualReviewIsApproved ? "approved" : "pending"}`} aria-labelledby="manual-review-title">
                  <div className="manual-review-heading"><div><span className="section-kicker">{copy.quality.manualReviewKicker}</span><h3 id="manual-review-title">{copy.quality.manualReviewTitle}</h3></div><span className={`translation-verification ${manualReviewIsApproved ? "verified" : "unverified"}`}>{manualReviewIsApproved ? <CircleCheck size={15} aria-hidden="true" /> : <CircleAlert size={15} aria-hidden="true" />}{manualReviewIsApproved ? copy.quality.manualReviewApproved : copy.quality.manualReviewPending}</span></div>
                  <dl className="manual-review-grid">
                    <div><dt>{copy.quality.manualReviewer}</dt><dd><strong>{manualReview?.reviewer || copy.notRecorded}</strong>{manualReview?.reviewer_type && <small>{reviewerTypeLabel(manualReview.reviewer_type, copy)}</small>}</dd></div>
                    <div><dt>{copy.quality.manualReviewedAt}</dt><dd>{manualReview?.reviewed_at ? <time dateTime={manualReview.reviewed_at} title={manualReview.reviewed_at}>{formatDate(manualReview.reviewed_at, copy.locale, copy.notRecorded)}</time> : copy.notRecorded}</dd></div>
                    <div><dt>{copy.quality.manualCasesReviewed}</dt><dd>{manualReview?.cases_reviewed === undefined ? copy.notRecorded : copy.quality.manualCasesValue(manualReview.cases_reviewed)}</dd></div>
                    <div><dt>{copy.quality.manualAnswersDigest}</dt><dd>{manualReview?.answers_sha256 ? <code title={manualReview.answers_sha256}>{shortDigest(manualReview.answers_sha256)}</code> : copy.notRecorded}</dd></div>
                  </dl>
                  {!!manualReview?.criteria?.length && <p className="manual-review-criteria">{copy.quality.manualCriteriaRecorded(manualReview.criteria.length)}</p>}
                </section>}
              </section>

              <section className="panel context-panel" aria-labelledby="run-context-title">
                <div className="panel-heading"><div><span className="section-kicker">{copy.quality.contextKicker}</span><h2 id="run-context-title">{copy.quality.contextTitle}</h2><p>{copy.quality.contextIntro}</p></div></div>
                {!provenanceVerified && <div className="provenance-alert" role="alert"><CircleAlert size={20} aria-hidden="true" /><div><strong>{copy.quality.provenanceWarningTitle}</strong><p>{codeProvenance?.dirty === true ? copy.quality.provenanceDirtyCopy : copy.quality.provenanceMissingCopy}</p></div></div>}
                <dl className="context-grid">
                  <div><dt>{copy.quality.runId}</dt><dd title={qualityRun?.id}>{qualityRun?.id || "—"}</dd></div>
                  <div><dt>{copy.quality.dataset}</dt><dd>{qualityRun?.dataset_id || runContext?.dataset_id || "—"}</dd></div>
                  <div><dt>{copy.quality.revision}</dt><dd>{codeRevision ? <code title={codeRevision}>{shortRevision(codeRevision)}</code> : <span className="provenance-state unverified">{copy.quality.notVerified}</span>}</dd></div>
                  <div><dt>{copy.quality.workingTree}</dt><dd><span className={`provenance-state ${provenanceVerified ? "verified" : "unverified"}`}>{provenanceVerified ? copy.quality.clean : codeProvenance?.dirty === true ? copy.quality.dirty : copy.quality.notVerified}</span></dd></div>
                  <div><dt>{copy.quality.runtime}</dt><dd>{infrastructureLabel(runRuntime.runtime ?? runRuntime.status)}</dd></div>
                  <div><dt>{copy.quality.model}</dt><dd>{nestedLabel(runRuntime.chat_model)}</dd></div>
                  <div><dt>{copy.quality.provider}</dt><dd>{infrastructureLabel(runRuntime.execution_provider ?? (typeof runRuntime.chat_model === "object" && runRuntime.chat_model ? (runRuntime.chat_model as Record<string, unknown>).execution_provider : undefined))}</dd></div>
                  <div><dt>{copy.quality.corpus}</dt><dd>{copy.quality.corpusValue(runContext?.documents ?? "—", runContext?.chunks ?? "—")}</dd></div>
                  <div><dt>{copy.quality.configuration}</dt><dd>{copy.quality.configurationValue(Number(runConfiguration.top_k ?? qualityMetrics.top_k ?? 5), formatDecimal(Number(runConfiguration.retrieval_threshold ?? qualityMetrics.retrieval_threshold ?? 0), copy.locale))}</dd></div>
                </dl>
              </section>

              {!!evaluationCases.length && (
                <section className="case-results" aria-labelledby="case-results-title">
                  <div className="case-results-heading"><div><span className="section-kicker">{copy.quality.casesKicker}</span><h2 id="case-results-title">{copy.quality.casesTitle}</h2><p>{copy.quality.casesIntro}</p></div></div>
                  <div className="case-filters" role="group" aria-label={copy.quality.filterLabel}>
                    {([
                      ["all", copy.quality.allCases],
                      ["passed", copy.quality.passedOnly],
                      ["failed", copy.quality.needsReview],
                      ["en", copy.quality.englishCases],
                      ["tr", copy.quality.turkishCases],
                    ] as Array<[CaseFilter, string]>).map(([value, label]) => <button key={value} aria-pressed={caseFilter === value} onClick={() => setCaseFilter(value)}>{label}</button>)}
                  </div>
                  <div className="case-list">
                    {!filteredCases.length ? <p className="case-empty">{copy.quality.noMatchingCases}</p> : filteredCases.map((item, index) => {
                      const id = item.id || `#${index + 1}`;
                      const locale = caseLocale(item);
                      const sentenceCoverage = item.citation_sentence_coverage;
                      const categoryKey = item.category as keyof typeof copy.quality.categories | undefined;
                      const categoryLabel = categoryKey && copy.quality.categories[categoryKey]
                        ? copy.quality.categories[categoryKey]
                        : item.answerable === false ? copy.quality.refusalCase : copy.quality.answerable;
                      return (
                        <details className={`case-item ${item.passed ? "pass" : "fail"}`} key={id}>
                          <summary>
                            <span className="case-result-icon" aria-hidden="true">{item.passed ? <Check size={18} /> : <CircleAlert size={18} />}</span>
                            <span className="case-summary-copy"><span className="case-tags"><b>{id}</b><i>{locale.toUpperCase()}</i><i>{categoryLabel}</i></span><strong>{item.question || "—"}</strong></span>
                            <span className="case-result-label">{item.passed ? copy.quality.casePassed : copy.quality.caseFailed}</span>
                          </summary>
                          <div className="case-detail">
                            <div className="case-answer"><span>{copy.quality.answer}</span><p>{item.answer || "—"}</p></div>
                            <div className="case-sources">
                              <div><h3>{copy.quality.expectedSources}</h3><SourceList values={item.expected_sources} empty={copy.quality.noSources} /></div>
                              <div><h3>{copy.quality.retrievedSources}</h3><SourceList values={item.retrieved_sources} empty={copy.quality.noSources} /></div>
                              <div><h3>{copy.quality.citedSources}</h3><SourceList values={item.cited_sources} empty={copy.quality.noSources} /></div>
                            </div>
                            <dl className="case-metrics">
                              <div><dt>{copy.quality.locale}</dt><dd>{locale === "tr" ? copy.quality.turkishLocale : copy.quality.englishLocale}</dd></div>
                              <div><dt>{copy.quality.category}</dt><dd>{categoryLabel}</dd></div>
                              <div><dt>{copy.quality.hit}</dt><dd>{item.hit === null || item.hit === undefined ? "—" : item.hit ? copy.quality.yes : copy.quality.no}</dd></div>
                              <div><dt>{copy.quality.sourceRecall}</dt><dd>{formatPercent(item.source_recall ?? undefined, copy.locale)}</dd></div>
                              <div><dt>{copy.quality.citationPrecision}</dt><dd>{formatPercent(item.citation_precision ?? undefined, copy.locale)}</dd></div>
                              <div><dt>{copy.quality.factRecall}</dt><dd>{formatPercent(item.required_fact_recall ?? undefined, copy.locale)}</dd></div>
                              {sentenceCoverage !== null && sentenceCoverage !== undefined && <div><dt>{copy.quality.sentenceCoverage}</dt><dd>{typeof sentenceCoverage === "boolean" ? (sentenceCoverage ? copy.quality.yes : copy.quality.no) : formatPercent(sentenceCoverage, copy.locale)}</dd></div>}
                              {item.protected_fact_grounded !== null && item.protected_fact_grounded !== undefined && <div><dt>{copy.quality.protectedFacts}</dt><dd>{item.protected_fact_grounded ? copy.quality.yes : copy.quality.no}</dd></div>}
                              {item.language_quality !== null && item.language_quality !== undefined && <div><dt>{copy.quality.languageQuality}</dt><dd>{item.language_quality ? copy.quality.yes : copy.quality.no}</dd></div>}
                              <div><dt>{copy.quality.refusalCorrect}</dt><dd>{item.refusal_correct === null || item.refusal_correct === undefined ? "—" : item.refusal_correct ? copy.quality.yes : copy.quality.no}</dd></div>
                              <div><dt>{copy.quality.retrievalTime}</dt><dd>{item.retrieval_ms !== undefined ? `${Math.round(item.retrieval_ms)} ms` : "—"}</dd></div>
                            </dl>
                          </div>
                        </details>
                      );
                    })}
                  </div>
                </section>
              )}
            </>}
          </section>
        )}
      </main>

      <footer><Brand copy={copy} /><p>{copy.footer.copy}</p><span>{copy.footer.builtFor}</span></footer>

      {preferencesOpen && createPortal(
        <div className="preferences-layer" role="presentation" onMouseDown={(event) => {
          if (event.target === event.currentTarget) {
            setPreferencesOpen(false);
            window.requestAnimationFrame(() => preferencesButton.current?.focus());
          }
        }}>
          <section ref={preferencesDialog} id="tracefold-preferences" className="preferences-popover" role="dialog" aria-modal="true" aria-labelledby="preferences-title" aria-describedby="preferences-description">
            <div className="preferences-heading">
              <div><span className="section-kicker">Tracefold</span><h2 id="preferences-title">{copy.preferences.title}</h2><p id="preferences-description">{copy.preferences.description}</p></div>
              <button ref={preferencesClose} className="icon-button" onClick={() => { setPreferencesOpen(false); preferencesButton.current?.focus(); }} aria-label={copy.preferences.close}><X size={17} aria-hidden="true" /></button>
            </div>
            <PreferenceGroup<Language>
              label={copy.preferences.language}
              hint={copy.preferences.languageHint}
              name="language"
              value={preferences.language}
              options={[{ value: "en", label: copy.preferences.english, icon: <Languages size={14} aria-hidden="true" /> }, { value: "tr", label: copy.preferences.turkish, icon: <Languages size={14} aria-hidden="true" /> }]}
              onChange={(language) => setPreferences((current) => ({ ...current, language }))}
            />
            <PreferenceGroup<ThemePreference>
              label={copy.preferences.appearance}
              hint={copy.preferences.appearanceHint}
              name="theme"
              value={preferences.theme}
              options={[{ value: "system", label: copy.preferences.system, icon: <Monitor size={14} aria-hidden="true" /> }, { value: "light", label: copy.preferences.light, icon: <Sun size={14} aria-hidden="true" /> }, { value: "dark", label: copy.preferences.dark, icon: <Moon size={14} aria-hidden="true" /> }]}
              onChange={(theme) => setPreferences((current) => ({ ...current, theme }))}
            />
            <PreferenceGroup<MotionPreference>
              label={copy.preferences.motion}
              hint={copy.preferences.motionHint}
              name="motion"
              value={preferences.motion}
              options={[{ value: "system", label: copy.preferences.system, icon: <Monitor size={14} aria-hidden="true" /> }, { value: "full", label: copy.preferences.full, icon: <Waves size={14} aria-hidden="true" /> }, { value: "reduced", label: copy.preferences.reduced, icon: <ShieldCheck size={14} aria-hidden="true" /> }]}
              onChange={(motion) => setPreferences((current) => ({ ...current, motion }))}
            />
            <div className="preferences-saved"><Check size={14} aria-hidden="true" />{copy.preferences.saved}</div>
          </section>
        </div>,
        document.body,
      )}

      {selectedCitation && (
        <div className="drawer-layer" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) closeCitation(); }}>
          <section ref={evidenceDialog} className="evidence-drawer" role="dialog" aria-modal="true" aria-labelledby="evidence-title" aria-describedby="evidence-description">
            <div className="drawer-header">
              <div><span className="section-kicker">{copy.evidence.kicker}</span><h2 id="evidence-title">{copy.evidence.source(selectedCitation.index)}</h2></div>
              <button ref={evidenceCloseButton} className="icon-button" autoFocus onClick={closeCitation} aria-label={copy.evidence.close}><PanelRightClose size={20} /></button>
            </div>
            <div className="source-card"><span className="document-icon"><File size={19} /></span><div><strong>{selectedCitation.source}</strong><span>{selectedCitation.page !== undefined ? copy.evidence.page(selectedCitation.page) : selectedCitation.section || copy.evidence.passage}</span></div></div>
            <div className="evidence-meta"><span>{copy.evidence.similarity} <strong>{selectedCitation.score !== undefined ? formatPercent(selectedCitation.score, copy.locale) : copy.evidence.notReported}</strong></span>{selectedCitation.chunk_id && <span>{copy.evidence.chunk} <strong>{selectedCitation.chunk_id}</strong></span>}</div>
            <blockquote><span>{copy.evidence.excerpt}</span><p>{formatExcerpt(selectedCitation.excerpt)}</p></blockquote>
            <div className="drawer-foot"><ShieldCheck size={16} aria-hidden="true" /><p id="evidence-description">{copy.evidence.foot}</p></div>
          </section>
        </div>
      )}
    </div>
  );
}

export default App;
