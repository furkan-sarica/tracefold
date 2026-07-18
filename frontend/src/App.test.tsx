import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import App from "./App";

const releaseRevision = "9f8e7d6c5b4a32100123456789abcdef01234567";
const trToEnRevision = "2261c8fc7b1af59caee87f8ff0ecf3fbccfe8391";
const enToTrRevision = "e539fc16a8a1a0ea5950eb339b595bfcce990e90";
const reviewDigest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const translationPayload = {
  runtime: "marian_opus",
  ready: true,
  offline_only: true,
  models: {
    tr_to_en: { id: "Helsinki-NLP/opus-mt-tc-big-tr-en", revision: trToEnRevision, cached: true, loaded: true },
    en_to_tr: { id: "Helsinki-NLP/opus-mt-tc-big-en-tr", revision: enToTrRevision, cached: true, loaded: true },
  },
  glossary: { id: "northstar-summer-lab-en-tr-v1", entries: 42, scope: "Northstar demo corpus" },
  round_trip_threshold: 0.6,
  error: null,
};
const runtimeTranslationPayload = {
  ...translationPayload,
  models: {
    tr_to_en: { ...translationPayload.models.tr_to_en, loaded: false },
    en_to_tr: { ...translationPayload.models.en_to_tr, loaded: false },
  },
};

const evaluationPayload = {
  id: "eval_bilingual_release",
  dataset_id: "northstar-summer-lab-2026-bilingual-parity-v3",
  status: "complete",
  created_at: "2026-07-18T09:30:00Z",
  metrics: {
    total_cases: 24,
    passed_cases: 24,
    hit_at_k: 1,
    source_recall_at_k: 1,
    citation_accuracy: 1,
    required_fact_recall: 0.91,
    refusal_accuracy: 1,
    retrieval_p95_ms: 118,
    language_quality_accuracy: 1,
    citation_sentence_coverage: 1,
    protected_fact_grounding_accuracy: 1,
    bilingual_pair_accuracy: 1,
    locales: ["en", "tr"],
    locale_breakdown: {
      en: { total_cases: 12, passed_cases: 12 },
      tr: { total_cases: 12, passed_cases: 12 },
    },
    run_context: {
      dataset_id: "northstar-summer-lab-2026-bilingual-parity-v3",
      runtime: {
        runtime: "foundry_local",
        chat_model: { alias: "release-chat-model", execution_provider: "WebGpuExecutionProvider" },
        execution_provider: "WebGpuExecutionProvider",
      },
      configuration: {
        top_k: 5,
        retrieval_threshold: 0.6,
        translation_round_trip_threshold: 0.6,
        translation_glossary_id: "northstar-summer-lab-en-tr-v1",
        quality_policy: { manual_language_review_required_for_release: ["tr"] },
      },
      documents: 8,
      chunks: 53,
      code_provenance: { revision: releaseRevision, dirty: false, source: "git" },
      translation: translationPayload,
      manual_language_review: {
        status: "approved",
        verified: true,
        locale: "tr",
        reviewer: "Tracefold release QA",
        reviewer_type: "case-by-case release audit",
        reviewed_at: "2026-07-18T12:00:00+00:00",
        cases_reviewed: 12,
        answers_sha256: reviewDigest,
        criteria: ["natural Turkish", "complete meaning", "exact protected facts", "natural terminology", "grounded citations"],
      },
    },
  },
  cases: [
    {
      id: "NSL-A01",
      locale: "en",
      category: "single_source_fact",
      question: "When does the lab begin?",
      answer: "The lab begins on 3 August 2026.",
      answerable: true,
      passed: true,
      expected_sources: ["01-program-handbook.md"],
      retrieved_sources: ["01-program-handbook.md"],
      cited_sources: ["01-program-handbook.md"],
      hit: true,
      source_recall: 1,
      citation_precision: 1,
      required_fact_recall: 1,
      language_quality: true,
      protected_fact_grounded: true,
      citation_sentence_coverage: 1,
      refusal_correct: true,
      retrieval_ms: 42,
    },
    {
      id: "NSL-TR-A01",
      locale: "tr",
      category: "single_source_fact",
      question: "Program ne zaman başlıyor?",
      answer: "Program 3 Ağustos 2026 tarihinde başlıyor.",
      answerable: true,
      passed: true,
      expected_sources: ["01-program-handbook.md"],
      retrieved_sources: ["01-program-handbook.md"],
      cited_sources: ["01-program-handbook.md"],
      hit: true,
      source_recall: 1,
      citation_precision: 1,
      required_fact_recall: 1,
      language_quality: true,
      protected_fact_grounded: true,
      citation_sentence_coverage: 1,
      refusal_correct: true,
      retrieval_ms: 47,
    },
  ],
};

function response(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function installApiMock(evaluation = evaluationPayload) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith("/api/runtime")) return response({
      status: "ready",
      ready: true,
      runtime: "foundry_local",
      chat_model: { alias: "release-chat-model", execution_provider: "WebGpuExecutionProvider" },
      embedding_model: { alias: "release-embedding-model", execution_provider: "WebGpuExecutionProvider" },
      translation: runtimeTranslationPayload,
    });
    if (url.endsWith("/api/documents") && init?.method === "POST") return response({ id: "doc-new", status: "ready" }, 201);
    if (url.endsWith("/api/documents")) return response({ documents: [{ id: "doc-1", filename: "01-program-handbook.md", chunks: 8 }] });
    if (url.endsWith("/api/evaluations/latest")) return response(evaluation);
    if (url.endsWith("/api/chat/stream")) return response({
      answer: "The program begins on 3 August 2026.",
      refused: false,
      citations: [{ id: "chunk-1", index: 1, source: "01-program-handbook.md", excerpt: "The program begins on 3 August 2026.", score: 0.92 }],
    });
    throw new Error(`Unexpected request: ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

test("shows dynamic bilingual quality targets and case drilldown", async () => {
  installApiMock();
  const user = userEvent.setup();
  render(<App />);

  const navigation = screen.getByRole("navigation", { name: "Primary navigation" });
  const askTab = within(navigation).getByRole("button", { name: "Ask" });
  expect(askTab).toHaveAttribute("aria-current", "page");
  await user.click(within(navigation).getByRole("button", { name: "Quality" }));

  expect(await screen.findByRole("heading", { name: "Quality report" })).toBeVisible();
  expect(await screen.findByText("24 of 24 cases passed")).toBeVisible();
  expect(screen.getByText("northstar-summer-lab-2026-bilingual-parity-v3")).toBeVisible();
  expect(screen.getByText("Results by language")).toBeVisible();
  expect(screen.getAllByText(/Target met/).length).toBeGreaterThan(0);
  expect(screen.getByRole("status", { name: "Release accepted" })).toHaveClass("accepted");
  expect(screen.getByRole("heading", { name: "Translation provenance and gates" })).toBeVisible();
  expect(screen.getByText("Marian OPUS-MT")).toBeVisible();
  expect(screen.getByText("Foundry Local")).toBeVisible();
  expect(screen.getByText("WebGPU")).toBeVisible();
  expect(screen.queryByText("foundry_local")).not.toBeInTheDocument();
  expect(screen.queryByText("WebGpuExecutionProvider")).not.toBeInTheDocument();
  expect(screen.getByText("≥ 60% content recall required")).toBeVisible();
  expect(screen.getAllByText("Approved")).toHaveLength(2);
  expect(screen.getByRole("heading", { name: "Manual language review" })).toBeVisible();
  expect(screen.getByText("Tracefold release QA")).toBeVisible();
  expect(screen.getByText("Case-by-case release audit")).toBeVisible();
  expect(screen.getByText("12 Turkish cases")).toBeVisible();
  expect(screen.getByText(`${reviewDigest.slice(0, 12)}…`)).toHaveAttribute("title", reviewDigest);
  expect(screen.getByTitle("2026-07-18T12:00:00+00:00")).toBeVisible();
  expect(screen.getByText(releaseRevision.slice(0, 10))).toHaveAttribute("title", releaseRevision);
  expect(screen.getByText("Clean")).toBeVisible();
  expect(screen.queryByText("Stale or unverified code provenance")).not.toBeInTheDocument();

  await user.click(screen.getByText("When does the lab begin?"));
  expect(screen.getByText("The lab begins on 3 August 2026.")).toBeVisible();
  const openedCase = screen.getByText("When does the lab begin?").closest("details");
  expect(openedCase).not.toBeNull();
  expect(within(openedCase as HTMLElement).getAllByText("01-program-handbook.md")).toHaveLength(3);

  await user.click(screen.getByRole("button", { name: "Turkish" }));
  expect(screen.queryByText("When does the lab begin?")).not.toBeInTheDocument();
  expect(screen.getByText("Program ne zaman başlıyor?")).toBeVisible();
});

test("keeps a clean 24 of 24 run non-green while manual language review is pending", async () => {
  const pendingRelease = structuredClone(evaluationPayload);
  pendingRelease.metrics.run_context.manual_language_review = { status: "pending", verified: false, locale: "tr" } as typeof pendingRelease.metrics.run_context.manual_language_review;
  installApiMock(pendingRelease);
  const user = userEvent.setup();
  render(<App />);

  const navigation = screen.getByRole("navigation", { name: "Primary navigation" });
  await user.click(within(navigation).getByRole("button", { name: "Quality" }));

  expect(await screen.findByRole("status", { name: "Automated gates passed · Manual language review required" })).toHaveClass("unverified");
  expect(screen.getByRole("heading", { name: "Manual language review" })).toBeVisible();
  expect(screen.getByText("Pending or incomplete")).toBeVisible();
  expect(screen.getByText("Turkish manual language review required for release")).toBeVisible();
});

test.each([
  ["verification flag", { verified: false }],
  ["status", { status: "pending" }],
  ["locale", { locale: "en" }],
  ["reviewer", { reviewer: "   " }],
  ["reviewer type", { reviewer_type: "   " }],
  ["review timestamp", { reviewed_at: "not-a-date" }],
  ["case count", { cases_reviewed: 11 }],
  ["answer digest", { answers_sha256: "abc123" }],
  ["review criteria", { criteria: ["one", "two", "three", "four"] }],
] as const)("fails the manual language review gate for an invalid %s", async (_field, patch) => {
  const invalidReview = structuredClone(evaluationPayload);
  Object.assign(invalidReview.metrics.run_context.manual_language_review, patch);
  installApiMock(invalidReview);
  const user = userEvent.setup();
  render(<App />);

  const navigation = screen.getByRole("navigation", { name: "Primary navigation" });
  await user.click(within(navigation).getByRole("button", { name: "Quality" }));

  expect(await screen.findByRole("status", { name: "Automated gates passed · Manual language review required" })).toHaveClass("unverified");
});

test("shows pinned OPUS-MT checkpoints and glossary in Setup in both languages", async () => {
  installApiMock();
  const user = userEvent.setup();
  render(<App />);

  const navigation = screen.getByRole("navigation", { name: "Primary navigation" });
  await user.click(within(navigation).getByRole("button", { name: "Setup" }));

  expect(await screen.findByRole("heading", { name: "Offline OPUS-MT translation" })).toBeVisible();
  expect(screen.getByText("Helsinki-NLP/opus-mt-tc-big-tr-en")).toBeVisible();
  expect(screen.getByText("Helsinki-NLP/opus-mt-tc-big-en-tr")).toBeVisible();
  expect(screen.getByText(trToEnRevision.slice(0, 10))).toHaveAttribute("title", trToEnRevision);
  expect(screen.getByText(enToTrRevision.slice(0, 10))).toHaveAttribute("title", enToTrRevision);
  expect(screen.getByText("northstar-summer-lab-en-tr-v1")).toBeVisible();
  expect(screen.getByText("42")).toBeVisible();
  expect(screen.getAllByText("Cached locally")).toHaveLength(2);
  expect(screen.getByText("Checkpoints verified")).toBeVisible();

  await user.click(screen.getByRole("button", { name: "Open preferences" }));
  await user.click(within(screen.getByRole("dialog", { name: "Preferences" })).getByRole("radio", { name: "Türkçe" }));
  await user.keyboard("{Escape}");
  expect(screen.getByRole("heading", { name: "Çevrimdışı OPUS-MT çevirisi" })).toBeVisible();
  expect(screen.getByText("Türkçe → İngilizce")).toBeVisible();
  expect(screen.getByText("Model sürümleri doğrulandı")).toBeVisible();
  expect(screen.getByText("Sözlük sürümü")).toBeVisible();
});

test("blocks release acceptance when the evaluation lacks translation provenance", async () => {
  const withoutTranslation = structuredClone(evaluationPayload);
  delete (withoutTranslation.metrics.run_context as Partial<typeof withoutTranslation.metrics.run_context>).translation;
  installApiMock(withoutTranslation);
  const user = userEvent.setup();
  render(<App />);

  const navigation = screen.getByRole("navigation", { name: "Primary navigation" });
  await user.click(within(navigation).getByRole("button", { name: "Quality" }));

  expect(await screen.findByRole("status", { name: "Translation provenance unverified" })).toHaveClass("unverified");
  expect(screen.getByRole("heading", { name: "Translation provenance and gates" })).toBeVisible();
  expect(screen.getByText("Translation provenance is incomplete")).toBeVisible();
  expect(screen.getByText(/cannot be accepted for release until both pinned checkpoints/)).toBeVisible();
});

test("requires both OPUS checkpoints to be cached and loaded in evaluation provenance", async () => {
  const cachedOnlyEvaluation = structuredClone(evaluationPayload);
  cachedOnlyEvaluation.metrics.run_context.translation.models.tr_to_en.loaded = false;
  cachedOnlyEvaluation.metrics.run_context.translation.models.en_to_tr.loaded = false;
  installApiMock(cachedOnlyEvaluation);
  const user = userEvent.setup();
  render(<App />);

  const navigation = screen.getByRole("navigation", { name: "Primary navigation" });
  await user.click(within(navigation).getByRole("button", { name: "Quality" }));

  expect(await screen.findByRole("status", { name: "Translation provenance unverified" })).toHaveClass("unverified");
  expect(screen.getByText("Translation provenance is incomplete")).toBeVisible();
});

test("blocks the green release state when code provenance is missing", async () => {
  const withoutProvenance = structuredClone(evaluationPayload);
  delete (withoutProvenance.metrics.run_context as Partial<typeof withoutProvenance.metrics.run_context>).code_provenance;
  installApiMock(withoutProvenance);
  const user = userEvent.setup();
  render(<App />);

  const navigation = screen.getByRole("navigation", { name: "Primary navigation" });
  await user.click(within(navigation).getByRole("button", { name: "Quality" }));

  expect(await screen.findByRole("status", { name: "Release unverified" })).toHaveClass("unverified");
  expect(screen.getByRole("alert")).toHaveTextContent("Stale or unverified code provenance");
  expect(screen.getByRole("alert")).toHaveTextContent("Rerun the evaluation from a clean commit");
  expect(screen.getAllByText("Not verified")).toHaveLength(2);
});

test("marks a dirty revision as unverified in English and Turkish", async () => {
  const dirtyEvaluation = structuredClone(evaluationPayload);
  dirtyEvaluation.metrics.run_context.code_provenance.dirty = true;
  installApiMock(dirtyEvaluation);
  const user = userEvent.setup();
  render(<App />);

  const navigation = screen.getByRole("navigation", { name: "Primary navigation" });
  await user.click(within(navigation).getByRole("button", { name: "Quality" }));

  expect(await screen.findByRole("status", { name: "Release unverified" })).toHaveClass("unverified");
  expect(screen.getByText("Uncommitted changes")).toBeVisible();
  expect(screen.getByRole("alert")).toHaveTextContent("This evaluation was run with uncommitted changes");

  await user.click(screen.getByRole("button", { name: "Open preferences" }));
  await user.click(within(screen.getByRole("dialog", { name: "Preferences" })).getByRole("radio", { name: "Türkçe" }));
  await user.keyboard("{Escape}");
  expect(screen.getByRole("status", { name: "Yayın doğrulanmadı" })).toBeVisible();
  expect(screen.getByText("Git’e işlenmemiş değişiklik var")).toBeVisible();
  expect(screen.getByRole("alert")).toHaveTextContent("Eski veya doğrulanmamış kod kaydı");
});

test("accepts the long Markdown extension and exposes it in the file picker", async () => {
  const fetchMock = installApiMock();
  const user = userEvent.setup();
  render(<App />);

  const navigation = screen.getByRole("navigation", { name: "Primary navigation" });
  await user.click(within(navigation).getByRole("button", { name: "Library" }));
  const input = document.querySelector<HTMLInputElement>('input[type="file"]');
  expect(input).not.toBeNull();
  expect(input).toHaveAttribute("accept", ".pdf,.docx,.txt,.md,.markdown");

  await user.upload(input as HTMLInputElement, new File(["# Release notes"], "release-notes.markdown", { type: "text/markdown" }));
  expect(await screen.findByText("release-notes.markdown")).toBeVisible();
  expect(fetchMock).toHaveBeenCalledWith("/api/documents", expect.objectContaining({ method: "POST" }));
});

test("persists Turkish preferences, updates navigation, and restores focus on Escape", async () => {
  installApiMock();
  const user = userEvent.setup();
  render(<App />);

  const trigger = screen.getByRole("button", { name: "Open preferences" });
  await user.click(trigger);
  const dialog = screen.getByRole("dialog", { name: "Preferences" });
  expect(dialog).toBeVisible();
  await user.click(within(dialog).getByRole("radio", { name: "Türkçe" }));

  await waitFor(() => expect(document.documentElement).toHaveAttribute("lang", "tr"));
  expect(screen.getByRole("navigation", { name: "Ana menü" })).toBeVisible();
  await user.keyboard("{Escape}");
  expect(screen.queryByRole("dialog", { name: "Tercihler" })).not.toBeInTheDocument();
  await waitFor(() => expect(screen.getByRole("button", { name: "Tercihleri aç" })).toHaveFocus());
  expect(JSON.parse(window.localStorage.getItem("tracefold.preferences.v1") || "{}").language).toBe("tr");
});

test("submits a grounded question and returns focus after closing evidence", async () => {
  installApiMock();
  const user = userEvent.setup();
  render(<App />);

  const composer = await screen.findByRole("textbox", { name: "Ask your library" });
  await waitFor(() => expect(composer).toBeEnabled());
  await user.type(composer, "When does it begin?");
  await user.click(screen.getByRole("button", { name: "Send question" }));

  expect(await screen.findByText("The program begins on 3 August 2026.")).toBeVisible();
  const citation = screen.getByRole("button", { name: /01-program-handbook\.md/ });
  await user.click(citation);
  expect(screen.getByRole("dialog", { name: "Source [1]" })).toBeVisible();
  await user.keyboard("{Escape}");
  expect(screen.queryByRole("dialog", { name: "Source [1]" })).not.toBeInTheDocument();
  await waitFor(() => expect(citation).toHaveFocus());
});
