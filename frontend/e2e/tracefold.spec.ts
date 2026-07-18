import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const releaseRevision = "9f8e7d6c5b4a32100123456789abcdef01234567";
const trToEnRevision = "2261c8fc7b1af59caee87f8ff0ecf3fbccfe8391";
const enToTrRevision = "e539fc16a8a1a0ea5950eb339b595bfcce990e90";
const reviewDigest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const translation = {
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
const runtimeTranslation = {
  ...translation,
  models: {
    tr_to_en: { ...translation.models.tr_to_en, loaded: false },
    en_to_tr: { ...translation.models.en_to_tr, loaded: false },
  },
};

const evaluation = {
  id: "eval_browser_release",
  dataset_id: "northstar-summer-lab-2026-bilingual-parity-v3",
  status: "complete",
  created_at: "2026-07-18T09:30:00Z",
  metrics: {
    total_cases: 24,
    passed_cases: 24,
    hit_at_k: 1,
    source_recall_at_k: 1,
    citation_accuracy: 1,
    refusal_accuracy: 1,
    required_fact_recall: 0.91,
    retrieval_p95_ms: 118,
    language_quality_accuracy: 1,
    citation_sentence_coverage: 1,
    protected_fact_grounding_accuracy: 1,
    bilingual_pair_accuracy: 1,
    locales: ["en", "tr"],
    locale_breakdown: { en: { total_cases: 12, passed_cases: 12 }, tr: { total_cases: 12, passed_cases: 12 } },
    run_context: {
      dataset_id: "northstar-summer-lab-2026-bilingual-parity-v3",
      runtime: { runtime: "foundry_local", chat_model: { alias: "release-chat-model" }, execution_provider: "WebGpuExecutionProvider" },
      configuration: { top_k: 5, retrieval_threshold: 0.6, translation_round_trip_threshold: 0.6, translation_glossary_id: "northstar-summer-lab-en-tr-v1", quality_policy: { manual_language_review_required_for_release: ["tr"] } },
      documents: 8,
      chunks: 53,
      code_provenance: { revision: releaseRevision, dirty: false, source: "git" },
      translation,
      manual_language_review: { status: "approved", verified: true, locale: "tr", reviewer: "Tracefold release QA", reviewer_type: "case-by-case release audit", reviewed_at: "2026-07-18T12:00:00+00:00", cases_reviewed: 12, answers_sha256: reviewDigest, criteria: ["natural Turkish", "complete meaning", "exact protected facts", "natural terminology", "grounded citations"] },
    },
  },
  cases: [
    { id: "NSL-A01", locale: "en", category: "single_source_fact", question: "When does the lab begin?", answer: "The lab begins on 3 August 2026.", answerable: true, passed: true, expected_sources: ["01-program-handbook.md"], retrieved_sources: ["01-program-handbook.md"], cited_sources: ["01-program-handbook.md"], hit: true, source_recall: 1, citation_precision: 1, required_fact_recall: 1, refusal_correct: true, retrieval_ms: 42 },
    { id: "NSL-TR-A01", locale: "tr", category: "single_source_fact", question: "Program ne zaman başlıyor?", answer: "Program 3 Ağustos 2026 tarihinde başlıyor.", answerable: true, passed: true, expected_sources: ["01-program-handbook.md"], retrieved_sources: ["01-program-handbook.md"], cited_sources: ["01-program-handbook.md"], hit: true, source_recall: 1, citation_precision: 1, required_fact_recall: 1, language_quality: true, protected_fact_grounded: true, citation_sentence_coverage: 1, refusal_correct: true, retrieval_ms: 47 },
  ],
};

async function mockApi(page: Page) {
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    let body: unknown;
    if (url.pathname === "/api/runtime") body = { status: "ready", ready: true, runtime: "foundry_local", chat_model: { alias: "release-chat-model", execution_provider: "WebGpuExecutionProvider" }, embedding_model: { alias: "release-embedding-model", execution_provider: "WebGpuExecutionProvider" }, translation: runtimeTranslation };
    else if (url.pathname === "/api/documents") body = { documents: [{ id: "doc-1", filename: "01-program-handbook.md", chunks: 8 }] };
    else if (url.pathname === "/api/evaluations/latest" || url.pathname === "/api/evaluations/run") body = evaluation;
    else if (url.pathname === "/api/chat/stream") body = { answer: "The program begins on 3 August 2026.", refused: false, citations: [{ id: "chunk-1", index: 1, source: "01-program-handbook.md", excerpt: "The program begins on 3 August 2026.", score: 0.92 }] };
    else return route.fulfill({ status: 404, contentType: "application/json", body: JSON.stringify({ detail: "Not found" }) });
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
}

async function openNavigation(page: Page, mobile: boolean) {
  if (mobile) await page.getByRole("button", { name: "Open navigation" }).click();
}

async function expectNoBlockingA11yViolations(page: Page) {
  const results = await new AxeBuilder({ page }).analyze();
  const blocking = results.violations
    .filter((violation) => violation.impact === "serious" || violation.impact === "critical")
    .map((violation) => ({
      id: violation.id,
      impact: violation.impact,
      description: violation.description,
      targets: violation.nodes.flatMap((node) => node.target),
    }));
  expect(blocking).toEqual([]);
}

test.beforeEach(async ({ page }) => {
  await mockApi(page);
  await page.emulateMedia({ colorScheme: "light", reducedMotion: "reduce" });
  await page.goto("/");
});

test("quality report exposes targets, bilingual totals, context, and case evidence", async ({ page }, testInfo) => {
  const mobile = testInfo.project.name === "mobile-chromium";
  await openNavigation(page, mobile);
  await page.getByRole("button", { name: "Quality", exact: true }).click();

  await expect(page.getByRole("heading", { name: "Quality report" })).toBeVisible();
  await expect(page.getByText("24 of 24 cases passed")).toBeVisible();
  await expect(page.getByText("northstar-summer-lab-2026-bilingual-parity-v3")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Results by language" })).toBeVisible();
  await expect(page.getByText("Target met").first()).toBeVisible();
  await expect(page.getByRole("status", { name: "Release accepted" })).toBeVisible();
  await expect(page.getByText(releaseRevision.slice(0, 10))).toHaveAttribute("title", releaseRevision);
  await expect(page.getByText("Clean")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Translation provenance and gates" })).toBeVisible();
  await expect(page.getByText("Marian OPUS-MT")).toBeVisible();
  await expect(page.getByText("Foundry Local", { exact: true })).toBeVisible();
  await expect(page.getByText("WebGPU")).toBeVisible();
  await expect(page.getByText("foundry_local")).toHaveCount(0);
  await expect(page.getByText("WebGpuExecutionProvider")).toHaveCount(0);
  await expect(page.getByText("≥ 60% content recall required")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Manual language review" })).toBeVisible();
  await expect(page.getByText("Tracefold release QA")).toBeVisible();
  await expect(page.getByText("Case-by-case release audit")).toBeVisible();
  await expect(page.getByText("12 Turkish cases")).toBeVisible();
  await expect(page.getByText(`${reviewDigest.slice(0, 12)}…`)).toHaveAttribute("title", reviewDigest);
  await expect(page.getByTitle("2026-07-18T12:00:00+00:00")).toBeVisible();
  await expectNoBlockingA11yViolations(page);

  await page.getByText("Program ne zaman başlıyor?").click();
  await expect(page.getByText("Program 3 Ağustos 2026 tarihinde başlıyor.")).toBeVisible();
  await page.getByRole("button", { name: "Turkish", exact: true }).click();
  await expect(page.getByText("When does the lab begin?")).toBeHidden();

  await page.getByRole("button", { name: "Open preferences" }).click();
  const preferences = page.getByRole("dialog", { name: "Preferences" });
  await preferences.locator("label").filter({ hasText: "Türkçe" }).click();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("status", { name: "Yayın için onaylandı" })).toBeVisible();
  await expect(page.getByText("Temiz")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Çeviri kaydı ve yayın eşikleri" })).toBeVisible();
  await expect(page.getByText("En az %60 içerik kapsaması gerekli")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Ayrıntılı dil incelemesi" })).toBeVisible();
  await expect(page.getByText("Senaryo bazlı yayın denetimi")).toBeVisible();
  await expect(page.getByText("12 Türkçe senaryo")).toBeVisible();
});

test("setup exposes both pinned OPUS-MT checkpoints and the glossary", async ({ page }, testInfo) => {
  const mobile = testInfo.project.name === "mobile-chromium";
  await openNavigation(page, mobile);
  await page.getByRole("button", { name: "Setup", exact: true }).click();

  await expect(page.getByRole("heading", { name: "Offline OPUS-MT translation" })).toBeVisible();
  await expect(page.getByText("Helsinki-NLP/opus-mt-tc-big-tr-en")).toBeVisible();
  await expect(page.getByText("Helsinki-NLP/opus-mt-tc-big-en-tr")).toBeVisible();
  await expect(page.getByText(trToEnRevision.slice(0, 10))).toHaveAttribute("title", trToEnRevision);
  await expect(page.getByText(enToTrRevision.slice(0, 10))).toHaveAttribute("title", enToTrRevision);
  await expect(page.getByText("northstar-summer-lab-en-tr-v1")).toBeVisible();
  await expect(page.getByText("42", { exact: true })).toBeVisible();
  await expect(page.getByText("Checkpoints verified")).toBeVisible();
  await expectNoBlockingA11yViolations(page);

  await page.getByRole("button", { name: "Open preferences" }).click();
  const preferences = page.getByRole("dialog", { name: "Preferences" });
  await preferences.locator("label").filter({ hasText: "Türkçe" }).click();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("heading", { name: "Çevrimdışı OPUS-MT çevirisi" })).toBeVisible();
  await expect(page.getByText("Türkçe → İngilizce")).toBeVisible();
  await expect(page.getByText("Model sürümleri doğrulandı")).toBeVisible();
});

test("composer, evidence dialog, and preferences work with keyboard and touch layouts", async ({ page }, testInfo) => {
  const mobile = testInfo.project.name === "mobile-chromium";
  await expect(page.getByRole("textbox", { name: "Ask your library" })).toBeEnabled();
  await expectNoBlockingA11yViolations(page);
  await page.getByRole("textbox", { name: "Ask your library" }).fill("When does it begin?");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.getByText("The program begins on 3 August 2026.")).toBeVisible();

  const citation = page.getByRole("button", { name: /01-program-handbook\.md/ });
  await citation.click();
  await expect(page.getByRole("dialog", { name: "Source [1]" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(citation).toBeFocused();

  await page.getByRole("button", { name: "Open preferences" }).click();
  const preferences = page.getByRole("dialog", { name: "Preferences" });
  await preferences.locator("label").filter({ hasText: "Türkçe" }).click();
  await expect(page.locator("html")).toHaveAttribute("lang", "tr");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("button", { name: "Tercihleri aç" })).toBeFocused();

  if (mobile) {
    await page.getByRole("button", { name: "Menüyü aç" }).click();
    await expect(page.getByRole("navigation", { name: "Mobil menü" })).toBeVisible();
  }
});
