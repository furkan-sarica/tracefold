import { afterEach, expect, test, vi } from "vitest";
import { getLatestEvaluation, getRuntime, streamChat } from "./api";

function response(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => vi.unstubAllGlobals());

test("normalizes nested runtime models", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response({
    ready: true,
    chat_model: { alias: "chat-alias", variant_id: "chat-variant", execution_provider: "WebGPU", device: "GPU" },
    embedding_model: { alias: "embed-alias", variant_id: "embed-variant" },
    translation: {
      runtime: "marian_opus",
      ready: true,
      offline_only: true,
      models: {
        tr_to_en: { id: "Helsinki-NLP/opus-mt-tc-big-tr-en", revision: "2261c8fc7b1af59caee87f8ff0ecf3fbccfe8391", cached: true },
        en_to_tr: { id: "Helsinki-NLP/opus-mt-tc-big-en-tr", revision: "e539fc16a8a1a0ea5950eb339b595bfcce990e90", cached: true },
      },
      glossary: { id: "northstar-summer-lab-en-tr-v1", entries: 42 },
      round_trip_threshold: 0.6,
    },
  })));

  await expect(getRuntime()).resolves.toMatchObject({
    chat_model: "chat-alias",
    embedding_model: "embed-alias",
    chat_variant: "chat-variant",
    embedding_variant: "embed-variant",
    execution_provider: "WebGPU",
    device: "GPU",
    translation: expect.objectContaining({
      runtime: "marian_opus",
      ready: true,
      glossary: { id: "northstar-summer-lab-en-tr-v1", entries: 42 },
      round_trip_threshold: 0.6,
    }),
  });
});

test("normalizes evaluation metrics without discarding bilingual run data", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response({
    id: "eval-1",
    metrics: { total_cases: 24, passed_cases: 23, hit_at_k: 0.95, locale_breakdown: { tr: { total_cases: 12, passed_cases: 11 } } },
    cases: [{ id: "TR-1", locale: "tr", passed: false }],
  })));

  const result = await getLatestEvaluation();
  expect(result).toMatchObject({ total_cases: 24, passed_cases: 23, hit_at_5: 0.95 });
  expect((result as unknown as { metrics: { locale_breakdown: unknown } }).metrics.locale_breakdown).toBeTruthy();
});

test("sends the selected locale and handles a JSON chat response", async () => {
  const fetchMock = vi.fn().mockResolvedValue(response({ answer: "Kanıtlı yanıt", citations: [{ filename: "guide.md", text: "Kanıt" }] }));
  vi.stubGlobal("fetch", fetchMock);
  const onToken = vi.fn();
  const onCitations = vi.fn();

  await streamChat("Soru", 5, "tr", { onToken, onCitations });

  expect(JSON.parse(String((fetchMock.mock.calls[0][1] as RequestInit).body))).toMatchObject({ question: "Soru", top_k: 5, locale: "tr" });
  expect(onToken).toHaveBeenCalledWith("Kanıtlı yanıt");
  expect(onCitations).toHaveBeenCalledWith([expect.objectContaining({ source: "guide.md", excerpt: "Kanıt" })]);
});
