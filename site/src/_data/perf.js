// Every value here is read from the book's tables. To change a number, change the chapter.
import { chapterUrl } from "../../lib/book.js";
import { barChart, divergingChart, fmt, lineChart } from "../../lib/charts.js";
import { numberAfter, openChapter, toNumber } from "../../lib/tables.js";

const perfFile = "guide/performance.md";
const compareFile = "architecture/comparison.md";
const perf = openChapter(perfFile);
const compare = openChapter(compareFile);

const source = (label, anchor, file = perfFile) => ({ label, url: `${chapterUrl(file)}#${anchor}` });
const tur = (values) => ({ name: "Tur", color: "s1", values });
const llama = (values) => ({ name: "llama.cpp", color: "s2", values });
const percent = (a, b) => (a / b - 1) * 100;

const headline = compare.table("How much faster", "Test");
const headlineRow = (label, name) => {
  const [a, b] = [headline.num(label, "Tur"), headline.num(label, "llama.cpp")];
  return { label: name, tur: a, base: b, percent: percent(a, b) };
};
const m1Single = perf.table("Compared with llama.cpp", "Test");
const dense = perf.table("Qwen3.8-27B", "Test");
const otherRow = (table, label, a, b, name) => {
  const [turValue, base] = [table.num(label, a), table.num(label, b)];
  return { label: name, tur: turValue, base, percent: percent(turValue, base) };
};

const versus = divergingChart({
  id: "versus",
  title: "Tur against llama.cpp",
  note: "Percent difference in tokens per second, on the same Mac. Tur is ahead when requests share a pass, and level or behind on long prefill, which is mostly compute.",
  other: "llama.cpp",
  unit: "tok/s",
  rows: [
    headlineRow("Decode, 8 sequences", "Decode, 8 sequences, M3 Pro"),
    headlineRow("Qwen3.6-35B, 4-bit, burst of 8, end to end", "Qwen3.6-35B 4-bit, burst of 8, M3 Pro"),
    headlineRow("Qwen3.6-35B, 4-bit, decode, 1 sequence", "Qwen3.6-35B 4-bit, decode, 1 sequence, M3 Pro"),
    headlineRow("Decode, 1 sequence, 4-bit", "Decode, 1 sequence, 4-bit, M3 Pro"),
    headlineRow("Qwen3.8-27B, 4-bit, decode, 8 sequences", "Qwen3.8-27B 4-bit, decode, 8 sequences, M3 Pro"),
    headlineRow("Prefill, 2K tokens", "Prefill 2K, M3 Pro"),
    headlineRow("Decode, 1 sequence", "Decode, 1 sequence, M3 Pro"),
    otherRow(dense, "Prefill, 2K", "Tur", "llama.cpp", "Qwen3.8-27B 4-bit, prefill 2K, M3 Pro"),
    otherRow(m1Single, "Prefill 8K", "Tur Q8_0", "llama.cpp Q8_0", "Prefill 8K, M1"),
  ],
  source: source("Tur and other engines", "how-much-faster", compareFile),
});

const batchedChart = (section, title, note, anchor) => {
  const table = perf.table(section, "Sequences");
  return lineChart({
    title,
    note,
    x: table.headers.slice(1),
    xLabel: "sequences",
    unit: "tok/s",
    series: [tur(table.rowNums("Tur")), llama(table.rowNums("llama.cpp"))],
    source: source("Measured performance", anchor),
  });
};

const batchedM3 = batchedChart(
  "On an M3 Pro",
  "Batched decode on an M3 Pro",
  "Qwen3.5-0.8B, Q8_0 weights, total tokens per second. The weights are read once per pass, so sequences added to a pass cost little.",
  "on-an-m3-pro",
);
const batchedM1 = batchedChart(
  "Compared with llama.cpp",
  "Batched decode on a 16 GB M1",
  "Qwen3.5-0.8B, Q8_0 weights, total tokens per second.",
  "compared-with-llamacpp",
);
const batchedDense = batchedChart(
  "Qwen3.8-27B",
  "A 27B dense model on an M3 Pro",
  "Qwen3.8-27B, 4-bit weights, total tokens per second.",
  "qwen38-27b",
);

const moeTable = perf.table("Qwen3.6-35B-A3B", "Concurrent", 1);
// Sixteen requests compare different configurations, so the chart stops at eight and says why.
const moeRows = moeTable.rows.filter((row) => /^\d+(\.\d+)?$/.test(row[1]));
const burstMoe = lineChart({
  title: "A 35B mixture-of-experts model, end to end",
  note: "Qwen3.6-35B-A3B, 4-bit weights, bursts of 128-token prompts and 64 output tokens. At 16 requests llama.cpp finishes with every weight resident, while Tur runs out of GPU memory unless it uses its expert cache, so that row isn't plotted.",
  x: moeRows.map((row) => row[0]),
  xLabel: "concurrent requests",
  unit: "output tok/s",
  series: [tur(moeRows.map((row) => toNumber(row[1]))), llama(moeRows.map((row) => toNumber(row[2])))],
  source: source("Measured performance", "qwen36-35b-a3b"),
});

const moeSection = "Qwen3.6-35B-A3B";
const moeCases = perf.table(moeSection, "Case");
const moeDecode = lineChart({
  title: "Decode as sequences share a pass",
  note: "Qwen3.6-35B-A3B, 4-bit weights, at 128 tokens of context, total tokens per second.",
  x: ["1", "4", "8"],
  xLabel: "sequences",
  unit: "tok/s",
  series: [
    tur([
      moeCases.num("Decode, 1 sequence at 128 tokens", "Result"),
      moeCases.num("Decode, 4 sequences at 128 tokens", "Result"),
      moeCases.num("Decode, 8 sequences at 128 tokens", "Result"),
    ]),
  ],
  source: source("Measured performance", "qwen36-35b-a3b"),
});

const moeEngines = perf.table(moeSection, "Engine, weights");
const engineBars = (column, title, note) =>
  barChart({
    title,
    note,
    unit: "tok/s",
    series: [{ name: "", color: "s1" }],
    groups: moeEngines.firstColumn().map((label) => ({ label, values: [moeEngines.num(label, column)] })),
    source: source("Measured performance", "qwen36-35b-a3b"),
  });
const moeEngineDecode = engineBars(
  "Decode, 1 sequence",
  "Decode against other engines",
  "One sequence on an M3 Pro, with each engine's usual 4-bit weights. Tur and llama.cpp at 128 tokens of context. The oMLX row is a reported result at 1K, not a rerun here.",
);
const moeEnginePrefill = engineBars(
  "Prefill",
  "Prefill against other engines",
  "Tur and llama.cpp at 512 tokens, where llama.cpp is slightly ahead. The oMLX row is a reported result at 1K, not a rerun here.",
);

const moeSixteen = moeTable.rows.find((row) => row[0] === "16");

const omlxTable = perf.table("Compared with MLX and oMLX", "Sequences");
const omlx = lineChart({
  title: "Against oMLX, 4-bit",
  note: "Qwen3.5-0.8B on a 16 GB M1. oMLX's figures are its published results, not a rerun here. From two sequences on the lead disappears.",
  x: omlxTable.headers.slice(1),
  xLabel: "sequences",
  unit: "tok/s",
  series: [
    tur(omlxTable.rowNums("Tur Q4K GPTQ")),
    { name: "oMLX", color: "s3", values: omlxTable.rowNums("oMLX OptiQ-4bit") },
  ],
  source: source("Measured performance", "compared-with-mlx-and-omlx"),
});

const serving = perf.table("Serving", "Concurrent");
const servingBars = (column, title, note, unit) =>
  barChart({
    title,
    note,
    unit,
    series: [{ name: "", color: "s1" }],
    groups: serving.firstColumn().map((label) => ({
      label: `${label} at once`,
      values: [serving.num(label, column)],
    })),
    source: source("Measured performance", "serving"),
  });
const servingOutput = servingBars(
  "Output tok/s",
  "Output through the full server",
  "16 GB M1, Qwen3.5-0.8B, Q8_0, sampled, total across the burst.",
  "tok/s",
);
const servingGap = servingBars(
  "ITL p50",
  "Gap between tokens",
  "Median inter-token latency for one request in the same bursts. Lower is better.",
  "ms",
);

const reuse = perf.table("Prefix reuse", "Workload");
const reuseCell = (label) => numberAfter(reuse.cell(label, "Result"), "TTFT p50");
const coldLabel = "The same 8 users with prefix reuse off";
const cachedLabel = "8 users behind an already cached 1.5K-token prompt";
const freshLabel = "8 users with the same new 1.5K-token prompt at once";
const prefix = barChart({
  title: "First token behind a shared 1.5K-token prompt",
  note: "Eight users on a 16 GB M1, median time to first token. Lower is better.",
  unit: "s",
  series: [{ name: "", color: "s1" }],
  groups: [
    { label: "Prefix reuse off", values: [reuseCell(coldLabel)] },
    { label: "New prompt, arriving together", values: [reuseCell(freshLabel)] },
    { label: "Prompt already cached", values: [reuseCell(cachedLabel)] },
  ],
  source: source("Measured performance", "prefix-reuse"),
});

const speculationTable = perf.table("Speculative decoding", "Prompt");
const speculationRows = speculationTable.rows.filter((row) => row[1] === "1");
const speculation = barChart({
  title: "Speculative decoding, one sequence",
  note: "Greedy, 16 GB M1, Qwen3.5-0.8B. Output matched normal decoding in every run.",
  unit: "tok/s",
  series: [
    { name: "Normal", color: "s-muted" },
    { name: "Prompt lookup", color: "s2" },
    { name: "Lookup + MTP head", color: "s1" },
  ],
  groups: speculationRows.map((row) => ({
    label: row[0],
    values: [toNumber(row[2]), toNumber(row[3]), toNumber(row[4])],
  })),
  source: source("Measured performance", "speculative-decoding"),
});

const loading = perf.table("Loading", "Weights");
const loadBars = (column, title, note, unit) =>
  barChart({
    title,
    note,
    unit,
    series: [{ name: "", color: "s1" }],
    groups: loading.firstColumn().map((label) => ({ label, values: [loading.num(label, column)] })),
    source: source("Measured performance", "loading"),
  });
const loadTime = loadBars("Load time", "Time to load", "A fresh process with a warm file cache, 16 GB M1. Lower is better.", "s");
const loadMemory = loadBars("Peak RSS", "Peak memory while loading", "The same runs, from /usr/bin/time. Lower is better.", "GB");

const quality = perf.table("Quality of weight formats", "Weights");
const formats = barChart({
  title: "What smaller weights cost in quality",
  note: "How often a format's most likely token matches F32 on held-out prose. Calibrated GPTQ loses less than plain rounding at the same size.",
  unit: "%",
  series: [{ name: "", color: "s1" }],
  groups: quality.firstColumn().map((label) => {
    const value = numberAfter(quality.cell(label, "Prose KL / top-1"), "/");
    return { label, values: [value], display: [`${value}%`] };
  }),
  source: source("Measured performance", "quality-of-weight-formats"),
});

const eight = headline.num("Decode, 8 sequences", "Tur");
const eightBase = headline.num("Decode, 8 sequences", "llama.cpp");
const repetitive = speculationRows.find((row) => row[0] === "Repetitive");

const stats = {
  batched: Math.round(percent(eight, eightBase)),
  batchedTok: fmt(eight),
  batchedBase: fmt(eightBase),
  prefixCached: `${reuseCell(cachedLabel)} s`,
  prefixCold: `${reuseCell(coldLabel)} s`,
  speculation: `${(toNumber(repetitive[3]) / toNumber(repetitive[2])).toFixed(1)}×`,
  load: `${loading.num("Q4K GPTQ artifact", "Load time")} s`,
};

const servingScenario = (name, section, nth = 0) => {
  const table = perf.table(section, "Concurrent", nth);
  const label = (row) => row[0];
  const column = (title) => table.rows.map((row) => toNumber(row[table.headers.indexOf(title)]));
  return {
    name,
    requests: table.rows.map(label),
    metrics: [
      { label: "Output", unit: "tok/s", best: "high", values: column("Output tok/s") },
      { label: "First token, median", unit: "s", best: "low", values: column("TTFT p50") },
      { label: "Gap between tokens, median", unit: "ms", best: "low", values: column("ITL p50") },
      { label: "Gap between tokens, p95", unit: "ms", best: "low", values: column("ITL p95") },
      { label: "Gap between tokens, p99", unit: "ms", best: "low", values: column("ITL p99") },
    ],
  };
};

const raceScenario = (name, section) => {
  const table = perf.table(section, "Sequences");
  return {
    name,
    sequences: table.headers.slice(1).map(Number),
    tur: table.rowNums("Tur"),
    llama: table.rowNums("llama.cpp"),
  };
};

// The chapter leaves the prompt cell blank on continuation rows.
const speculationModes = (nth) => {
  let prompt = "";
  return perf.table("Speculative decoding", "Prompt", nth).rows.map((row) => {
    prompt = row[0] || prompt;
    return { prompt, sequences: row[1], values: row.slice(2).map((cell) => toNumber(cell)) };
  });
};

const singlePass = perf.table("Single model pass", "Weights");
const formatOption = (name, artifact, qualityRow) => {
  const top1 = (column) => numberAfter(quality.cell(qualityRow, column), "/");
  return {
    name,
    load: loading.num(artifact, "Load time"),
    memory: loading.num(artifact, "Weight memory"),
    peak: loading.num(artifact, "Peak RSS"),
    decode1: singlePass.num(artifact, "1 seq"),
    decode8: singlePass.num(artifact, "8 seq"),
    quality: [
      ["Prose", top1("Prose KL / top-1")],
      ["Code", top1("Code KL / top-1")],
      ["Retrieval", top1("Retrieval KL / top-1")],
      ["Tools", top1("Tools KL / top-1")],
    ],
  };
};

const lab = {
  race: [
    raceScenario("Qwen3.5-0.8B on an M3 Pro", "On an M3 Pro"),
    raceScenario("Qwen3.5-0.8B on a 16 GB M1", "Compared with llama.cpp"),
    raceScenario("Qwen3.8-27B on an M3 Pro", "Qwen3.8-27B"),
    {
      name: "Qwen3.6-35B on an M3 Pro",
      sequences: moeRows.map((row) => Number(row[0])),
      tur: moeRows.map((row) => toNumber(row[1])),
      llama: moeRows.map((row) => toNumber(row[2])),
    },
  ],
  serving: [
    servingScenario("Qwen3.5-0.8B on a 16 GB M1", "Serving"),
    servingScenario("Qwen3.6-35B on an M3 Pro", "Qwen3.6-35B-A3B"),
    servingScenario("Qwen3.8-27B on an M3 Pro", "Qwen3.8-27B"),
  ],
  speculation: { Greedy: speculationModes(0), Sampled: speculationModes(1) },
  moeSpeculation: perf.table(moeSection, "Requests").rows.map((row) => ({
    requests: row[0],
    prompt: row[1],
    normal: toNumber(row[2]),
    mtp: toNumber(row[3]),
    speedup: row[4],
  })),
  formats: [
    formatOption("Q4K GPTQ", "Q4K GPTQ artifact", "Q4K GPTQ"),
    formatOption("Mixed 5.5-bit", "Mixed 5.5-bit GPTQ artifact", "Mixed 5.5-bit GPTQ"),
    formatOption("Q8_0", "Q8_0 artifact", "Q8_0"),
  ],
  chooser: compare.table("", "You want").rows.map(([need, engine]) => ({ need, engine })),
};

export default {
  lab,
  stats,
  moe: {
    load: moeCases.num("Load time", "Result"),
    sixteenTur: toNumber(moeSixteen[1]),
    sixteenLlama: toNumber(moeSixteen[2]),
  },
  charts: {
    moeDecode,
    moeEngineDecode,
    moeEnginePrefill,
    versus,
    batchedM3,
    batchedM1,
    batchedDense,
    burstMoe,
    omlx,
    servingOutput,
    servingGap,
    prefix,
    speculation,
    loadTime,
    loadMemory,
    formats,
  },
};
