/** Node-only DOM smoke test. No browser, network, screenshot, or visual QA. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const html = fs.readFileSync(path.join(root, "src/gig/service/static/index.html"), "utf8");
const source = fs.readFileSync(path.join(root, "src/gig/service/static/app.js"), "utf8");
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
assert.equal(new Set(ids).size, ids.length, "HTML IDs must be unique");
for (const match of html.matchAll(/\bfor="([^"]+)"/g)) assert(ids.includes(match[1]), `Missing label target ${match[1]}`);
for (const match of source.matchAll(/\$\("([^"]+)"\)/g)) assert(ids.includes(match[1]), `Missing DOM reference ${match[1]}`);
assert(!/\b(?:innerHTML|outerHTML|eval)\b/.test(source), "Provider content must not use HTML execution sinks");
assert(!/https?:\/\//.test(html), "No external asset fetches in the terminal shell");

let clickedDownloads = [];
class Element {
  constructor(tagName = "div", id = "") {
    this.tagName = tagName;
    this.id = id;
    this.children = [];
    this.listeners = new Map();
    this.attributes = new Map();
    this.dataset = {};
    this.style = { setProperty(name, value) { this[name] = value; } };
    this.value = "";
    this.disabled = false;
    this._text = "";
    this.classes = new Set();
    this.classList = {
      add: (...names) => names.forEach((name) => this.classes.add(name)),
      remove: (...names) => names.forEach((name) => this.classes.delete(name)),
      contains: (name) => this.classes.has(name),
      toggle: (name, force) => {
        const add = force === undefined ? !this.classes.has(name) : force;
        if (add) this.classes.add(name); else this.classes.delete(name);
        return add;
      },
    };
  }
  get className() { return [...this.classes].join(" "); }
  set className(value) { this.classes = new Set(String(value).split(/\s+/).filter(Boolean)); }
  get textContent() { return this._text + this.children.map((child) => typeof child === "string" ? child : child.textContent).join(""); }
  set textContent(value) { this._text = String(value); this.children = []; }
  append(...nodes) {
    for (const node of nodes) {
      if (node?.tagName === "#fragment") this.children.push(...node.children);
      else this.children.push(node);
    }
  }
  replaceChildren(...nodes) { this.children = []; this._text = ""; this.append(...nodes); }
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
    if (name.startsWith("data-")) this.dataset[name.slice(5).replace(/-([a-z])/g, (_, char) => char.toUpperCase())] = String(value);
  }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  removeAttribute(name) {
    this.attributes.delete(name);
    if (name.startsWith("data-")) delete this.dataset[name.slice(5).replace(/-([a-z])/g, (_, char) => char.toUpperCase())];
  }
  addEventListener(name, callback) {
    if (!this.listeners.has(name)) this.listeners.set(name, []);
    this.listeners.get(name).push(callback);
  }
  async dispatch(name, detail = {}) {
    const event = { preventDefault() {}, ...detail };
    await Promise.all((this.listeners.get(name) || []).map((callback) => callback(event)));
  }
  get hash() { return String(this.href || "").startsWith("#") ? this.href : ""; }
  querySelector(selector) {
    const found = this.children.find((node) => node.tagName === selector);
    assert(found, `Missing query target ${selector}`);
    return found;
  }
  click() {
    if (this.download) clickedDownloads.push(this);
    return this.dispatch("click");
  }
  getBoundingClientRect() { return { width: 800, height: 340, left: 0, top: 0 }; }
  getContext() { return drawingContext; }
  setPointerCapture() {}
}
let drawnPoints = 0;
const drawingContext = new Proxy({}, { get: (_, name) => name === "arc" ? () => { drawnPoints++; } : () => {}, set: () => true });
const elements = new Map(ids.map((id) => [id, new Element("div", id)]));
for (const match of html.matchAll(/<[^>]*\bid="([^"]+)"[^>]*>/g)) {
  const classes = /\bclass="([^"]*)"/.exec(match[0]);
  if (classes) elements.get(match[1]).className = classes[1];
}
elements.get("analytics-kind").value = "options";
elements.get("data-mode").value = "local";
elements.get("analysis-engine").value = "default";
elements.get("freshness-policy").value = "86400";
elements.get("observations-table").append(new Element("thead"), new Element("tbody"));
const documentRoot = new Element("html");
const walk = (node) => [node, ...node.children.filter((child) => child instanceof Element).flatMap(walk)];
const document = {
  title: "",
  documentElement: documentRoot,
  getElementById(id) { assert(elements.has(id), `Unknown element ${id}`); return elements.get(id); },
  createElement: (tag) => new Element(tag),
  createElementNS: (_, tag) => new Element(tag),
  createDocumentFragment: () => new Element("#fragment"),
  querySelectorAll(selector) {
    assert.equal(selector, ".nav-link");
    return walk(elements.get("section-nav")).filter((node) => node.classList.contains("nav-link"));
  },
};
const windowEvents = new Map();
const window = {
  devicePixelRatio: 1,
  addEventListener(name, callback) { windowEvents.set(name, callback); },
};
const location = { hash: "" };
let replyOverride = null;
const calls = [];
const blobs = [];
const localUnavailable = () => ({
  schema_version: "1", state: "UNAVAILABLE", data: null,
  provenance: { source: "local", classification: "UNAVAILABLE", as_of: null },
  error: { code: "DATA_UNAVAILABLE", message: "No admitted data is available." }, scope: "terminal.test", content_sha256: "0".repeat(64),
});
function fixture(overrides = {}) {
  return {
    schema_version: "1", state: "AVAILABLE", error: null, scope: "terminal.test", content_sha256: "0".repeat(64),
    provenance: { source: "deterministic-test", classification: "SYNTHETIC", as_of: "2026-09-30T20:00:00.123456789+00:00", readiness: "RESEARCH_ONLY" },
    data: {
      title: "Test analysis", description: "Authored synthetic fixture",
      metrics: { zero: 0, missing: null, identity: { $integer: "9007199254740993" }, precise: { $decimal: "1.00000000000000000001" }, negative_zero: -0 },
      rows: [
        { symbol: "SYN002", identity: { $integer: "9007199254740993" }, value: 2, note: "=HYPERLINK(\"example\")" },
        { symbol: "SYN001", identity: { $integer: "9007199254740992" }, value: 1, note: "<img src=x onerror=alert(1)>" },
        { symbol: "SYN003", identity: null, value: null, note: "Original NaN research source text" },
      ],
      chart: { kind: "factor-space", points: [{ symbol: "SYN001", x: 0, y: 1, z: 2 }, { symbol: "SYN002", x: 1, y: 2, z: 3 }] },
    }, ...overrides,
  };
}
const context = vm.createContext({
  document, window, location, AbortController, console, Intl, Date, Blob, BigInt,
  getComputedStyle: () => ({ getPropertyValue: () => "#777777" }),
  localStorage: { getItem: () => null, setItem() {} },
  setTimeout: () => 1, clearTimeout() {}, setInterval: () => 1,
  URL: { createObjectURL(blob) { blobs.push(blob); return "blob:terminal-test"; }, revokeObjectURL() {} },
  async fetch(url, options = {}) {
    calls.push({ url, options });
    if (replyOverride instanceof Error) throw replyOverride;
    const provided = typeof replyOverride === "function" ? replyOverride(url, options) : replyOverride;
    const body = provided?.body ?? (url.includes("mode=local") ? localUnavailable() : fixture());
    return { ok: provided?.ok ?? true, status: provided?.status ?? 200, json: async () => structuredClone(body) };
  },
});
vm.runInContext(source, context, { filename: "app.js" });
const settle = async () => { for (let pass = 0; pass < 4; pass++) await new Promise(setImmediate); };
const get = (id) => elements.get(id);
const tbody = () => get("observations-table").querySelector("tbody");
const headerButtons = () => get("observations-table").querySelector("thead").children[0].children.map((th) => th.children[0]);
const sections = ["command-center", "markets", "signals", "portfolio", "risk", "models", "news", "stat-arb", "options", "execution", "research", "operations"];
let checks = 0;
function checked(label, callback) { callback(); checks++; console.log(`PASS ${label}`); }

await settle();
checked("initial request uses local mode and renders explicit unavailable", () => {
  assert.equal(calls[0].url, "/api/v1/terminal/command-center?mode=local&max_age_seconds=86400");
  assert.equal(get("state-badge").textContent, "UNAVAILABLE");
  assert.equal(tbody().children.length, 0);
  assert.match(get("error-panel").textContent, /DATA_UNAVAILABLE/);
  assert(!get("error-panel").classList.contains("hidden"));
  assert.equal(document.querySelectorAll(".nav-link").length, 12);
});
for (const section of sections) {
  location.hash = `#${section}`;
  await windowEvents.get("hashchange")(); await settle();
  assert.equal(calls.at(-1).url, `/api/v1/terminal/${section}?mode=local&max_age_seconds=86400`);
  assert.equal(get("state-badge").textContent, "UNAVAILABLE");
}
checked("all twelve sections request typed backend paths", () => assert.equal(document.querySelectorAll(".nav-link").filter((link) => link.getAttribute("aria-current") === "page").length, 1));
get("freshness-policy").value = "3600";
await get("freshness-policy").dispatch("change"); await settle();
checked("local queries use the visible explicit freshness policy", () => assert.match(calls.at(-1).url, /max_age_seconds=3600$/));
const replayCutoff = "2026-09-28T20:00:00.123456789+00:00";
get("replay-asof").value = replayCutoff;
await get("replay-asof").dispatch("change"); await settle();
checked("point-in-time input preserves original timezone and nanosecond precision", () => {
  assert(calls.at(-1).url.endsWith(`&as_of=${encodeURIComponent(replayCutoff)}`));
  assert.equal(get("replay-asof").value, replayCutoff);
});
get("freshness-policy").value = "86400";
get("replay-asof").value = "";
get("data-mode").value = "synthetic";
await get("data-mode").dispatch("change"); await settle();
checked("synthetic mode requires explicit selection and shows provenance", () => {
  assert.match(calls.at(-1).url, /mode=synthetic$/);
  assert.match(get("truth-notice").textContent, /SYNTHETIC INPUTS/);
  assert.match(get("source-label").textContent, /deterministic-test · SYNTHETIC/);
  assert.equal(get("asof-label").textContent, "As of 2026-09-30T20:00:00.123456789+00:00");
});
for (const section of sections) {
  location.hash = `#${section}`;
  await windowEvents.get("hashchange")(); await settle();
  assert.equal(calls.at(-1).url, `/api/v1/terminal/${section}?mode=synthetic`);
  assert.equal(get("state-badge").textContent, "AVAILABLE");
}
checked("all twelve sections retain explicitly selected synthetic mode", () => assert.equal(get("data-mode").value, "synthetic"));
checked("zero, unknown, signed zero and exact scalar tags remain distinct", () => {
  const cards = get("metrics").children.map((card) => card.children[1].textContent);
  assert.deepEqual(cards, ["0", "Unknown", "9007199254740993", "1.00000000000000000001", "−0"]);
  assert.equal(tbody().children[0].children[1].textContent, "9007199254740993");
});
checked("provider text is rendered as text without HTML injection", () => {
  assert(tbody().textContent.includes("<img src=x onerror=alert(1)>"));
  const noteCell = tbody().children[1].children[3];
  assert.equal(noteCell.children.length, 0);
  assert.equal(noteCell.textContent, "<img src=x onerror=alert(1)>");
});
await headerButtons()[1].dispatch("click");
checked("large identity integers sort without binary64 rounding", () => assert.equal(tbody().children[0].children[1].textContent, "9007199254740992"));
await headerButtons()[1].dispatch("click");
checked("descending sort keeps unknown identities last", () => {
  assert.equal(tbody().children[0].children[1].textContent, "9007199254740993");
  assert.equal(tbody().children.at(-1).children[1].textContent, "Unknown");
});
get("table-search").value = "NaN research";
await get("table-search").dispatch("input");
checked("table filter preserves source text and matches observations", () => {
  assert.equal(tbody().children.length, 1);
  assert.equal(tbody().children[0].children[0].textContent, "SYN003");
});
get("table-search").value = ""; await get("table-search").dispatch("input");
await get("export-csv").dispatch("click");
const csv = await blobs.at(-1).text();
checked("CSV exports visible rows, formula-safe strings and exact identities as text", () => {
  assert.match(csv, /"'9007199254740993"/);
  assert.match(csv, /"'=HYPERLINK/);
  assert.match(csv, /Original NaN research source text/);
  assert.equal(clickedDownloads.at(-1).download, "gig-operations-synthetic.csv");
});
checked("3D chart draws finite observations", () => assert(drawnPoints > 0));
await get("view-2d").dispatch("click");
checked("2D alternative retains axis explanation and access state", () => {
  assert.match(get("chart-legend").textContent, /2D projection/);
  assert.equal(get("view-2d").getAttribute("aria-pressed"), "true");
  assert.equal(get("view-3d").getAttribute("aria-pressed"), "false");
});
await get("analytics-chart").dispatch("keydown", { key: "ArrowLeft" });
await get("theme-toggle").dispatch("click");
checked("keyboard chart navigation and light theme execute", () => assert.equal(documentRoot.dataset.theme, "light"));
const numericRows = [
  { label: "tiny", amount: { $decimal: "1E-1000000000" } },
  { label: "zero", amount: 0 },
  { label: "negative", amount: { $integer: "-9007199254740993" } },
  { label: "precise", amount: { $decimal: "1.00000000000000000001" } },
  { label: "one", amount: 1 },
  { label: "huge", amount: { $decimal: "1E+1000000000" } },
];
replyOverride = { body: fixture({ data: { metrics: {}, rows: numericRows } }) };
await get("refresh").dispatch("click"); await settle();
await headerButtons()[1].dispatch("click");
checked("mixed exact decimal/integer sorting handles subnormal-scale and huge exponents", () => {
  assert.deepEqual(tbody().children.map((row) => row.children[0].textContent), ["negative", "zero", "tiny", "one", "precise", "huge"]);
});
replyOverride = { body: fixture({ state: "STALE" }) };
await get("refresh").dispatch("click"); await settle();
checked("stale snapshots retain timestamp and visible stale disclosure", () => {
  assert.equal(get("state-badge").dataset.state, "STALE");
  assert.match(get("truth-notice").textContent, /freshness requirement is not satisfied/);
  assert.match(get("asof-label").textContent, /123456789/);
});
replyOverride = { body: fixture({ state: "UNKNOWN", data: null }) };
await get("refresh").dispatch("click"); await settle();
checked("unknown is not displayed as empty success or zero", () => {
  assert.equal(get("state-badge").textContent, "UNKNOWN");
  assert.equal(get("metrics").children.length, 0);
  assert.equal(tbody().children.length, 0);
  assert(!get("error-panel").classList.contains("hidden"));
});
replyOverride = { ok: false, status: 422, body: fixture({ state: "ERROR", data: null, error: { code: "INVALID_REQUEST", message: "The request does not satisfy the API contract." } }) };
await get("refresh").dispatch("click"); await settle();
checked("HTTP error preserves the public typed backend error", () => {
  assert.equal(get("state-badge").textContent, "ERROR");
  assert.match(get("error-panel").textContent, /^INVALID_REQUEST:/);
});
replyOverride = { body: fixture({ state: "ERROR", error: { code: "BOGUS", message: "bad" } }) };
await get("refresh").dispatch("click"); await settle();
checked("invalid ERROR-with-data envelope cannot render apparent success", () => {
  assert.equal(get("state-badge").textContent, "ERROR");
  assert.equal(get("metrics").children.length, 0);
  assert.equal(tbody().children.length, 0);
  assert.match(get("error-panel").textContent, /^REQUEST_FAILED:/);
});
for (const invalid of [fixture({ schema_version: "999" }), fixture({ state: "UNKNOWN" }), fixture({ state: "STALE", provenance: { source: "test", classification: "HISTORICAL", as_of: null } })]) {
  replyOverride = { body: invalid };
  await get("refresh").dispatch("click"); await settle();
  assert.equal(get("state-badge").textContent, "ERROR");
  assert.equal(tbody().children.length, 0);
}
checked("wrong schema, unknown-with-data and stale-without-time are rejected", () => assert.equal(get("metrics").children.length, 0));
replyOverride = new Error("mock transport failure with secret-shaped text");
await get("refresh").dispatch("click"); await settle();
checked("network failures expose a generic error and clear prior data", () => {
  assert.equal(get("state-badge").textContent, "ERROR");
  assert(!get("error-panel").textContent.includes("secret-shaped"));
  assert.equal(get("metrics").children.length, 0);
});
replyOverride = { body: fixture() };
get("analytics-kind").value = "options";
const rawSource = ' {"spot":100,"strike":100,"identity":9007199254740993,"source_text":" Original NaN  ","spot":101} ';
get("analytics-input").value = rawSource;
await get("run-analysis").dispatch("click"); await settle();
checked("workbench sends original JSON text including duplicate keys and exact integer lexemes", () => {
  assert.equal(calls.at(-1).url, "/api/v1/analytics/options");
  assert.equal(calls.at(-1).options.body, rawSource);
  assert.equal(calls.at(-1).options.method, "POST");
});
const before = calls.length;
get("analytics-input").value = "not JSON";
await get("run-analysis").dispatch("click");
checked("invalid workbench JSON is rejected before transport", () => {
  assert.equal(calls.length, before);
  assert.match(get("analytics-output").textContent, /No request was sent/);
});
replyOverride = { body: fixture() };
const submittedWorkspaceExamples = {};
for (const action of ["bars", "news", "research", "simulate"]) {
  get("analytics-kind").value = `workspace-${action}`;
  const priorCalls = calls.length;
  await get("analytics-kind").dispatch("change"); await settle();
  assert.equal(calls.length, priorCalls, "Loading authored examples must not admit data automatically");
  const example = get("analytics-input").value;
  const parsed = JSON.parse(example);
  submittedWorkspaceExamples[action] = parsed;
  assert.match(example, /synthetic/i);
  if (action === "bars" || action === "research") assert.equal(parsed.classification, "SYNTHETIC");
  if (action === "simulate") assert.equal(parsed.classification, "SYNTHETIC");
  await get("run-analysis").dispatch("click"); await settle();
  assert.equal(calls.at(-1).url, `/api/v1/workspace/${action}`);
  assert.equal(calls.at(-1).options.method, "POST");
  assert.equal(calls.at(-1).options.headers["Content-Type"], "application/json");
  assert.equal(calls.at(-1).options.body, example);
}
for (const action of ["ingest", "universe", "train-sequence", "predict-sequence", "finbert", "execution-engine"]) {
  get("analytics-kind").value = `workspace-${action}`;
  const previous = calls.length;
  await get("analytics-kind").dispatch("change"); await settle();
  assert.equal(calls.length, previous, "Example selection must not perform actions");
  const example = get("analytics-input").value;
  assert.equal(typeof JSON.parse(example), "object");
  await get("run-analysis").dispatch("click"); await settle();
  assert.equal(calls.at(-1).url, `/api/v1/workspace/${action}`);
  assert.equal(calls.at(-1).options.body, example);
  checked(`explicit ${action} fixture uses its backend contract without automatic submission`, () => {
    assert.match(get("workbench-scope").textContent, /SYNTHETIC|research/);
  });
}
get("analytics-kind").value = "workspace-treasury";
const beforeTreasury = calls.length;
await get("analytics-kind").dispatch("change"); await settle();
checked("public Treasury selection declares acquisition without fetching automatically", () => {
  assert.equal(calls.length, beforeTreasury);
  assert.equal(JSON.parse(get("analytics-input").value).mode, "PUBLIC_OBSERVED");
  assert.match(get("workbench-scope").textContent, /fixed official Treasury/);
});
await get("run-analysis").dispatch("click"); await settle();
checked("explicit Treasury submission uses the bounded provider contract", () => {
  assert.equal(calls.at(-1).url, "/api/v1/workspace/ingest");
  assert.deepEqual(JSON.parse(calls.at(-1).options.body), { provider: "us_treasury", mode: "PUBLIC_OBSERVED", year: 2025 });
});
get("analytics-kind").value = "workspace-bars";
await get("analytics-kind").dispatch("change");
checked("all four explicit workspace actions use distinct JSON endpoints without automatic admission", () => {
  assert.match(get("workbench-scope").textContent, /writes caller-supplied evidence/);
  assert.match(get("analytics-output").textContent, /no broker orders/);
  assert.equal(get("data-mode").value, "synthetic", "A saved fixture must not silently switch environments");
});
replyOverride = { ok: false, status: 422, body: fixture({ state: "ERROR", data: null, error: { code: "INVALID_REQUEST", message: "The request does not satisfy the API contract." } }) };
get("analytics-kind").value = "workspace-bars";
const rawAdmission = ' {"rows":[],"volume":9007199254740993,"source_text":" Original NaN ","source":"one","source":"two"} ';
get("analytics-input").value = rawAdmission;
const beforeRejected = calls.length;
await get("run-analysis").dispatch("click"); await settle();
checked("rejected workspace admission preserves exact raw evidence and cannot render successful save", () => {
  assert.equal(calls.length, beforeRejected + 1, "Failed admission must not trigger a follow-up success refresh");
  assert.equal(calls.at(-1).options.body, rawAdmission);
  assert.match(get("analytics-output").textContent, /^HTTP 422\nERROR/);
  assert(!get("analytics-output").textContent.includes("SAVED"));
  assert.equal(get("run-analysis").disabled, false);
});
replyOverride = { body: fixture({ provenance: { source: "caller-asserted-source", classification: "HISTORICAL", as_of: replayCutoff, evidence_state: "CALLER_SUPPLIED_NOT_VENDOR_VERIFIED" } }) };
await get("refresh").dispatch("click"); await settle();
checked("persisted caller source claims never become vendor authenticity", () => assert.match(get("truth-notice").textContent, /Caller-supplied provenance · Vendor authenticity unverified/));
replyOverride = { body: fixture() };
for (const kind of ["calendar", "corporate-actions"]) {
  get("analytics-kind").value = kind;
  const prior = calls.length;
  await get("analytics-kind").dispatch("change"); await settle();
  assert.equal(calls.length, prior);
  const body = get("analytics-input").value;
  await get("run-analysis").dispatch("click"); await settle();
  checked(`${kind} research uses explicit calculation without admitting market data`, () => {
    assert.equal(calls.at(-1).url, `/api/v1/analytics/${kind}`);
    assert.equal(calls.at(-1).options.body, body);
    assert(!get("workbench-scope").textContent.includes("writes caller-supplied evidence"));
  });
}
get("data-mode").value = "local";
await get("data-mode").dispatch("change"); await settle();
for (const engine of ["sequence", "finbert", "provider", "execution-engine", "persisted", "accounting", "allocation"]) {
  get("analysis-engine").value = engine;
  await get("analysis-engine").dispatch("change"); await settle();
  assert.match(calls.at(-1).url, new RegExp(`engine=${engine}`));
}
checked("explicit local evidence views preserve engine selection in backend requests", () => assert.equal(get("analysis-engine").disabled, false));
get("analysis-engine").value = "provider";
get("universe-id").value = "authored-synthetic-universe";
await get("universe-id").dispatch("change"); await settle();
checked("point-in-time universe selection uses explicit saved identity", () => assert.match(calls.at(-1).url, /engine=provider&universe_id=authored-synthetic-universe/));
get("data-mode").value = "synthetic";
await get("data-mode").dispatch("change"); await settle();
checked("synthetic demonstrations cannot impersonate a selected local model engine", () => {
  assert.equal(get("analysis-engine").disabled, true);
  assert.equal(get("analysis-engine").value, "default");
  assert(!calls.at(-1).url.includes("engine="));
});
const pipelineActions = {
  "pipeline-bars": "bars", "pipeline-universe": "universe", "research-dataset": "research-dataset",
  "research-model": "research-model", "research-predict": "research-predict", "research-allocate": "research-allocate",
  "research-prepare": "research-prepare", "portfolio-accounting": "portfolio-accounting",
  "offline-dispatch": "execution-engine", "offline-fill": "execution-engine",
  "schedule-twap": "execution-engine", "advance-clock": "execution-engine",
};
for (const [kind, action] of Object.entries(pipelineActions)) {
  get("analytics-kind").value = "workspace-" + kind;
  const previous = calls.length;
  await get("analytics-kind").dispatch("change"); await settle();
  assert.equal(calls.length, previous, "Selecting a pipeline action cannot submit it");
  const body = get("analytics-input").value;
  await get("run-analysis").dispatch("click"); await settle();
  checked(`explicit ${kind} workflow uses its backend contract`, () => {
    assert.equal(calls.at(-1).url, "/api/v1/workspace/" + action);
    assert.equal(calls.at(-1).options.body, body);
    assert.match(get("workbench-scope").textContent, /SYNTHETIC|research|offline|SIMULATED|simulation/);
  });
}
const datasetId = "d".repeat(64), runId = "a".repeat(64), allocationId = "b".repeat(64);
async function successfulPipeline(kind, data) {
  get("analytics-kind").value = "workspace-" + kind;
  await get("analytics-kind").dispatch("change");
  replyOverride = { body: fixture({ data: { title: "Authored result", metrics: {}, rows: [], ...data } }) };
  await get("run-analysis").dispatch("click"); await settle();
}
await successfulPipeline("research-dataset", { dataset_id: datasetId });
get("analytics-kind").value = "workspace-research-model";
await get("analytics-kind").dispatch("change");
checked("successful persisted dataset identity fills the next explicit model request", () => {
  assert.equal(JSON.parse(get("analytics-input").value).dataset_id, datasetId);
  assert.match(get("workflow-identities").textContent, new RegExp(datasetId));
});
await successfulPipeline("research-model", { run_id: runId });
get("analytics-kind").value = "workspace-research-allocate";
await get("analytics-kind").dispatch("change");
checked("successful research identity fills allocation without automatically preparing an order", () => assert.equal(JSON.parse(get("analytics-input").value).run_id, runId));
await successfulPipeline("research-allocate", { allocation_id: allocationId });
get("analytics-kind").value = "workspace-research-prepare";
await get("analytics-kind").dispatch("change");
checked("successful allocation identity fills explicit durable preparation", () => assert.equal(JSON.parse(get("analytics-input").value).allocation_id, allocationId));
await successfulPipeline("research-prepare", { prepared_client_ids: ["allocation:" + allocationId + ":0"] });
get("analytics-kind").value = "workspace-offline-fill";
await get("analytics-kind").dispatch("change");
checked("offline fill example uses a prepared client identity and remains explicitly simulated", () => assert.equal(JSON.parse(get("analytics-input").value).client_id, "allocation:" + allocationId + ":0"));
await successfulPipeline("research-dataset", { dataset_id: "c".repeat(64) });
get("analytics-kind").value = "workspace-research-allocate";
await get("analytics-kind").dispatch("change");
checked("a newly selected dataset clears dependent run and allocation identities", () => assert.equal(JSON.parse(get("analytics-input").value).run_id, "0".repeat(64)));
replyOverride = { body: fixture({ data: { title: "Admitted depth", metrics: {}, rows: [], depth: { instrument_id: "SYN101", source: "authored-depth", classification: "SYNTHETIC", feed: "CALLER_SUPPLIED", currency: "USD", sequence: 1, event_at: replayCutoff, available_at: replayCutoff, bids: [{price:"100",quantity:"9007199254740993"},{price:"99",quantity:"10"}], asks: [{price:"101",quantity:"20"}] } } }) };
await get("refresh").dispatch("click"); await settle();
checked("depth ladder preserves exact quantity text and source classification", () => {
  assert.equal(get("depth-levels").children.length, 2);
  assert.equal(get("depth-levels").children[0].children[0].textContent, "9007199254740993");
  assert.match(get("depth-provenance").textContent, /SYNTHETIC/);
});
replyOverride = new Error("offline transport fixture");
await get("refresh").dispatch("click"); await settle();
checked("failed depth request clears previous levels", () => assert.equal(get("depth-levels").children.length, 0));
await get("context-expand").dispatch("click");
checked("full provenance has an explicit accessible toggle", () => assert.equal(get("context-expand").getAttribute("aria-expanded"), "true"));
checked("camera starts paused under reduced-motion fallback", () => assert.equal(get("orbit-toggle").getAttribute("aria-pressed"), "false"));
const sprint5Actions = { "pairs-research": "pairs-research", "pairs-replay": "pairs-replay", "options-admit": "options-admit", "options-research": "options-research", "options-replay": "options-replay", "research-signals": "research-signals", "signals-replay": "signals-replay", "allocate-signals": "research-allocate", "allocate-pairs": "research-allocate" };
replyOverride = { body: localUnavailable() };
for (const [kind, endpoint] of Object.entries(sprint5Actions)) {
  get("analytics-kind").value = "workspace-" + kind;
  const previous = calls.length;
  await get("analytics-kind").dispatch("change"); await settle();
  assert.equal(calls.length, previous, "Selecting Sprint 5 research must never submit it");
  const text = get("analytics-input").value;
  await get("run-analysis").dispatch("click"); await settle();
  checked(`Sprint 5 ${kind} is an explicit backend request`, () => {
    assert.equal(calls.at(-1).url, "/api/v1/workspace/" + endpoint);
    assert.equal(calls.at(-1).options.body, text);
  });
}
get("analytics-kind").value = "workspace-options-admit";
await get("analytics-kind").dispatch("change");
checked("options fixture declares modeled synthetic quotes without OPRA coverage", () => {
  const body = JSON.parse(get("analytics-input").value);
  assert.equal(body.classification, "SYNTHETIC");
  assert.equal(body.feed, "MODELED_FIXTURE_NOT_OPRA");
  assert.equal(body.quotes.length, 12);
  assert.match(get("workbench-scope").textContent, /not observed OPRA/);
});
get("data-mode").value = "local";
await get("data-mode").dispatch("change"); await settle();
for (const engine of ["pairs", "options-research", "signals-research"]) {
  get("analysis-engine").value = engine;
  await get("analysis-engine").dispatch("change"); await settle();
  checked(`Sprint 5 ${engine} view preserves backend engine identity`, () => assert(calls.at(-1).url.includes("engine=" + engine)));
}
await successfulPipeline("pairs-research", { run_id: "e".repeat(64), dataset_id: datasetId });
await successfulPipeline("research-signals", { signal_id: "f".repeat(64), dataset_id: datasetId });
get("analytics-kind").value = "workspace-pairs-replay";
await get("analytics-kind").dispatch("change");
checked("saved pair identity fills replay independently of CPU model ID", () => assert.equal(JSON.parse(get("analytics-input").value).run_id, "e".repeat(64)));
get("analytics-kind").value = "workspace-allocate-signals";
await get("analytics-kind").dispatch("change");
checked("signal allocation selects exactly one saved origin", () => {
  const body = JSON.parse(get("analytics-input").value);
  assert.equal(body.signal_id, "f".repeat(64));
  assert.equal(body.run_id, undefined);
  assert.equal(body.pair_run_id, undefined);
});
checked("experiment comparison records only actual submitted identities", () => {
  assert.equal(get("experiment-comparison").classList.contains("hidden"), false);
  assert.match(get("comparison-policy").textContent, /Session-only comparison/);
  assert.match(get("comparison-body").textContent, /eeeeeeee/);
  assert.match(get("comparison-body").textContent, /ffffffff/);
});
replyOverride = { body: fixture({ data: { metrics: {}, rows: [], folds: [{ screening: [{ dependent: "SYN101", independent: "SYN102", selected: false, reason: "NO_COINTEGRATION" }] }] } }) };
await get("refresh").dispatch("click"); await settle();
checked("rejected pairs remain visible as backend screening evidence", () => assert.match(tbody().textContent, /NO_COINTEGRATION/));
const sprint6Actions = {
  "funded-initialize": "funded-account", "funded-register": "funded-account", "funded-reserve": "funded-account",
  "funded-dispatch": "funded-account", "funded-fill": "funded-account", "funded-cancel": "funded-account",
  "funded-mark": "funded-account", "funded-settle": "funded-account", "funded-state": "funded-account", "funded-replay": "funded-account",
  "economic-research": "economic-research", "economic-replay": "economic-replay", "funded-risk": "funded-risk", "funded-risk-replay": "funded-risk-replay",
  "microstructure-admit": "microstructure-admit", "microstructure-research": "microstructure-research", "microstructure-replay": "microstructure-replay",
};
replyOverride = { body: localUnavailable() };
for (const [kind, endpoint] of Object.entries(sprint6Actions)) {
  get("analytics-kind").value = "workspace-" + kind;
  const previous = calls.length;
  await get("analytics-kind").dispatch("change"); await settle();
  assert.equal(calls.length, previous, "Selecting Sprint 6 actions cannot submit them");
  const text = get("analytics-input").value;
  await get("run-analysis").dispatch("click"); await settle();
  checked(`Sprint 6 ${kind} submits the original explicit JSON`, () => {
    assert.equal(calls.at(-1).url, "/api/v1/workspace/" + endpoint);
    assert.equal(calls.at(-1).options.body, text);
  });
}
for (const engine of ["funded-account", "funded-risk", "economic-research", "microstructure-research"]) {
  get("analysis-engine").value = engine;
  await get("analysis-engine").dispatch("change"); await settle();
  checked(`Sprint 6 ${engine} keeps its evidence identity`, () => assert(calls.at(-1).url.includes("engine=" + engine)));
}
await successfulPipeline("economic-research", { report_id: "1".repeat(64), economically_validated: false });
get("analytics-kind").value = "workspace-economic-replay";
await get("analytics-kind").dispatch("change");
checked("economic report identity is separate from model and pair experiments", () => assert.equal(JSON.parse(get("analytics-input").value).report_id, "1".repeat(64)));
await successfulPipeline("microstructure-research", { report_id: "2".repeat(64) });
get("analytics-kind").value = "workspace-microstructure-replay";
await get("analytics-kind").dispatch("change");
checked("microstructure replay selects only its saved report identity", () => assert.equal(JSON.parse(get("analytics-input").value).report_id, "2".repeat(64)));
await successfulPipeline("funded-mark", { snapshot_id: "3".repeat(64) });
get("analytics-kind").value = "workspace-funded-replay";
await get("analytics-kind").dispatch("change");
checked("simulated funded snapshot uses the account replay operation", () => {
  const body = JSON.parse(get("analytics-input").value);
  assert.equal(body.snapshot_id, "3".repeat(64));
  assert.equal(body.operation, "replay");
});
replyOverride = { body: fixture({ data: { metrics: {}, rows: [], account: { funding: { state: "AVAILABLE", scope: "SUPPORTED_CONTRACTS_ONLY" } } } }) };
await get("refresh").dispatch("click"); await settle();
checked("scoped cash funding never becomes a global certification", () => assert.match(get("truth-notice").textContent, /no global funding certification/));
get("analytics-kind").value = "workspace-funded-fill";
await get("analytics-kind").dispatch("change");
checked("funded partial fill is an explicit idempotent protocol event", () => {
  const body = JSON.parse(get("analytics-input").value);
  assert.equal(body.operation, "event");
  assert.equal(body.event.kind, "FILL");
  assert.equal(body.event.event_id, "authored-funded-fill-1");
  assert.equal(body.event.quantity, "1");
});
replyOverride = { body: fixture({ data: { metrics: { nav: null }, rows: [], account: { funding_scope: "USD_LONG_ONLY_SUPPORTED_CONTRACTS", mark_state: "UNKNOWN", reservations: [] } } }) };
await get("refresh").dispatch("click"); await settle();
checked("real funded account shape retains scope and unknown valuation notice", () => {
  assert.match(get("truth-notice").textContent, /no global funding certification/);
  assert.match(get("truth-notice").textContent, /Mark valuation UNKNOWN/);
});
replyOverride = { body: fixture({ data: { metrics: {}, rows: [], account: { funding_scope: "USD_LONG_ONLY_SUPPORTED_CONTRACTS", mark_state: "AVAILABLE", reservations: [{ client_id: "fixture-open", remaining_quantity: "1", buy_cash: "300", fee_cash: "1.9" }] } } }) };
await get("refresh").dispatch("click"); await settle();
checked("outstanding cash reservations remain inspectable as backend evidence", () => assert.match(tbody().textContent, /fixture-open/));
const sprint7Actions = {
  "funded-factor-risk": "funded-risk",
  "lifecycle-configure": "funded-lifecycle", "lifecycle-split": "funded-lifecycle", "lifecycle-dividend": "funded-lifecycle", "lifecycle-apply": "funded-lifecycle", "lifecycle-state": "funded-lifecycle", "lifecycle-replay": "funded-lifecycle",
  "statement-admit": "statement-admit", "statement-reconcile": "statement-reconcile", "statement-replay": "statement-replay",
  "observed-fetch": "observed-reference", "observed-admit": "observed-reference", "observed-research": "observed-reference", "observed-replay": "observed-reference"
};
replyOverride = { body: localUnavailable() };
for (const [kind, endpoint] of Object.entries(sprint7Actions)) {
  get("analytics-kind").value = "workspace-" + kind;
  const previous = calls.length;
  await get("analytics-kind").dispatch("change"); await settle();
  assert.equal(calls.length, previous, "Selecting Sprint 7 actions cannot submit them");
  const text = get("analytics-input").value;
  await get("run-analysis").dispatch("click"); await settle();
  checked(`Sprint 7 ${kind} retains explicit original JSON`, () => { assert.equal(calls.at(-1).url, "/api/v1/workspace/" + endpoint); assert.equal(calls.at(-1).options.body, text); });
}
for (const engine of ["funded-lifecycle", "independent-statement", "observed-reference"]) {
  get("analysis-engine").value = engine;
  await get("analysis-engine").dispatch("change"); await settle();
  checked(`Sprint 7 ${engine} preserves backend evidence identity`, () => assert(calls.at(-1).url.includes("engine=" + engine)));
}
await successfulPipeline("statement-admit", { statement_hash: "4".repeat(64) });
await successfulPipeline("statement-reconcile", { report_id: "5".repeat(64), reconciliation_state: "DISCREPANCY", external_reconciled: false });
get("analytics-kind").value = "workspace-statement-replay";
await get("analytics-kind").dispatch("change");
checked("independent statement replay identity remains separate", () => assert.equal(JSON.parse(get("analytics-input").value).report_id, "5".repeat(64)));
await successfulPipeline("lifecycle-split", { action_hash: "6".repeat(64), snapshot_id: "7".repeat(64) });
get("analytics-kind").value = "workspace-lifecycle-apply";
await get("analytics-kind").dispatch("change");
checked("lifecycle application selects only a successful admitted action hash", () => assert.equal(JSON.parse(get("analytics-input").value).action_hash, "6".repeat(64)));
await successfulPipeline("observed-admit", { snapshot_id: "8".repeat(64) });
get("analytics-kind").value = "workspace-observed-research";
await get("analytics-kind").dispatch("change");
checked("reference research uses separate successful source identity", () => assert.equal(JSON.parse(get("analytics-input").value).snapshot_id, "8".repeat(64)));
get("analytics-kind").value = "workspace-observed-fetch";
await get("analytics-kind").dispatch("change");
checked("public reference acquisition warns about dates unknown volume and historic availability", () => { assert.match(get("workbench-scope").textContent, /DATE/); assert.match(get("workbench-scope").textContent, /UNKNOWN/); });
replyOverride = { body: fixture({ data: { rows: [{ field: "cash", expected: "899.8", supplied: "900" }], metrics: {}, reconciliation_state: "DISCREPANCY", external_reconciled: false } }) };
await get("refresh").dispatch("click"); await settle();
checked("completed discrepancy report never claims successful external reconciliation", () => { assert.match(get("truth-notice").textContent, /DISCREPANCY/); assert.match(get("truth-notice").textContent, /not reconciled/); });
const sprint8Actions = { "lifecycle-accrue": "funded-lifecycle", "lifecycle-dividend-revise": "funded-lifecycle", "lifecycle-dividend-cancel": "funded-lifecycle", "observed-evaluate": "observed-reference", "observed-evaluation-replay": "observed-reference" };
replyOverride = { body: localUnavailable() };
for (const [kind, endpoint] of Object.entries(sprint8Actions)) {
  get("analytics-kind").value = "workspace-" + kind;
  const count = calls.length;
  await get("analytics-kind").dispatch("change"); await settle();
  assert.equal(calls.length, count, "Selecting Sprint 8 examples cannot submit them");
  const original = get("analytics-input").value;
  await get("run-analysis").dispatch("click"); await settle();
  checked(`Sprint 8 ${kind} uses explicit original input and endpoint`, () => { assert.equal(calls.at(-1).url, "/api/v1/workspace/" + endpoint); assert.equal(calls.at(-1).options.body, original); });
}
get("analytics-kind").value = "workspace-lifecycle-dividend-cancel";
await get("analytics-kind").dispatch("change");
checked("cancellation is a later explicit zero-rate revision", () => { const request = JSON.parse(get("analytics-input").value); assert.equal(request.action.status, "CANCELLED"); assert.equal(request.action.amount_per_share, "0"); assert.equal(request.action.revision, 3); });
await successfulPipeline("lifecycle-accrue", { action_hash: "9".repeat(64), account: { cash: "899.800000000000000001", available_cash: "899.8", dividend_receivable: "10", dividend_income: "10", nav: "999.8", dividend_accrual_state: "AVAILABLE" } });
checked("receivable summary preserves exact decimals and nonspendable scope", () => { assert.match(get("analysis-summary-values").textContent, /899\.800000000000000001/); assert.match(get("analysis-summary-values").textContent, /Dividend receivable/); assert.match(get("analysis-summary-note").textContent, /not available cash/); });
await successfulPipeline("observed-evaluate", { evaluation_id: "a".repeat(64), scoring_family: "CROSS_SECTIONAL_RANKING", evaluation_state: "AVAILABLE", fit_diagnostics: { coverage_state: "PARTIAL", converged_models: 2, models: 3 }, metrics: { mean_ndcg: .75, mean_spearman_ic: null }, economically_validated: false });
checked("model summary retains method-specific unknown score and partial coverage", () => { assert.match(get("analysis-summary-values").textContent, /CROSS_SECTIONAL_RANKING/); assert.match(get("analysis-summary-values").textContent, /PARTIAL/); assert.match(get("analysis-summary-values").textContent, /Unknown/); assert.match(get("analysis-summary-values").textContent, /Not established/); });
checked("saved evaluation identity is visible and independently replayable", () => { assert.match(get("workflow-identities").textContent, /Observed Evaluation Id: a{64}/); });
get("analytics-kind").value = "workspace-observed-evaluation-replay";
await get("analytics-kind").dispatch("change");
checked("evaluation replay selects its own successful identity", () => assert.equal(JSON.parse(get("analytics-input").value).evaluation_id, "a".repeat(64)));
get("analytics-input").value = "invalid";
await get("run-analysis").dispatch("click");
checked("invalid new input cannot retain an apparent successful summary", () => assert(get("analysis-summary").classList.contains("hidden")));
for (const kind of ["equity-fetch", "equity-admit", "equity-replay"]) {
  replyOverride = { body: localUnavailable() };
  get("analytics-kind").value = "workspace-" + kind;
  const count = calls.length;
  await get("analytics-kind").dispatch("change"); await settle();
  checked(`${kind} requires explicit submission`, () => assert.equal(calls.length, count));
  const original = get("analytics-input").value;
  await get("run-analysis").dispatch("click"); await settle();
  checked(`${kind} preserves bounded equity endpoint and input`, () => { assert.equal(calls.at(-1).url, "/api/v1/workspace/equity-reference"); assert.equal(calls.at(-1).options.body, original); });
}
await successfulPipeline("equity-admit", { snapshot_id: "b".repeat(64) });
get("analytics-kind").value = "workspace-equity-replay";
await get("analytics-kind").dispatch("change");
checked("equity replay retains its distinct source identity", () => assert.equal(JSON.parse(get("analytics-input").value).snapshot_id, "b".repeat(64)));
get("analysis-engine").value = "observed-equity";
replyOverride = { body: fixture({ data: { rows: [], metrics: {} } }) };
await get("analysis-engine").dispatch("change"); await settle();
checked("weekly archive view retains source eligibility limits", () => { assert(calls.at(-1).url.includes("engine=observed-equity")); assert.match(get("truth-notice").textContent, /permanent security/); assert.match(get("truth-notice").textContent, /Not eligible/); });
if (process.argv[2]) fs.writeFileSync(process.argv[2], JSON.stringify(submittedWorkspaceExamples, null, 2));
console.log(`Verified ${checks} frontend smoke checks. This Node DOM stub does not establish browser rendering, accessibility conformance, or visual QA.`);
