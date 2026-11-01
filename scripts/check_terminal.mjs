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
    this.style = {};
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
if (process.argv[2]) fs.writeFileSync(process.argv[2], JSON.stringify(submittedWorkspaceExamples, null, 2));
console.log(`Verified ${checks} frontend smoke checks. This Node DOM stub does not establish browser rendering, accessibility conformance, or visual QA.`);
