/*
  Nodule measurement — browser client.

  Talks to the FastAPI backend over HTTP and holds no model, no weights and no
  opinion about medicine. Everything it shows comes from the API.

  The one interaction that matters is drawing the box. Measuring on 100 real
  ROIs showed the automatic prompt lands on the wrong object 71% of the time,
  so the radiologist places it -- and dragging across a lesion is a better way
  to say "this one" than typing three numbers into spinners, which is what the
  Streamlit version asked for.
*/
"use strict";

const API = window.API_BASE || "http://localhost:8000";

const $ = (id) => document.getElementById(id);
const el = {
  device: $("device"), queue: $("queue"), alerts: $("alerts"),
  cases: $("cases"), file: $("file"), drop: $("drop"), phantom: $("phantom"),
  viewport: $("viewport"), slice: $("slice"), draw: $("draw"), hint: $("hint"),
  z: $("z"), zlabel: $("zlabel"), run: $("run"), boxlabel: $("boxlabel"),
  idle: $("idle"), result: $("result"), conf: $("conf"), flags: $("flags"),
  overlay: $("overlay"), prov: $("prov"), json: $("json"),
  diam: $("d-diam"), vol: $("d-vol"), max: $("d-max"), hu: $("d-hu"),
};

const state = {
  caseId: null, shape: null, spacing: null,   // shape = [z, y, x]
  z: 0, box: null,                            // box in IMAGE pixels
  drag: null, jobId: null, polling: false,
};

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ------------------------------------------------------------------ http */

async function api(path, opts = {}) {
  const r = await fetch(API + path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch { /* not json */ }
    throw new Error(detail);
  }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
}

function alert_(message, kind = "error") {
  const node = document.createElement("div");
  node.className = kind === "warn" ? "alert warn" : "alert";
  node.textContent = message;
  el.alerts.replaceChildren(node);
}
const clearAlerts = () => el.alerts.replaceChildren();

/* ---------------------------------------------------------------- health */

async function health() {
  try {
    const h = await api("/healthz");
    const w = h.worker || {};
    el.device.textContent = w.device_detail || "worker has not reported";
    el.device.className = "state live";
    el.queue.textContent = h.queued ? `${h.queued} queued` : "";

    if (h.workers_alive === 0) {
      el.device.textContent = "no worker connected";
      el.device.className = "state down";
      alert_("No worker is connected, so measurements will queue and never " +
             "run. Check the worker container.");
    } else if (w.checkpoint_present === false && !w.dry_run) {
      alert_("The worker has no model checkpoint, so measuring will fail.");
    } else if (w.dry_run) {
      alert_("Dry run: a stand-in segmenter is active and every number below " +
             "is invented.", "warn");
    }
  } catch {
    el.device.textContent = `cannot reach ${API}`;
    el.device.className = "state down";
  }
}

/* ----------------------------------------------------------------- cases */

async function loadCases(select) {
  const { cases } = await api("/api/cases");
  el.cases.replaceChildren(...cases.map((c) => {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.className = "case";
    b.type = "button";
    b.setAttribute("aria-current", String(c.case_id === state.caseId));
    b.innerHTML = "";
    b.append(c.case_id);
    const s = document.createElement("small");
    s.textContent = c.note || c.source;
    b.append(s);
    b.onclick = () => openCase(c.case_id);
    li.append(b);
    return li;
  }));
  if (select) openCase(select);
  else if (!state.caseId && cases.length) openCase(cases[0].case_id);
}

async function openCase(id) {
  state.caseId = id;
  state.box = null;
  state.jobId = null;
  clearAlerts();

  const c = await api(`/api/cases/${id}`);
  state.shape = c.shape;
  state.spacing = c.spacing_mm;

  const mid = Math.floor(c.shape[0] / 2);
  el.z.max = String(c.shape[0] - 1);
  el.z.value = String(mid);
  setSlice(mid);

  el.idle.hidden = false;
  el.result.hidden = true;
  el.hint.hidden = false;
  el.run.disabled = true;
  el.boxlabel.textContent = "";
  paint();
  loadCases();                       // refresh the current-case marker
}

function setSlice(z) {
  state.z = z;
  el.zlabel.textContent = `${z} / ${state.shape[0] - 1}`;
  el.slice.src = `${API}/api/cases/${state.caseId}/slice/${z}.png`;
}

/* ------------------------------------------------------- drawing the box */

function canvasToImage(ev) {
  // The <img> is letterboxed inside the viewport, so map through its real
  // rendered rectangle rather than the canvas, or the box lands off-target.
  const r = el.slice.getBoundingClientRect();
  const [, H, W] = state.shape;
  return {
    x: Math.max(0, Math.min(W - 1, ((ev.clientX - r.left) / r.width) * W)),
    y: Math.max(0, Math.min(H - 1, ((ev.clientY - r.top) / r.height) * H)),
  };
}

function paint() {
  const cv = el.draw;
  const r = el.viewport.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  cv.width = r.width * dpr;
  cv.height = r.height * dpr;
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, r.width, r.height);
  if (!state.box || !state.shape) return;

  const ir = el.slice.getBoundingClientRect();
  const [, H, W] = state.shape;
  const sx = ir.width / W, sy = ir.height / H;
  const ox = ir.left - r.left, oy = ir.top - r.top;
  const [x0, y0, x1, y1] = state.box;
  const L = ox + x0 * sx, T = oy + y0 * sy;
  const R = ox + x1 * sx, B = oy + y1 * sy;

  // CLOSURE: four corner marks, not a rectangle. The eye completes the shape,
  // and what is left on screen reads as a reticle rather than a border -- so
  // the lesion is framed instead of covered.
  const arm = Math.max(7, Math.min(18, (R - L) * 0.28));
  g.strokeStyle = "#35d0c0";
  g.lineWidth = 1.5;
  g.lineCap = "round";
  g.beginPath();
  for (const [cx, cy, dx, dy] of [
    [L, T, 1, 1], [R, T, -1, 1], [L, B, 1, -1], [R, B, -1, -1],
  ]) {
    g.moveTo(cx + dx * arm, cy); g.lineTo(cx, cy); g.lineTo(cx, cy + dy * arm);
  }
  g.stroke();

  g.fillStyle = "rgba(53, 208, 192, 0.07)";
  g.fillRect(L, T, R - L, B - T);
}

function boxLabel() {
  if (!state.box) { el.boxlabel.textContent = ""; return; }
  const [, sy, sx] = state.spacing;
  const [x0, y0, x1, y1] = state.box;
  const w = (x1 - x0) * sx, h = (y1 - y0) * sy;
  el.boxlabel.textContent = `${w.toFixed(1)} × ${h.toFixed(1)} mm`;
}

el.draw.addEventListener("pointerdown", (ev) => {
  if (!state.shape) return;
  el.draw.setPointerCapture(ev.pointerId);
  state.drag = canvasToImage(ev);
  el.hint.hidden = true;
});

el.draw.addEventListener("pointermove", (ev) => {
  if (!state.drag) return;
  const p = canvasToImage(ev);
  state.box = [
    Math.min(state.drag.x, p.x), Math.min(state.drag.y, p.y),
    Math.max(state.drag.x, p.x), Math.max(state.drag.y, p.y),
  ];
  paint();
  boxLabel();
});

el.draw.addEventListener("pointerup", () => {
  state.drag = null;
  // A stray click is not a box. Require something a lesion could fit in.
  if (state.box && (state.box[2] - state.box[0] < 3 || state.box[3] - state.box[1] < 3)) {
    state.box = null;
    el.hint.hidden = false;
    paint();
    boxLabel();
  }
  el.run.disabled = !state.box;
});

el.z.addEventListener("input", () => {
  setSlice(Number(el.z.value));
  paint();
});
window.addEventListener("resize", paint);
el.slice.addEventListener("load", paint);

/* ------------------------------------------------------------- measuring */

el.run.addEventListener("click", async () => {
  if (!state.box || state.polling) return;
  clearAlerts();
  el.run.disabled = true;
  el.run.classList.add("busy");
  el.run.textContent = "Measuring…";
  try {
    const job = await api(`/api/cases/${state.caseId}/segment`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        slice: state.z,
        box: state.box.map((v) => Math.round(v)),
      }),
    });
    state.jobId = job.job_id;
    await poll(job.job_id);
  } catch (e) {
    alert_(`Could not measure: ${e.message}`);
  } finally {
    el.run.classList.remove("busy");
    el.run.textContent = "Measure";
    el.run.disabled = !state.box;
  }
});

async function poll(jobId) {
  state.polling = true;
  try {
    for (let i = 0; i < 300; i++) {
      const j = await api(`/api/jobs/${jobId}`);
      if (j.status === "done") return show(j.result || {}, jobId);
      if (j.status === "failed") {
        alert_(`The measurement failed. ${(j.error || "").slice(-200)}`);
        return;
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
    alert_("Still running after five minutes. Reload to keep waiting.", "warn");
  } finally {
    state.polling = false;
  }
}

/* --------------------------------------------------------------- results */

const CONF = {
  high:   "Measurement looks stable.",
  medium: "Check the outline before using these numbers.",
  low:    "Low confidence — verify manually.",
};

function countTo(node, value, digits) {
  const to = Number(value);
  if (!isFinite(to)) { node.textContent = "—"; return; }
  if (reduceMotion) { node.textContent = to.toFixed(digits); return; }
  // Counting up is not decoration here: it shows the magnitude arriving,
  // which is the one thing the reader is waiting for.
  const t0 = performance.now(), ms = 420;
  const step = (t) => {
    const k = Math.min(1, (t - t0) / ms);
    const eased = 1 - Math.pow(1 - k, 3);
    node.textContent = (to * eased).toFixed(digits);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function show(res, jobId) {
  if (res.empty) {
    alert_("The model returned nothing for that box. Try drawing it tighter " +
           "around the nodule.", "warn");
    return;
  }
  el.idle.hidden = true;
  el.result.hidden = false;

  const level = res.confidence || "unknown";
  el.conf.className = `conf ${level}`;
  el.conf.textContent = CONF[level] || "No measurement.";

  el.flags.replaceChildren(...(res.flags || []).map((f) => {
    const li = document.createElement("li");
    li.textContent = f;
    return li;
  }));

  el.overlay.src = `${API}/api/jobs/${jobId}/overlay.png?t=${Date.now()}`;

  const m = res.measurement || {};
  countTo(el.diam, res.diameter_mean_mm, 1);
  countTo(el.vol, res.volume_mm3, 0);
  countTo(el.max, m.diameter_max_mm, 1);
  countTo(el.hu, m.core_hu_p50, 0);

  const bits = [
    `${res.slices_covered ?? "?"} slices`,
    `${res.seconds ?? "?"} s on ${res.device ?? "?"}`,
  ];
  if (res.cropped_to_roi) {
    const [oz, oy, ox] = res.roi_origin || [];
    bits.push(`measured on a 33×128×128 region at (${oz}, ${oy}, ${ox})`);
  }
  // The density class is withheld on purpose: on 902 nodules it agrees with
  // reader texture 51% of the time against 85% for always answering "solid".
  bits.push("Density class is not reported — it is less accurate than a " +
            "constant answer, so read the attenuation instead.");
  el.prov.textContent = bits.join(" · ");

  el.json.textContent = JSON.stringify(m, null, 2);
}

/* ----------------------------------------------------------------- input */

async function upload(file) {
  if (!file) return;
  const dicom = file.name.toLowerCase().endsWith(".zip");
  const body = new FormData();
  body.append("file", file);
  el.drop.querySelector(".drop-main").textContent =
    dicom ? "Reading series…" : "Uploading…";
  try {
    const out = await api(dicom ? "/api/cases/dicom" : "/api/cases",
                          { method: "POST", body });
    clearAlerts();
    const info = out.dicom || {};
    for (const k of ["note", "warning", "thickness_vs_spacing"]) {
      if (info[k]) alert_(info[k], "warn");
    }
    await loadCases(out.case_id);
  } catch (e) {
    alert_(`Could not add that file: ${e.message}`);
  } finally {
    el.drop.querySelector(".drop-main").textContent = "Drop a file";
  }
}

el.file.addEventListener("change", (e) => upload(e.target.files[0]));
for (const type of ["dragenter", "dragover"]) {
  el.drop.addEventListener(type, (e) => {
    e.preventDefault(); el.drop.classList.add("over");
  });
}
for (const type of ["dragleave", "drop"]) {
  el.drop.addEventListener(type, (e) => {
    e.preventDefault(); el.drop.classList.remove("over");
    if (type === "drop") upload(e.dataTransfer?.files?.[0]);
  });
}

el.phantom.addEventListener("click", async () => {
  el.phantom.disabled = true;
  try {
    const out = await api("/api/cases/phantom", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ diameter_mm: 9.0 }),
    });
    await loadCases(out.case_id);
  } catch (e) {
    alert_(`Could not create a phantom: ${e.message}`);
  } finally {
    el.phantom.disabled = false;
  }
});

/* ------------------------------------------------------------------ boot */

health();
setInterval(health, 15000);
loadCases().catch((e) => alert_(`Cannot reach the API at ${API}. ${e.message}`));
