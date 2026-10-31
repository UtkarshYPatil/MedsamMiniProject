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

/* Two kinds of message share this strip, and they used to fight over one
   slot: a STATUS (the stack is broken, true until it isn't) and a NOTICE
   (about this case or this action). alert_ replaced the whole strip, so
   opening a case wiped "no worker connected" and the reader was left with
   a red word in the corner and no explanation of it. They are kept apart
   now: status survives case changes, notice does not, and neither can
   silently delete the other. */
let _status = null;              // {message, kind} or null
let _notice = null;

function renderAlerts() {
  const make = (a) => {
    const node = document.createElement("div");
    node.className = a.kind === "warn" ? "alert warn" : "alert";
    node.textContent = a.message;
    return node;
  };
  el.alerts.replaceChildren(...[_status, _notice].filter(Boolean).map(make));
}

function setStatus(message, kind = "error") {
  const next = message ? { message, kind } : null;
  // Idempotent: health() runs every 15s and must not restack or re-animate
  // a banner that has not changed.
  if (JSON.stringify(next) === JSON.stringify(_status)) return;
  _status = next;
  renderAlerts();
}

function alert_(message, kind = "error") {
  _notice = { message, kind };
  renderAlerts();
}
const clearAlerts = () => { _notice = null; renderAlerts(); };

/* ---------------------------------------------------------------- health */

async function health() {
  try {
    const h = await api("/healthz");
    const w = h.worker || {};
    el.device.textContent = w.device_detail || "worker has not reported";
    el.device.className = "state live";
    el.queue.textContent = h.queued ? `${h.queued} queued` : "";

    // worker_live is the heartbeat; workers_alive is RQ's registry, which a
    // Redis restart erases while the worker carries on working. Trust the
    // heartbeat, and fall back to the registry only on an older API.
    const live = h.worker_live !== undefined ? h.worker_live : h.workers_alive !== 0;
    if (!live) {
      el.device.textContent = "no worker connected";
      el.device.className = "state down";
      setStatus("No worker is connected, so a measurement would queue and " +
                "never run. Start it with:  docker compose up -d worker  " +
                "(add -f docker-compose.gpu.yml for CUDA), then check " +
                "docker compose logs worker.");
    } else if (w.checkpoint_present === false && !w.dry_run) {
      setStatus("The worker has no model checkpoint, so measuring will fail. " +
                "Put MedSAM2_CTLesion.pt in ./checkpoints and restart it.");
    } else if (w.dry_run) {
      setStatus("Dry run: a stand-in segmenter is active and every number " +
                "below is invented.", "warn");
    } else {
      setStatus(null);
    }
  } catch {
    el.device.textContent = `cannot reach ${API}`;
    el.device.className = "state down";
    setStatus(`Cannot reach the API at ${API}. Is the api container up?`);
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
  state.boxSource = null;
  state.jobId = null;
  clearAlerts();
  // Only on a CASE change -- never on scrubbing, which happens constantly
  // and must stay a 1:1 response to the slider.
  if (!reduceMotion) {
    el.slice.classList.remove("swap");
    void el.slice.offsetWidth;
    el.slice.classList.add("swap");
  }

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
  proposeBox(id);                    // offered, not awaited: the viewer is usable now
}

function setSlice(z) {
  state.z = z;
  el.zlabel.textContent = `${z} / ${state.shape[0] - 1}`;
  el.slice.src = `${API}/api/cases/${state.caseId}/slice/${z}.png`;
}

/* ------------------------------------------------------- drawing the box */

/* The <img> is object-fit:contain, so its element box is the whole viewport
   while the picture is letterboxed inside it. Every pointer/overlay
   coordinate must go through the PICTURE rect, not the element rect. */
/* A proposed box fades up rather than blinking into existence -- it is a
   state change the reader did not cause, so it has to announce itself. */
let suggestFade = 1;
function fadeInProposal() {
  if (reduceMotion) { suggestFade = 1; paint(); return; }
  const t0 = performance.now(), ms = 260;
  const step = (t) => {
    suggestFade = Math.min(1, (t - t0) / ms);
    paint();
    if (suggestFade < 1) requestAnimationFrame(step);
  };
  suggestFade = 0;
  requestAnimationFrame(step);
}

/* Ask the API where it thinks the nodule is. This is a PROPOSAL, and a
   weak one: on 100 validation ROIs the auto-prompt missed the nodule
   entirely about 7 times in 10. It is offered to save a drag on the easy
   cases, never as a finding, which is why it arrives dashed and with the
   miss rate stated rather than silently filled in. */
async function proposeBox(id) {
  let sg;
  try {
    sg = await api(`/api/cases/${id}/suggest`);
  } catch {
    return;                       // a missing proposal is not an error
  }
  if (state.caseId !== id) return;        // the reader moved on while we asked

  if (!sg || sg.box === null) {
    const detail = sg && sg.info && sg.info.detail;
    if (detail) alert_(detail, "warn");
    return;
  }

  // A sub-millimetre box is not a proposal, it is noise -- and the tool
  // already refuses to trust anything under 4mm. Offering it would enable
  // Measure on a box that cannot produce a usable number.
  const [, spy, spx] = state.spacing || [1, 1, 1];
  const bw = (sg.box[2] - sg.box[0]) * spx, bh = (sg.box[3] - sg.box[1]) * spy;
  if (bw < 4 || bh < 4) {
    alert_("The automatic prompt did not find anything convincing on this " +
           "case, so no starting box was placed. Draw one around the nodule.",
           "warn");
    return;
  }

  const z = Number.isFinite(sg.slice) ? sg.slice : state.z;
  el.z.value = String(z);
  setSlice(z);
  state.box = sg.box.map(Number);
  state.boxSource = sg.source || "auto";
  el.run.disabled = false;
  el.hint.hidden = true;
  boxLabel();
  fadeInProposal();

  alert_(sg.source === "centred_fallback"
    ? "No candidate stood out, so this is simply a box in the middle of the " +
      "crop. Treat it as a starting point and move it onto the nodule."
    : "Starting box placed automatically. It assumes the nodule sits near " +
      "the centre of the crop; on the validation set that assumption missed " +
      "the nodule in roughly 7 of 10 cases. Check it before measuring.",
    "warn");
}

function pictureRect() {
  const r = el.slice.getBoundingClientRect();
  const nw = el.slice.naturalWidth, nh = el.slice.naturalHeight;
  if (!nw || !nh) return r;
  const s = Math.min(r.width / nw, r.height / nh);
  const w = nw * s, h = nh * s;
  return { left: r.left + (r.width - w) / 2, top: r.top + (r.height - h) / 2,
           width: w, height: h };
}

let _accent = "";
function accentFill(a) {
  const c = accent();
  if (/^#[0-9a-f]{6}$/i.test(c)) {
    const n = parseInt(c.slice(1), 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
  }
  return `rgba(90, 185, 255, ${a})`;
}

function accent() {
  if (!_accent) {
    _accent = getComputedStyle(document.documentElement)
      .getPropertyValue("--signal").trim() || "#5ab9ff";
  }
  return _accent;
}

function canvasToImage(ev) {
  const r = pictureRect();
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

  const ir = pictureRect();
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

  // A box the TOOL proposed must not look like a box the READER placed.
  // Dashed and dimmer, until the reader takes hold of it -- at which point
  // pointerdown clears boxSource and it becomes solid.
  const proposed = !!state.boxSource;
  g.save();
  if (proposed) {
    g.globalAlpha = suggestFade;
    g.setLineDash([5, 4]);
  }
  // One source of truth for the accent: the reticle cannot drift away
  // from the palette, because it reads the same token the CSS does.
  g.strokeStyle = accent();
  g.lineWidth = proposed ? 1.2 : 1.5;
  g.lineCap = "round";
  g.beginPath();
  for (const [cx, cy, dx, dy] of [
    [L, T, 1, 1], [R, T, -1, 1], [L, B, 1, -1], [R, B, -1, -1],
  ]) {
    g.moveTo(cx + dx * arm, cy); g.lineTo(cx, cy); g.lineTo(cx, cy + dy * arm);
  }
  g.stroke();

  g.fillStyle = accentFill(proposed ? 0.04 : 0.07);
  g.fillRect(L, T, R - L, B - T);
  g.restore();
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
  state.boxSource = null;            // from here it is the reader's box
  suggestFade = 1;
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
  // Re-trigger the reveal: a measurement is occasional, so it can afford
  // to arrive rather than appear.
  el.result.classList.remove("reveal");
  void el.result.offsetWidth;
  el.result.classList.add("reveal");

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
