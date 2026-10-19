"""
Radiologist-facing client. Talks only to the API -- no model, no torch.

    streamlit run services/ui/app.py

DESIGN RULES (carried over from app/app.py -- do not drift from these)
1. The tool MEASURES. It never diagnoses, scores malignancy, or recommends a
   clinical action. The radiologist reads the scan and decides.
2. Every number is shown with the mask it came from, so it can be checked at
   a glance. Re-checking must be easy, never discouraged.
3. Uncertain cases are flagged loudly.
4. No confidence language that implies certainty.
"""
import os
import time

import requests
import streamlit as st

API = os.environ.get("API_URL", "http://api:8000").rstrip("/")
POLL_SECONDS = 1.0
POLL_LIMIT = 180

CONF = {
    "high":    ("#1b7f4d", "Measurement looks stable"),
    "medium":  ("#9a6700", "Check the outline before using these numbers"),
    "low":     ("#a33333", "Low confidence - verify manually"),
    "unknown": ("#666666", "No measurement"),
}

st.set_page_config(page_title="Nodule measurement assistant", layout="wide")


def api(method, path, **kw):
    r = requests.request(method, f"{API}{path}", timeout=kw.pop("timeout", 30), **kw)
    r.raise_for_status()
    return r.json() if r.headers.get("content-type", "").startswith("application/json") else r


st.title("Nodule measurement assistant")
st.caption(
    "Measurements only - not a diagnosis and not a clinical recommendation. "
    "All interpretation remains with the radiologist. "
    "Research and education use only; LIDC-IDRI data."
)

# ------------------------------------------------------------------ health
try:
    health = api("GET", "/healthz")
except Exception as e:
    st.error(f"The API is not reachable at {API}.\n\n{e}")
    st.info("If you started this with docker compose, give the api container a "
            "few seconds and reload.")
    st.stop()

cfg = health.get("config", {})
if cfg.get("dry_run"):
    st.warning(
        "**DRY RUN** - the dummy segmenter is running, so the outlines and "
        "numbers below are placeholders. Set `DRY_RUN=0` and provide the "
        "checkpoint for real inference.", icon="⚠️")

# -------------------------------------------------------------------- case
with st.sidebar:
    st.header("Case")
    st.caption(f"device `{cfg.get('device')}` · queue {health.get('workers', 0)} waiting")

    cases = api("GET", "/api/cases").get("cases", [])
    labels = {c["case_id"]: f"{c['case_id']}  ({c['source']})" for c in cases}

    if cases:
        case_id = st.selectbox("Select", list(labels), format_func=labels.get)
    else:
        case_id = None
        st.info("No cases yet. Create a phantom or upload an ROI below.")

    st.divider()
    st.subheader("Add a case")

    up = st.file_uploader("ROI volume (.npz)", type=["npz"],
                          help="image_hu (Z, Y, X) + spacing, as written by "
                               "scripts/02_prepare_data.py")
    if up is not None and st.button("Upload", width="stretch"):
        try:
            out = api("POST", "/api/cases",
                      files={"file": (up.name, up.getvalue(), "application/octet-stream")},
                      timeout=120)
            st.success(f"created {out['case_id']}")
            st.rerun()
        except Exception as e:
            st.error(f"upload failed: {e}")

    dia = st.slider("Phantom diameter (mm)", 3.0, 30.0, 9.0, 0.5)
    if st.button("Create phantom", width="stretch",
                 help="Synthetic sphere, for demonstrating the stack without the dataset"):
        out = api("POST", "/api/cases/phantom", json={"diameter_mm": dia})
        st.success(f"created {out['case_id']}")
        st.rerun()

if not case_id:
    st.stop()

case = api("GET", f"/api/cases/{case_id}")
D, H, W = case["shape"]
sz, sy, sx = case["spacing_mm"]

if case["source"] == "phantom":
    st.info("This is a **synthetic phantom**, not a patient scan. It exists to "
            "exercise the pipeline end to end.", icon="🧪")

# ------------------------------------------------------------------ prompt
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Image")
    z = st.slider("Slice", 0, D - 1, D // 2)
    st.image(f"{API}/api/cases/{case_id}/slice/{z}.png",
             width="stretch", caption=f"slice {z} of {D - 1}")

    st.markdown("**Prompt** — place a box around the nodule on this slice")
    c1, c2, c3 = st.columns(3)
    cx = c1.number_input("centre x", 0, W - 1, W // 2)
    cy = c2.number_input("centre y", 0, H - 1, H // 2)
    half = c3.number_input("half-size (px)", 4, min(H, W) // 2, 16)

    box = [max(0, cx - half), max(0, cy - half),
           min(W - 1, cx + half), min(H - 1, cy + half)]
    st.caption(f"box = {box}  ·  the model is given this and nothing else")

    run = st.button("Segment", type="primary", width="stretch")

# ------------------------------------------------------------------ result
with right:
    st.subheader("Measurements")

    if run:
        job = api("POST", f"/api/cases/{case_id}/segment",
                  json={"slice": int(z), "box": [float(v) for v in box]})
        st.session_state["job_id"] = job["job_id"]

    job_id = st.session_state.get("job_id")
    if not job_id:
        st.info("Set a box on the image, then press Segment.")
        st.stop()

    status = st.empty()
    job = api("GET", f"/api/jobs/{job_id}")
    waited = 0
    while job["status"] in ("queued", "running") and waited < POLL_LIMIT:
        status.info(f"{job['status']}… ({waited}s)")
        time.sleep(POLL_SECONDS)
        waited += 1
        job = api("GET", f"/api/jobs/{job_id}")
    status.empty()

    if job["status"] == "failed":
        st.error("The job failed.")
        with st.expander("Worker traceback"):
            st.code(job.get("error") or "no detail recorded")
        st.stop()
    if job["status"] != "done":
        st.warning(f"Still {job['status']} after {POLL_LIMIT}s. Reload to keep waiting.")
        st.stop()

    res = job.get("result") or {}
    if res.get("empty"):
        st.warning("The model returned an empty mask for that prompt. "
                   "Try centring the box on the nodule.")
        st.stop()

    level = res.get("confidence", "unknown")
    colour, text = CONF.get(level, CONF["unknown"])
    st.markdown(
        f"<div style='background:{colour};color:#fff;padding:10px 14px;"
        f"border-radius:6px;font-weight:600'>{level.upper()} — {text}</div>",
        unsafe_allow_html=True)

    for f in res.get("flags", []):
        st.caption(f"• {f}")

    st.image(f"{API}/api/jobs/{job_id}/overlay.png", width="stretch",
             caption="red = model outline, yellow = the prompt box")

    m1, m2 = st.columns(2)
    m1.metric("Mean diameter", f"{res.get('diameter_mean_mm', float('nan')):.1f} mm")
    m2.metric("Volume", f"{res.get('volume_mm3', float('nan')):.0f} mm³")

    st.caption(f"{res.get('slices_covered', '?')} slices · "
               f"{res.get('seconds', '?')}s on {res.get('device', '?')}")

    if res.get("density_class") and not res.get("density_calibrated", False):
        st.caption(f"Density reads as **{res['density_class']}**, but the density "
                   "classifier is not calibrated — treat it as provisional and "
                   "read the HU values instead.")

    with st.expander("All measured values"):
        st.json(res.get("measurement", {}))
