"""
Radiologist-facing client. Talks only to the API -- no model, no torch.

    streamlit run frontend/app.py

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


@st.cache_data(show_spinner=False, ttl=600)
def suggest_for(case_id: str):
    """Ask the API where the nodule probably is. Cached: it is deterministic."""
    return api("GET", f"/api/cases/{case_id}/suggest")


def api(method, path, **kw):
    r = requests.request(method, f"{API}{path}", timeout=kw.pop("timeout", 30), **kw)
    r.raise_for_status()
    return r.json() if r.headers.get("content-type", "").startswith("application/json") else r


def api_image(path, timeout=60):
    """
    Fetch an image and return its bytes.

    Handing st.image() a URL does not work here: Streamlit embeds the URL in
    the page and the BROWSER fetches it, but API_URL is http://api:8000 --
    a name that only resolves inside the Docker network. The browser cannot
    see it and renders a broken image.

    Fetching server-side and passing bytes keeps the request inside the
    network, so it also works unchanged behind a tunnel or a reverse proxy
    where the browser has no route to the API at all.
    """
    r = requests.get(f"{API}{path}", timeout=timeout)
    r.raise_for_status()
    return r.content


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

    # The device shown is the one the WORKER resolved, not the API's setting.
    # The API has no torch, so its own DEVICE value said "cpu" on a GPU host.
    wk = health.get("worker") or {}
    queued = health.get("queued", health.get("workers", 0))
    alive = health.get("workers_alive")

    if wk:
        st.caption(f"running on `{wk.get('device_detail', wk.get('device', '?'))}` · "
                   f"{queued} queued")
    else:
        st.caption(f"{queued} queued · the worker has not reported in yet")

    if alive == 0:
        st.error("No worker is connected, so jobs will queue and never run. "
                 "Check the worker container: `docker compose logs worker`.",
                 icon="🛑")

    if wk and not wk.get("checkpoint_present") and not wk.get("dry_run"):
        st.error("The worker has no model checkpoint, so segmentation will fail.",
                 icon="🛑")

    cases = api("GET", "/api/cases").get("cases", [])

    labels = {c["case_id"]: f"{c['case_id']}  ({c['source']})" for c in cases}

    if cases:
        case_id = st.selectbox("Select", list(labels), format_func=labels.get)
    else:
        case_id = None
        st.info("No cases yet. Create a phantom or upload an ROI below.")

    st.divider()
    st.subheader("Add a case")

    t_dcm, t_roi, t_phan = st.tabs(["CT series", "ROI (.npz)", "Phantom"])

    with t_dcm:
        dcm = st.file_uploader(
            "DICOM series (.zip)", type=["zip"],
            help="Zip the folder of .dcm slices for ONE series. Slice order, "
                 "HU rescaling and spacing are read from the files.")
        st.caption(
            "Research and education use only, on public de-identified data. "
            "Do not upload a real patient scan. Only pixels and geometry are "
            "read — no identifiers are stored."
        )
        if dcm is not None and st.button("Import series", width="stretch",
                                         type="primary"):
            with st.spinner("Reading the series…"):
                try:
                    out = api("POST", "/api/cases/dicom",
                              files={"file": (dcm.name, dcm.getvalue(),
                                              "application/zip")},
                              timeout=600)
                    st.success(f"imported {out['case_id']}")
                    nfo = out.get("dicom", {})
                    st.caption(f"{nfo.get('slices_used', '?')} slices · "
                               f"HU {nfo.get('hu_range', ['?', '?'])}")
                    for k in ("note", "warning", "thickness_vs_spacing"):
                        if nfo.get(k):
                            st.warning(nfo[k], icon="⚠️")
                    st.rerun()
                except Exception as e:
                    st.error(f"import failed: {e}")

    with t_roi:
        up = st.file_uploader("ROI volume (.npz)", type=["npz"],
                              help="image_hu (Z, Y, X) + spacing, as written by "
                                   "scripts/02_prepare_data.py")
        if up is not None and st.button("Upload", width="stretch"):
            try:
                out = api("POST", "/api/cases",
                          files={"file": (up.name, up.getvalue(),
                                          "application/octet-stream")},
                          timeout=120)
                st.success(f"created {out['case_id']}")
                st.rerun()
            except Exception as e:
                st.error(f"upload failed: {e}")

    with t_phan:
        dia = st.slider("Phantom diameter (mm)", 3.0, 30.0, 9.0, 0.5)
        if st.button("Create phantom", width="stretch",
                     help="Synthetic sphere, for demonstrating the stack "
                          "without the dataset"):
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
elif case["source"] == "dicom":
    st.info(f"Imported CT series — {case.get('note') or 'geometry unknown'}. "
            "Research and education use only.", icon="🩻")

# ------------------------------------------------------------------ prompt
left, right = st.columns([3, 2], gap="large")

suggestion = suggest_for(case_id)
auto_z = int(suggestion["slice"])
source = suggestion.get("source")
found = source == "auto"
# A whole CT series gets no automatic box: the auto-prompt scores candidates by
# how central they are, which is meaningless on a full chest where the centre
# is mediastinum. box is None there and the radiologist places it.
whole_series = source == "unavailable"
auto_box = None if suggestion.get("box") is None else [float(v) for v in suggestion["box"]]

with left:
    st.subheader("Image")

    # The slider browses the stack. The prompt defaults to the slice the
    # suggestion landed on, so moving the slider to look around does not
    # silently change what gets segmented -- except on a whole series, where
    # browsing IS how the nodule gets found, so the two are tied together.
    z = st.slider("Slice", 0, D - 1, min(auto_z, D - 1), key=f"view_{case_id}")
    st.image(api_image(f"/api/cases/{case_id}/slice/{z}.png"),
             width="stretch", caption=f"slice {z} of {D - 1}")

    if whole_series:
        st.info("**Find the nodule yourself.** This is a full series, so there "
                "is no centre to assume. Scroll to the nodule, set the box "
                "below on it, then press Segment. The tool measures what you "
                "point it at — it does not detect nodules.", icon="🔎")
    elif found:
        st.success(f"Found a likely nodule on **slice {auto_z}** "
                   f"({suggestion['info'].get('voxels', '?')} voxels, spanning "
                   f"{suggestion['info'].get('slices_spanned', '?')} slices). "
                   f"Press Segment.", icon="✅")
    else:
        st.warning("No likely nodule found automatically, so the box below is "
                   "just centred. Check it before segmenting.", icon="⚠️")

    if whole_series:
        # Manual placement is the whole job here, so it is not hidden away.
        prompt_z = z
        c1, c2, c3 = st.columns(3)
        cx = c1.number_input("centre x", 0, W - 1, W // 2, key=f"cx_{case_id}")
        cy = c2.number_input("centre y", 0, H - 1, H // 2, key=f"cy_{case_id}")
        half = c3.number_input("half-size (px)", 4, min(H, W) // 2, 12,
                               key=f"hs_{case_id}")
        box = [float(max(0, cx - half)), float(max(0, cy - half)),
               float(min(W - 1, cx + half)), float(min(H - 1, cy + half))]
        st.caption(f"Prompting the slice you are viewing ({z}). The model is "
                   f"given a {2 * 16 + 1}x128x128 crop around this box — the "
                   f"same geometry every reported accuracy figure was measured "
                   f"on.")
    else:
        prompt_z, box = auto_z, auto_box

        with st.expander("Adjust the prompt"):
            st.caption(
                "The automatic box assumes the nodule is near the middle of the "
                "crop and ignores anything touching the edge, which is where the "
                "chest wall enters. Override it if it has picked the wrong thing."
            )
            if st.checkbox("Prompt the slice I am viewing instead",
                           value=False, key=f"usez_{case_id}"):
                prompt_z = z

            cx_d = int(round((auto_box[0] + auto_box[2]) / 2))
            cy_d = int(round((auto_box[1] + auto_box[3]) / 2))
            half_d = max(4, int(round(max(auto_box[2] - auto_box[0],
                                          auto_box[3] - auto_box[1]) / 2)))

            c1, c2, c3 = st.columns(3)
            cx = c1.number_input("centre x", 0, W - 1, cx_d, key=f"cx_{case_id}")
            cy = c2.number_input("centre y", 0, H - 1, cy_d, key=f"cy_{case_id}")
            half = c3.number_input("half-size (px)", 4, min(H, W) // 2, half_d,
                                   key=f"hs_{case_id}")

            if (cx, cy, half) != (cx_d, cy_d, half_d):
                box = [float(max(0, cx - half)), float(max(0, cy - half)),
                       float(min(W - 1, cx + half)), float(min(H - 1, cy + half))]
                st.caption("Using your box instead of the automatic one.")

    st.caption(f"prompt → slice {prompt_z}, box {[int(v) for v in box]} · "
               f"the model is given this and nothing else")

    run = st.button("Segment", type="primary", width="stretch")

# ------------------------------------------------------------------ result
with right:
    st.subheader("Measurements")

    if run:
        job = api("POST", f"/api/cases/{case_id}/segment",
                  json={"slice": int(prompt_z), "box": [float(v) for v in box]})
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

    st.image(api_image(f"/api/jobs/{job_id}/overlay.png"), width="stretch",
             caption="red = model outline, yellow = the prompt box")

    m1, m2 = st.columns(2)
    m1.metric("Mean diameter", f"{res.get('diameter_mean_mm', float('nan')):.1f} mm")
    m2.metric("Volume", f"{res.get('volume_mm3', float('nan')):.0f} mm³")

    st.caption(f"{res.get('slices_covered', '?')} slices · "
               f"{res.get('seconds', '?')}s on {res.get('device', '?')}")

    if res.get("cropped_to_roi"):
        oz, oy, ox = res.get("roi_origin", ["?"] * 3)
        st.caption(f"Measured on a 33x128x128 crop at (z={oz}, y={oy}, x={ox}) "
                   f"— the geometry the published accuracy figures were "
                   f"measured on, not the whole series.")

    # The density CLASS is not shown. On 902 nodules it agrees with reader
    # texture 51% of the time, against 85% for answering "solid" every time --
    # worse than a constant, so putting it on screen beside real measurements
    # would lend it credibility it has not earned. The HU statistics it is
    # derived from are genuine measurements, so those are shown instead.
    m = res.get("measurement", {})
    if m.get("core_hu_p50") is not None:
        st.caption(
            f"Core attenuation **{m['core_hu_p50']:.0f} HU** "
            f"(p90 {m.get('hu_p90', float('nan')):.0f}, "
            f"solid fraction {m.get('core_solid_fraction', float('nan')):.2f}). "
            "Density class is not reported: on 902 nodules it is less accurate "
            "than always answering 'solid', so read the HU."
        )

    with st.expander("All measured values"):
        st.json(res.get("measurement", {}))
