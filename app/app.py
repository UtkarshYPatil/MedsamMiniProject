"""
Nodule measurement assistant -- radiologist-facing tool.

    streamlit run app/app.py

DESIGN RULES (do not drift from these)
--------------------------------------
1. The tool MEASURES. It never diagnoses, scores malignancy, or recommends
   a clinical action. The radiologist reads the scan and decides.
2. Every number is shown with the mask it came from, so it can be checked
   at a glance. Re-checking must be easy, never discouraged.
3. Uncertain cases are flagged loudly. Ground-glass and part-solid nodules
   are flagged by default -- they are where segmentation is weakest.
4. No confidence language that implies certainty ("likely", "probably").
   State the measurement; let the reader judge it.
"""
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CFG
from src.data.preprocess import apply_window
from src.prompts.bbox import bbox_from_click, bbox_from_mask, best_slice
from src.models.medsam2 import build
from src.report.sizing import summarize, as_dict

st.set_page_config(page_title="Nodule measurement assistant", layout="wide")

CONF_STYLE = {
    "high":   ("#1b7f4d", "Measurement looks stable"),
    "medium": ("#9a6700", "Check the outline before using these numbers"),
    "low":    ("#a33", "Low confidence - verify manually"),
    "unknown": ("#666", "No measurement"),
}


# ---------------------------------------------------------------- model

@st.cache_resource(show_spinner="Loading MedSAM2 (once per session)...")
def get_model(dry_run: bool):
    return build(CFG.model.checkpoint, CFG.model.cfg_name,
                 CFG.model.device, dry_run=dry_run).load()


@st.cache_data(show_spinner=False)
def list_cases():
    proc = Path(CFG.paths.processed)
    man = proc / "manifest.json"
    if not man.exists():
        return []
    return json.loads(man.read_text())


@st.cache_data(show_spinner=False)
def load_case(file_name: str):
    d = np.load(Path(CFG.paths.processed) / file_name)
    return d["image_hu"].astype(np.float32), d["mask"], d["spacing"]


# ---------------------------------------------------------------- render

def overlay(slice_u8, mask_2d, color=(255, 90, 60), alpha=0.40):
    rgb = np.repeat(slice_u8[..., None], 3, axis=-1).astype(np.float32)
    if mask_2d is not None and mask_2d.any():
        for c in range(3):
            rgb[..., c] = np.where(
                mask_2d, (1 - alpha) * rgb[..., c] + alpha * color[c], rgb[..., c])
        # outline
        from scipy.ndimage import binary_erosion
        edge = mask_2d & ~binary_erosion(mask_2d)
        rgb[edge] = color
    return rgb.astype(np.uint8)


def confidence_banner(level):
    col, msg = CONF_STYLE.get(level, CONF_STYLE["unknown"])
    st.markdown(
        f"<div style='padding:10px 14px;border-left:4px solid {col};"
        f"background:rgba(0,0,0,0.03);border-radius:4px'>"
        f"<b style='color:{col}'>Confidence: {level}</b><br>"
        f"<span style='font-size:0.9em'>{msg}</span></div>",
        unsafe_allow_html=True)


# ---------------------------------------------------------------- app

st.title("Nodule measurement assistant")
st.caption(
    "Research prototype. Produces measurements only -- not a diagnosis, "
    "not a clinical recommendation. All interpretation remains with the radiologist."
)

with st.sidebar:
    st.header("Case")
    dry = st.checkbox("Dry run (no model)", value=False,
                      help="Dummy segmenter for UI testing without a GPU.")
    cases = list_cases()
    if not cases:
        st.error("No processed cases. Run scripts/02_prepare_data.py first.")
        st.stop()

    labels = [f"{c['key']}  ({c['texture_class']}, {c['diameter_mm']:.1f} mm)"
              for c in cases]
    i = st.selectbox("Nodule", range(len(cases)), format_func=lambda k: labels[k])
    case = cases[i]

    st.divider()
    st.header("Prompt")
    mode = st.radio("Source", ["Click centre", "Use reference outline"],
                    help="In clinical use only 'Click centre' exists. The "
                         "reference option is for development against LIDC.")
    half = st.slider("Box half-size (px)", 6, 48, 16)

vol_hu, gt, spacing = load_case(case["file"])
D, H, W = vol_hu.shape
vol_u8 = apply_window(vol_hu, CFG.preprocess.window_level, CFG.preprocess.window_width)

z_default = best_slice(gt) if gt.any() else D // 2
z = st.slider("Slice", 0, D - 1, int(z_default))

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Image")
    show_ref = st.checkbox("Show reference outline", value=False)
    disp = overlay(vol_u8[z], gt[z] if show_ref else None, color=(80, 160, 255))
    st.image(disp, width="stretch", caption=f"slice {z} of {D-1}")

    if mode == "Click centre":
        c1, c2 = st.columns(2)
        cx = c1.number_input("Centre x", 0, W - 1, W // 2)
        cy = c2.number_input("Centre y", 0, H - 1, H // 2)
        box = bbox_from_click(cx, cy, half, (H, W))
    else:
        box = bbox_from_mask(gt[z], padding=CFG.prompt.bbox_padding_px)
        if box is None:
            st.warning("No reference outline on this slice.")

    run = st.button("Segment", type="primary", width="stretch",
                    disabled=box is None)

with right:
    st.subheader("Measurements")

    if run and box is not None:
        model = get_model(dry)
        with st.spinner("Segmenting..."):
            pred = model.segment_volume(vol_u8, z, box,
                                        propagate=CFG.prompt.propagate)
        st.session_state["pred"] = pred
        st.session_state["pred_z"] = z

    pred = st.session_state.get("pred")

    if pred is None:
        st.info("Set a prompt on the image, then press Segment.")
    else:
        m = summarize(pred, vol_hu, spacing)
        confidence_banner(m.confidence)
        st.write("")

        a, b = st.columns(2)
        a.metric("Mean diameter", f"{m.diameter_mean_mm} mm")
        b.metric("Max diameter", f"{m.diameter_max_mm} mm")
        a.metric("Volume", f"{m.volume_mm3:.0f} mm³")
        b.metric("Median HU", f"{m.hu_p50:.0f}")

        st.caption(
            f"Attenuation — core median {m.core_hu_p50:.0f} HU, "
            f"{m.core_solid_fraction*100:.0f}% of core above −200 HU, "
            f"90th percentile {m.hu_p90:.0f} HU"
        )

        if m.flags:
            with st.expander(f"{len(m.flags)} quality flag(s)"):
                for f in m.flags:
                    st.write(f"- {f}")

        st.divider()
        if not m.density_calibrated:
            st.warning(
                f"**Density suggestion: {m.density_class.replace('_','-')}** "
                "— provisional, not calibrated. On a 30-nodule sample this "
                "agreed with radiologist texture ratings 60% of the time, and "
                "every error was a reader-rated solid nodule that attenuates "
                "like ground glass. Judge from the HU figures above, not this "
                "label.",
                icon=":material/warning:")
        else:
            st.metric("Density", m.density_class.replace("_", "-"))

        if m.density_class == "part_solid" and m.solid_component_mm:
            st.metric("Solid component", f"{m.solid_component_mm} mm")

        with st.expander("All values"):
            st.json(as_dict(m))
            st.caption(f"LIDC reader texture rating for this nodule: "
                       f"{case['texture_class']} (development reference only).")

        if gt.any():
            from src.evaluate.metrics import dice
            st.caption(f"Dice vs LIDC consensus outline: **{dice(pred, gt):.3f}** "
                       f"(development only -- not available in clinical use)")

        st.divider()
        st.write("**Radiologist review**")
        d1, d2 = st.columns(2)
        if d1.button("Accept", width="stretch"):
            st.success("Recorded as accepted.")
        if d2.button("Reject outline", width="stretch"):
            st.session_state.pop("pred", None)
            st.warning("Cleared. Adjust the prompt and segment again.")
            st.rerun()

# overlay the prediction back on the image
if st.session_state.get("pred") is not None:
    p = st.session_state["pred"]
    with left:
        st.subheader("Result")
        st.image(overlay(vol_u8[z], p[z]), width="stretch",
                 caption=f"predicted outline, slice {z}"
                         f"{'' if p[z].any() else '  (no mask on this slice)'}")
