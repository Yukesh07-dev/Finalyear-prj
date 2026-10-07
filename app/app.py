import sys
from pathlib import Path
from io import BytesIO
import base64

import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageOps

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import NUM_CLASSES, SELECTED_CLASSES
from src.models import build_finetuned_resnet50
from src.transforms import get_validation_transform
from src.gradcam import GradCAM
from src.bird_info import get_bird_info
from src.robustness import apply_perturbation


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Indian Bird Species Classification with Explainable AI & Robustness",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PREMIUM STYLING
# ============================================================

st.markdown(
    """
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(14, 165, 233, 0.12),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 0%,
            rgba(99, 102, 241, 0.12),
            transparent 30%
        ),
        #070b12;

    color: #f8fafc;
}

.main .block-container {
    max-width: 1250px;
    padding-top: 25px;
    padding-bottom: 60px;
    padding-left: 45px;
    padding-right: 45px;
}

.project-title {
    text-align: center;
    color: #e2e8f0;
    font-size: 23px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.project-line {
    width: 90px;
    height: 3px;
    margin: auto;
    border-radius: 20px;
    background: linear-gradient(
        90deg,
        #38bdf8,
        #6366f1
    );
}

.hero-area {
    text-align: center;
    padding-top: 30px;
    padding-bottom: 10px;
}

.hero-icon {
    font-size: 52px;
}

.hero-title {
    font-size: 64px;
    font-weight: 900;
    letter-spacing: -3px;
    color: #f8fafc;
    margin-top: 5px;
}

.hero-accent {
    color: #38bdf8;
}

.hero-description {
    max-width: 800px;
    margin: auto;
    color: #94a3b8;
    font-size: 16px;
    line-height: 1.7;
}

.hero-badge {
    margin-top: 18px;
    color: #7dd3fc;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.section-title {
    color: #f8fafc;
    font-size: 28px;
    font-weight: 850;
    margin-top: 40px;
    margin-bottom: 6px;
}

.section-subtitle {
    color: #718096;
    font-size: 14px;
    margin-bottom: 20px;
}

div[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.80);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 18px;
    padding: 20px;
    min-height: 105px;
}

div[data-testid="stMetricLabel"] {
    color: #64748b !important;
}

div[data-testid="stMetricValue"] {
    color: #f8fafc !important;
}

[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.75);
    border: 1px dashed rgba(56, 189, 248, 0.40);
    border-radius: 20px;
    padding: 12px;
}

.stButton > button {
    width: 100%;
    min-height: 52px;
    border-radius: 14px;
    border: 1px solid rgba(125, 211, 252, 0.35);
    background: linear-gradient(
        135deg,
        #0284c7,
        #4f46e5
    );
    color: white;
    font-size: 15px;
    font-weight: 800;
}

.stButton > button:hover {
    border-color: white;
    box-shadow: 0 12px 35px rgba(37, 99, 235, 0.30);
}

.info-box {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.13);
    border-radius: 20px;
    padding: 24px;
    min-height: 160px;
}

.info-title {
    color: #e2e8f0;
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 12px;
}

.info-text {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.75;
}

.result-box {
    background:
        radial-gradient(
            circle at 50% 0%,
            rgba(14, 165, 233, 0.16),
            transparent 60%
        ),
        rgba(15, 23, 42, 0.90);
    border: 1px solid rgba(56, 189, 248, 0.28);
    border-radius: 24px;
    padding: 28px;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 25px;
}

.result-title {
    color: #38bdf8;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.result-name {
    color: #f8fafc;
    font-size: 38px;
    font-weight: 900;
    margin-top: 8px;
}

.result-sci {
    color: #38bdf8;
    font-size: 18px;
    font-style: italic;
    margin-top: 2px;
}

.result-confidence {
    color: #94a3b8;
    font-size: 15px;
    margin-top: 8px;
}

.species-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.13);
    border-radius: 15px;
    padding: 15px;
    margin-bottom: 10px;
    color: #dbeafe;
    font-size: 15px;
    font-weight: 650;
    min-height: 50px;
    display: flex;
    align-items: center;
}

.bird-info-card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 20px;
    padding: 25px;
    margin-top: 15px;
}

.bird-field-title {
    color: #38bdf8;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 14px;
}

.bird-field-val {
    color: #e2e8f0;
    font-size: 15px;
    line-height: 1.6;
}

.footer-text {
    text-align: center;
    color: #475569;
    font-size: 12px;
    line-height: 1.8;
    margin-top: 60px;
    padding-top: 25px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DEVICE & PATHS
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "finetuned_resnet50"
    / "best_model.pth"
)


# ============================================================
# CACHED MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    model = build_finetuned_resnet50(
        num_classes=NUM_CLASSES,
        last_conv_layers=30,
    )

    if not CHECKPOINT_PATH.exists():
        st.error(
            f"⚠️ Trained model checkpoint file not found at `{CHECKPOINT_PATH.name}`.\n\n"
            "Please ensure the checkpoint `best_model.pth` is present in `checkpoints/finetuned_resnet50/` "
            "or tracked using Git LFS."
        )
        st.stop()

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# PREDICTION & GRAD-CAM INFERENCE
# ============================================================

def predict_and_explain(model, image: Image.Image):
    """
    Performs inference and generates Grad-CAM heatmap visualization.
    Returns:
    - predictions: List of (species_name, confidence_percent)
    - heatmap_img: Colorized Grad-CAM heatmap PIL Image
    - overlay_img: Blended overlay PIL Image
    """
    transform = get_validation_transform()
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # 1. Forward pass for prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=min(5, NUM_CLASSES),
        dim=1,
    )

    predictions = []
    for prob, idx in zip(top_probabilities[0], top_indices[0]):
        predictions.append((SELECTED_CLASSES[idx.item()], prob.item() * 100, idx.item()))

    target_class_idx = predictions[0][2]

    # 2. Grad-CAM generation
    gradcam = GradCAM(model=model, target_layer=model.layer4[-1])
    heatmap, _ = gradcam.generate(input_tensor=image_tensor.clone(), target_class=target_class_idx)
    heatmap_img, overlay_img = gradcam.overlay_heatmap(heatmap, image, alpha=0.5)
    gradcam.remove_hooks()  # Prevent memory leaks

    return predictions, heatmap_img, overlay_img


def create_preview_image(image, width=1000, height=430):
    image = image.convert("RGB")
    preview = Image.new("RGB", (width, height), (15, 23, 42))
    fitted = ImageOps.contain(image, (width - 30, height - 30))
    x = (width - fitted.width) // 2
    y = (height - fitted.height) // 2
    preview.paste(fitted, (x, y))
    return preview


# ============================================================
# HEADER & HERO SECTION
# ============================================================

st.markdown(
    '<div class="project-title">'
    'Taxonomic Classification Using Deep Learning'
    '</div>'
    '<div class="project-line"></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-area">'
    '<div class="hero-icon">🐦</div>'
    '<div class="hero-title">'
    'Avian<span class="hero-accent">Vision</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-description">
    An explainable Indian bird species identification system powered by a Fine-Tuned 
    <b>ResNet-50</b> deep learning model, integrated with <b>Grad-CAM visual explanation</b>, 
    real-world robustness analysis, and an offline species knowledge system.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-badge">
    FINE-TUNED RESNET-50 &nbsp; • &nbsp;
    GRAD-CAM EXPLAINABLE AI &nbsp; • &nbsp;
    ROBUSTNESS EVALUATED &nbsp; • &nbsp;
    PYTORCH
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PROJECT METRICS DASHBOARD
# ============================================================

stat1, stat2, stat3, stat4 = st.columns(4, gap="medium")

with stat1:
    st.metric("BIRD SPECIES", "15")

with stat2:
    st.metric("VALIDATION ACCURACY", "97.72%")

with stat3:
    st.metric("EXPLAINABILITY ENGINE", "Grad-CAM")

with stat4:
    st.metric("COMPUTE DEVICE", str(DEVICE).upper())


# ============================================================
# IDENTIFY BIRD & UPLOAD
# ============================================================

st.markdown('<div class="section-title">Identify a Bird</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">'
    'Upload a clear bird photograph to obtain instant classification, Grad-CAM interpretability, and field details.'
    '</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload your bird image",
    type=["jpg", "jpeg", "png", "bmp", "webp", "jfif"],
)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    image_col, info_col = st.columns([1.15, 0.85], gap="large")

    with image_col:
        st.subheader("Image Preview")
        preview_image = create_preview_image(image)
        st.image(preview_image, use_container_width=True)
        st.caption(f"{uploaded_file.name}  •  {image.width} × {image.height} pixels")

    with info_col:
        st.markdown(
            f'<div class="info-box">'
            f'<div class="info-title">Ready to Analyze</div>'
            f'<div class="info-text">'
            f'<b>Preprocessing:</b> Resized to 224 × 224 & Normalized via ImageNet stats.<br>'
            f'<b>Model:</b> Fine-Tuned ResNet-50 (Final 30 Conv Layers Unfrozen)<br>'
            f'<b>XAI Method:</b> Grad-CAM on layer4[-1] feature maps<br>'
            f'<b>Target Classes:</b> 15 Indian Bird Species<br>'
            f'<b>Execution Device:</b> {DEVICE}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.write("")
        classify = st.button("🔍  Classify & Explain Prediction", type="primary", use_container_width=True)

    if classify:
        with st.spinner("Analyzing image features & computing Grad-CAM gradients..."):
            model = load_model()
            predictions, heatmap_img, overlay_img = predict_and_explain(model, image)

        predicted_species = predictions[0][0]
        confidence = predictions[0][1]
        bird_data = get_bird_info(predicted_species)

        # ----------------------------------------------------
        # 1. PRIMARY RESULT CARD
        # ----------------------------------------------------
        st.markdown(
            f'<div class="result-box">'
            f'<div class="result-title">MOST LIKELY SPECIES</div>'
            f'<div class="result-name">{predicted_species}</div>'
            f'<div class="result-sci">{bird_data.get("scientific_name", "")}</div>'
            f'<div class="result-confidence">'
            f'Model Confidence: <b>{confidence:.2f}%</b>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # 2. EXPLAINABLE AI (GRAD-CAM) SECTION
        # ----------------------------------------------------
        st.markdown('<div class="section-title">Explainable AI (Grad-CAM Analysis)</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">'
            'Gradient-weighted Class Activation Mapping highlights the specific visual regions driving the ResNet-50 prediction.'
            '</div>',
            unsafe_allow_html=True,
        )

        cam_col1, cam_col2, cam_col3 = st.columns(3)

        with cam_col1:
            st.subheader("1. Original Image")
            st.image(create_preview_image(image, width=400, height=300), use_container_width=True)

        with cam_col2:
            st.subheader("2. Grad-CAM Heatmap")
            st.image(create_preview_image(heatmap_img, width=400, height=300), use_container_width=True)

        with cam_col3:
            st.subheader("3. Heatmap Overlay")
            st.image(create_preview_image(overlay_img, width=400, height=300), use_container_width=True)

        st.info(
            "💡 **Interpretation:** Red and yellow highlighted regions indicate the primary anatomical keypoints "
            "(e.g. beak shape, eye ring, plumage pattern, chest texture) that contributed most strongly to the model's classification."
        )

        # ----------------------------------------------------
        # 3. BIRD KNOWLEDGE INFORMATION SYSTEM
        # ----------------------------------------------------
        st.markdown('<div class="section-title">Species Field Guide & Details</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-subtitle">Automated field report for <b>{predicted_species}</b></div>',
            unsafe_allow_html=True,
        )

        info1, info2 = st.columns(2, gap="large")

        with info1:
            st.markdown(
                f'<div class="bird-info-card">'
                f'<div class="bird-field-title">Scientific Name</div>'
                f'<div class="bird-field-val"><i>{bird_data.get("scientific_name")}</i></div>'
                f'<div class="bird-field-title">Conservation Status</div>'
                f'<div class="bird-field-val">🟢 {bird_data.get("conservation_status")}</div>'
                f'<div class="bird-field-title">About the Species</div>'
                f'<div class="bird-field-val">{bird_data.get("short_description")}</div>'
                f'<div class="bird-field-title">Identification Features</div>'
                f'<div class="bird-field-val">{bird_data.get("identification_features")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with info2:
            st.markdown(
                f'<div class="bird-info-card">'
                f'<div class="bird-field-title">Habitat</div>'
                f'<div class="bird-field-val">{bird_data.get("habitat")}</div>'
                f'<div class="bird-field-title">Diet</div>'
                f'<div class="bird-field-val">{bird_data.get("diet")}</div>'
                f'<div class="bird-field-title">Geographic Distribution</div>'
                f'<div class="bird-field-val">{bird_data.get("geographic_distribution")}</div>'
                f'<div class="bird-field-title">Interesting Fact</div>'
                f'<div class="bird-field-val">✨ {bird_data.get("interesting_facts")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # 4. TOP 5 PROBABILITY DISTRIBUTION
        # ----------------------------------------------------
        st.markdown('<div class="section-title">Top 5 Prediction Probabilities</div>', unsafe_allow_html=True)

        for rank, (species, prob, _) in enumerate(predictions, start=1):
            l_col, r_col = st.columns([5, 1])
            with l_col:
                st.write(f"**{rank}. {species}**")
            with r_col:
                st.write(f"**{prob:.2f}%**")
            st.progress(min(prob / 100.0, 1.0))

        # ----------------------------------------------------
        # 5. REAL-WORLD ROBUSTNESS INTERACTIVE SIMULATOR
        # ----------------------------------------------------
        st.markdown('<div class="section-title">Real-World Robustness Simulator</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">'
            'Test how environmental degradation (shadows, blur, sensor noise, occlusion) affects model confidence and Grad-CAM attention.'
            '</div>',
            unsafe_allow_html=True,
        )

        with st.expander("🧪 Open Real-World Image Perturbation Sandbox", expanded=False):
            p_type = st.selectbox(
                "Select Environmental Perturbation:",
                [
                    ("original", "Clean / Unmodified"),
                    ("brightness_low", "Low Lighting / Shadow (0.5x)"),
                    ("brightness_high", "Harsh Sunlight / Overexposure (1.5x)"),
                    ("contrast_low", "Fog / Low Contrast (0.5x)"),
                    ("blur_gaussian", "Lens Blur (Gaussian)"),
                    ("noise_gaussian", "Digital Sensor Noise (High ISO)"),
                    ("rotation", "Camera Tilt (+15°)"),
                    ("scale_crop", "Distance Crop (Center 80%)"),
                    ("jpeg_compression", "JPEG Bandwidth Compression (Q=15)"),
                    ("occlusion", "Foliage Occlusion (30% Black Patch)"),
                ],
                format_func=lambda x: x[1],
            )

            perturbed_img = apply_perturbation(image, p_type[0])
            p_preds, p_heat, p_over = predict_and_explain(model, perturbed_img)

            sim_c1, sim_c2, sim_c3 = st.columns(3)
            with sim_c1:
                st.write("**Perturbed Image**")
                st.image(create_preview_image(perturbed_img, width=400, height=300), use_container_width=True)
            with sim_c2:
                st.write("**Grad-CAM Overlay**")
                st.image(create_preview_image(p_over, width=400, height=300), use_container_width=True)
            with sim_c3:
                st.write("**Perturbed Prediction**")
                st.metric("Predicted Species", p_preds[0][0])
                st.metric("Confidence", f"{p_preds[0][1]:.2f}%")


# ============================================================
# SUPPORTED SPECIES GRID
# ============================================================

st.markdown('<div class="section-title">Supported Bird Species</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">The 15 target Indian bird species recognized by this system.</div>',
    unsafe_allow_html=True,
)

species_cols = st.columns(3, gap="medium")
for index, species in enumerate(SELECTED_CLASSES):
    with species_cols[index % 3]:
        st.markdown(
            f"""
            <div class="species-card">
            {index + 1:02d} &nbsp;&nbsp; {species}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-text">
    <b>Taxonomic Classification Using Deep Learning with Grad-CAM & Robustness Analysis</b>
    <br>
    Indian Bird Species Classification using Fine-Tuned ResNet-50
    <br><br>
    PyTorch &nbsp; • &nbsp; TorchVision &nbsp; • &nbsp; Grad-CAM &nbsp; • &nbsp; Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)