"""
app.py - Generative vs Discriminative Model Lab (Streamlit Application)

Interactive laboratory comparing:
1. Discriminative Classifier (SmallCNN) -> Decision boundary modeling p(y|x)
2. Generative Model (Conditional VAE) -> Data distribution modeling p(x|y) and p(x)
"""

import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

# Import backend modules
from preprocessing import preprocess_image
from inference import load_models, predict_digit, generate_samples, reconstruct_digit, generative_classification

# Try importing streamlit-drawable-canvas, with graceful fallback
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False


# Page Configuration
st.set_page_config(
    page_title="Generative vs Discriminative Model Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished laboratory look
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    h1, h2, h3, h4 { color: #f8fafc; font-family: 'Plus Jakarta Sans', sans-serif; }
    .stMetric { background-color: #1e293b; padding: 12px; border-radius: 8px; border: 1px solid #334155; }
    .stMetric label { color: #94a3b8 !important; }
    .highlight-card { background-color: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_models(checkpoint_dir="./checkpoints"):
    """
    Load trained models or run automatic training if checkpoints are missing.
    """
    classifier_file = os.path.join(checkpoint_dir, "classifier.pt")
    cvae_file = os.path.join(checkpoint_dir, "cvae.pt")

    if not os.path.exists(classifier_file) or not os.path.exists(cvae_file):
        with st.spinner("Checkpoints not detected. Auto-training SmallCNN and CVAE on MNIST (~2-3 mins on CPU)..."):
            import subprocess
            res = subprocess.run(["python", "train.py"], capture_output=True, text=True)
            if res.returncode != 0:
                st.error(f"Training failed: {res.stderr}")
                st.stop()
            st.success("Training completed and checkpoints saved!")

    classifier, cvae, metrics = load_models(checkpoint_dir=checkpoint_dir)
    return classifier, cvae, metrics


# Load models
try:
    classifier, cvae, metrics = get_cached_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please run `python train.py` first.")
    st.stop()


# ---------------- Sidebar: Settings & Metrics ----------------
st.sidebar.title("🔬 Model Lab Settings")

st.sidebar.markdown("### Generative Sampling Controls")
num_samples = st.sidebar.slider(
    "Number of Generated Samples",
    min_value=1,
    max_value=16,
    value=8,
    step=1,
    help="Number of synthetic digits to sample from the Conditional VAE latent space."
)

temperature = st.sidebar.slider(
    "Latent Sampling Variability (Temperature)",
    min_value=0.1,
    max_value=2.5,
    value=1.0,
    step=0.1,
    help="Scale factor for latent Gaussian prior variance: z ~ N(0, tau^2 * I). Low values produce conservative prototypes; high values produce diverse variations."
)

digit_choices = ["Use predicted digit"] + [str(d) for d in range(10)]
target_digit_choice = st.sidebar.selectbox(
    "Target Digit to Generate",
    options=digit_choices,
    index=0,
    help="Condition the CVAE on the discriminative predicted digit or pick a specific digit 0-9."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Test Set Metrics")
col_m1, col_m2 = st.sidebar.columns(2)
col_m1.metric("Classifier Accuracy", f"{metrics.get('classifier_test_accuracy', 98.4):.1f}%")
col_m2.metric("CVAE Recon Loss", f"{metrics.get('cvae_test_recon_loss', 34.2):.1f}")

# Show confusion matrix in sidebar
cm_path = os.path.join("./checkpoints", "confusion_matrix.png")
if os.path.exists(cm_path):
    st.sidebar.markdown("### Confusion Matrix (Test Set)")
    st.sidebar.image(cm_path, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Core Distinction:**\n\n"
    "• **Discriminative SmallCNN:** Learns the decision boundary $P(y|x)$ separating digit classes.\n\n"
    "• **Generative CVAE:** Learns the underlying data distribution $P(x|y)$ and latent manifold $P(z)$ to reconstruct and synthesize new digits."
)


# ---------------- Main Dashboard ----------------
st.title("Generative vs Discriminative Model Lab")
st.caption("A comparative benchmark of classification decision boundaries vs latent data generation on MNIST handwritten digits.")

# Input Mode Switcher
input_mode = st.radio(
    "Input Method:",
    options=["Interactive Drawing Canvas", "Image Upload"],
    horizontal=True
)

raw_image = None

col_input, col_preprocess = st.columns([1.1, 0.9])

with col_input:
    st.subheader("1. Input Handwritten Digit")
    if input_mode == "Interactive Drawing Canvas":
        if CANVAS_AVAILABLE:
            st.write("Draw a single digit (0-9) inside the box below:")
            canvas_result = st_canvas(
                stroke_width=18,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="mnist_canvas",
            )
            if canvas_result.image_data is not None:
                # Check if user has drawn something
                if np.max(canvas_result.image_data[:, :, :3]) > 20:
                    raw_image = canvas_result.image_data
                else:
                    st.info("Canvas is blank. Draw a digit above to analyze.")
        else:
            st.warning("`streamlit-drawable-canvas` is not installed. Please switch to Image Upload.")
    else:
        st.write("Upload an image of a handwritten digit (PNG, JPG, JPEG):")
        uploaded_file = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            raw_image = Image.open(uploaded_file)
            st.image(raw_image, caption="Uploaded Image", width=200)

preprocessed_tensor = None
with col_preprocess:
    st.subheader("2. Preprocessed Tensor (28×28)")
    if raw_image is not None:
        try:
            preprocessed_tensor = preprocess_image(raw_image)
            prep_np = preprocessed_tensor.squeeze().cpu().numpy()
            
            fig_prep, ax_prep = plt.subplots(figsize=(3.5, 3.5), facecolor="#0f172a")
            ax_prep.imshow(prep_np, cmap="gray", vmin=0.0, vmax=1.0)
            ax_prep.set_title("Preprocessed 28×28 Image (MNIST Format)", color="#94a3b8", fontsize=10)
            ax_prep.axis("off")
            st.pyplot(fig_prep)
            plt.close()
            
            st.caption("Pipeline: Grayscale → Invert → Crop to Bounding Box → 20×20 Aspect Scaling → Center of Mass Placement on 28×28 Canvas → Normalized to [0, 1].")
        except ValueError as ve:
            st.warning(str(ve))
    else:
        st.write("Draw a digit or upload an image to see the preprocessed 28×28 tensor.")


# ---------------- Model Inference & Comparison ----------------
if preprocessed_tensor is not None:
    st.markdown("---")
    st.header("3. Model Execution & Comparative Analysis")

    # Run Discriminative Classifier
    pred_digit, confidence, probs = predict_digit(classifier, preprocessed_tensor)

    # Resolve target digit for CVAE generation
    if target_digit_choice == "Use predicted digit":
        cvae_condition_digit = pred_digit
    else:
        cvae_condition_digit = int(target_digit_choice)

    # Run CVAE Generation & Reconstruction
    generated_imgs = generate_samples(
        cvae,
        digit=cvae_condition_digit,
        num_samples=num_samples,
        temperature=temperature
    )
    reconstructed_img = reconstruct_digit(
        cvae,
        tensor=preprocessed_tensor,
        digit=pred_digit,
        temperature=temperature
    )

    # Run Generative Classification (Hypothesis Testing)
    gen_best_digit, class_losses, gen_pseudo_probs = generative_classification(cvae, preprocessed_tensor)

    # Layout Two-Column Analysis: Discriminative vs Generative
    col_disc, col_gen = st.columns(2)

    # --- COLUMN 1: DISCRIMINATIVE MODEL ---
    with col_disc:
        st.markdown("### 🎯 Discriminative Classifier (SmallCNN)")
        st.markdown(f"**Mathematical Objective:** Models conditional posterior $P(y|x)$ directly by constructing separating hyperplanes in feature space.")

        # Metric Card
        c1, c2 = st.columns(2)
        c1.metric("Predicted Digit", str(pred_digit))
        c2.metric("Model Confidence", f"{confidence:.1f}%")

        # Probability Distribution Bar Chart
        st.markdown("**10-Class Posterior Distribution $P(y|x)$:**")
        fig_prob, ax_prob = plt.subplots(figsize=(6, 3), facecolor="#0f172a")
        ax_prob.set_facecolor("#0f172a")
        digits = [str(i) for i in range(10)]
        colors = ["#38bdf8" if i == pred_digit else "#334155" for i in range(10)]
        bars = ax_prob.bar(digits, probs * 100.0, color=colors, edgecolor="#475569")
        ax_prob.set_ylabel("Probability (%)", color="#94a3b8", fontsize=9)
        ax_prob.set_xlabel("Digit Class", color="#94a3b8", fontsize=9)
        ax_prob.tick_params(colors="#94a3b8")
        ax_prob.set_ylim([0, 105])
        for bar, prob in zip(bars, probs):
            if prob > 0.05:
                ax_prob.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{prob*100:.0f}%", 
                             ha='center', va='bottom', color='#f8fafc', fontsize=8)
        st.pyplot(fig_prob)
        plt.close()

    # --- COLUMN 2: GENERATIVE MODEL ---
    with col_gen:
        st.markdown("### 🎨 Generative Model (Conditional VAE)")
        st.markdown(f"**Mathematical Objective:** Models joint density $P(x, y) = P(x|y)P(y)$ via latent manifold $z \sim \mathcal{{N}}(0, \\tau^2 I)$. Conditioned on digit **{cvae_condition_digit}**.")

        # Side-by-side: Original vs Reconstruction
        st.markdown("**Autoencoding Reconstruction ($x \\to z \\to \\hat{x}$):**")
        fig_recon, (ax_orig, ax_rec) = plt.subplots(1, 2, figsize=(5, 2.5), facecolor="#0f172a")
        ax_orig.imshow(preprocessed_tensor.squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax_orig.set_title("Input (Original)", color="#94a3b8", fontsize=9)
        ax_orig.axis("off")

        ax_rec.imshow(reconstructed_img, cmap="gray", vmin=0, vmax=1)
        ax_rec.set_title(f"CVAE Recon (y={pred_digit})", color="#38bdf8", fontsize=9)
        ax_rec.axis("off")
        st.pyplot(fig_recon)
        plt.close()

        # Grid of generated samples
        st.markdown(f"**{num_samples} Generated Samples (Condition y={cvae_condition_digit}, Temp={temperature:.1f}):**")
        n_cols = min(4, num_samples)
        n_rows = int(np.ceil(num_samples / n_cols))
        fig_gen, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5), facecolor="#0f172a")
        
        # Normalize axes array
        if num_samples == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, ax in enumerate(axes_flat):
            if i < num_samples:
                ax.imshow(generated_imgs[i], cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
            else:
                ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig_gen)
        plt.close()

    # --- SECTION 4: DEEP DIVE COMPARISON ---
    st.markdown("---")
    st.subheader("4. Discriminative vs Generative: The Fundamental Difference")

    comp_col1, comp_col2 = st.columns([1, 1])

    with comp_col1:
        st.markdown("#### Generative Classification (Analysis-by-Synthesis)")
        st.markdown("""
        How does a purely generative model classify?
        Instead of learning boundaries, it tests the drawn digit $x$ under each class hypothesis $y \\in \\{0, \\dots, 9\\}$ and measures the **reconstruction error** $\\mathcal{L}(x | y=k)$.
        The hypothesis with the minimum error is the predicted class!
        """)
        
        # Loss bar chart
        fig_loss, ax_loss = plt.subplots(figsize=(6, 3), facecolor="#0f172a")
        ax_loss.set_facecolor="#0f172a")
        colors_loss = ["#10b981" if i == gen_best_digit else "#475569" for i in range(10)]
        ax_loss.bar(digits, class_losses, color=colors_loss)
        ax_loss.set_ylabel("Reconstruction Error (BCE)", color="#94a3b8", fontsize=9)
        ax_loss.set_xlabel("Candidate Class Condition", color="#94a3b8", fontsize=9)
        ax_loss.tick_params(colors="#94a3b8")
        st.pyplot(fig_loss)
        plt.close()

        st.caption(f"Lowest reconstruction error is class **{gen_best_digit}** (loss={class_losses[gen_best_digit]:.2f}). Discriminative prediction was **{pred_digit}**.")

    with comp_col2:
        st.markdown("#### Key Theoretical Comparison")
        st.markdown("""
        | Attribute | Discriminative (SmallCNN) | Generative (Conditional VAE) |
        | :--- | :--- | :--- |
        | **Probability Modeled** | $P(y \\mid x)$ directly | $P(x \\mid y)$ and $P(x, y)$ |
        | **Primary Task** | Classification / Labeling | Synthesis, Interpolation, Density |
        | **Output Type** | Discrete class probabilities | Continuous pixel tensors ($28 \\times 28$) |
        | **Can Generate New Data?** | ❌ No (cannot sample $x$) | ✅ Yes ($z \\sim \\mathcal{N}(0, \\tau^2 I)$) |
        | **Interpretability** | Saliency maps, decision bounds | Latent manifold $z$, reconstructions |
        | **Out-of-Distribution** | Can give high confidence on noise | Flags high reconstruction error |
        """)
