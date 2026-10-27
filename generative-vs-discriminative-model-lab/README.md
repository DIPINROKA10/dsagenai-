# Generative vs Discriminative Model Lab (MNIST)

A complete, locally-runnable machine learning laboratory comparing **Discriminative** and **Generative** modeling paradigms using Python, PyTorch, MNIST, and Streamlit.

---

## 1. Project Overview

In machine learning:
- **Discriminative models** model the posterior distribution $P(y \mid x)$ directly. They learn a decision boundary between classes to predict a category label given input features.
- **Generative models** model the data distribution $P(x \mid y)$ (or the joint distribution $P(x, y) = P(x \mid y)P(y)$). They learn the underlying density of the data and can synthesize novel, diverse samples.

This project implements both paradigms on the classic MNIST handwritten digits dataset with real PyTorch neural networks running entirely locally on CPU/GPU.

---

## 2. Folder Structure

```
generative-vs-discriminative-lab/
│
├── app.py                  # Main Streamlit web application frontend
├── preprocessing.py        # Canvas/upload to 28x28 normalized MNIST tensor
├── models.py               # SmallCNN (Classifier) and ConditionalVAE (CVAE) architectures
├── train.py                # Training pipeline for both models, metrics, and confusion matrix
├── inference.py            # Model loading, prediction, sample generation, & CVAE classification
├── requirements.txt        # Python dependency specification
├── README.md               # Complete documentation and setup guide
│
├── checkpoints/            # Directory where trained weights and metrics are stored
│   ├── classifier.pt       # PyTorch checkpoint for SmallCNN (>98% test accuracy)
│   ├── cvae.pt             # PyTorch checkpoint for Conditional VAE
│   ├── metrics.json        # Test set accuracy, reconstruction loss, and metadata
│   └── confusion_matrix.png# Pre-computed test set confusion matrix
│
└── data/                   # Auto-downloaded MNIST dataset directory
```

---

## 3. Quick Setup & Local Execution

### Step 1: Create Virtual Environment
```bash
# Clone or navigate to the project directory
cd generative-vs-discriminative-lab

# Create Python virtual environment (Python 3.9 - 3.12 recommended)
python3 -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows (Command Prompt):
venv\Scripts\activate
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Train Models (CPU takes ~3-4 minutes)
```bash
python train.py
```
*Note: If you run `streamlit run app.py` without training first, the application will automatically detect missing checkpoints and execute training for you with a progress spinner.*

### Step 4: Run the Streamlit Application
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 4. Model Architectures & Theory

### 1. Discriminative Model: `SmallCNN`
- **Objective:** Learn $P(y \mid x)$ (maps image $x \in \mathbb{R}^{28 \times 28}$ to class logits $y \in \{0, \dots, 9\}$).
- **Architecture:**
  - `Conv2d(1, 32, kernel_size=3, padding=1)` $\to$ `ReLU` $\to$ `MaxPool2d(2, 2)`
  - `Conv2d(32, 64, kernel_size=3, padding=1)` $\to$ `ReLU` $\to$ `MaxPool2d(2, 2)`
  - `Dropout(0.25)` $\to$ `Flatten(3136)`
  - `Linear(3136, 128)` $\to$ `ReLU` $\to$ `Dropout(0.5)`
  - `Linear(128, 10)` (Output logits)
- **Loss:** Cross-Entropy Loss
- **Test Set Performance:** $>98.4\%$ accuracy on MNIST test set.

### 2. Generative Model: `ConditionalVAE` (CVAE)
- **Objective:** Learn $P(x \mid y)$ by maximizing the Evidence Lower Bound (ELBO):
  $$\log P(x \mid y) \ge \mathbb{E}_{q(z \mid x, y)} [\log p(x \mid z, y)] - D_{KL}(q(z \mid x, y) \parallel p(z))$$
- **Conditioning:** Both the encoder and decoder are conditioned on a 10-dimensional one-hot vector $y$.
- **Latent Dimension:** $z \in \mathbb{R}^{16}$.
- **Encoder:** $[x_{\text{flat}} (784) + y (10)] \to 512 \to 256 \to (\mu, \log \sigma^2)$.
- **Reparameterization with Temperature $\tau$:**
  $$z = \mu + \tau \cdot \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
- **Decoder:** $[z (16) + y (10)] \to 256 \to 512 \to 784 \to \text{Sigmoid}$.
- **Reconstruction Loss:** Binary Cross Entropy (BCE).

### 3. Generative Classification (Analysis-by-Synthesis)
To highlight the fundamental contrast, the lab demonstrates how a generative model performs classification without a discriminant boundary:
$$\hat{y}_{\text{gen}} = \arg\min_{k \in \{0, \dots, 9\}} \mathcal{L}_{\text{recon}}(x \mid y=k)$$
The digit image is evaluated under each candidate label. The label that yields the smallest reconstruction error is selected.

---

## 5. UI Controls & Features

| Control | Description |
| :--- | :--- |
| **Input Method Toggle** | Switch between the interactive drawing canvas (`streamlit-drawable-canvas`) and file uploader (PNG/JPG). |
| **Drawing Canvas** | Draw custom handwritten digits with smooth 18px brush. Auto-inverted if necessary. |
| **Number of Generated Samples** | Slider (1 to 16) controlling how many synthetic digits to sample from the CVAE. |
| **Latent Variability (Temperature $\tau$)** | Slider (0.1 to 2.5) scaling latent noise $z \sim \mathcal{N}(0, \tau^2 I)$. Lower values generate conservative standard digits; higher values explore creative stroke variations. |
| **Target Digit Selector** | Choose "Use predicted digit" or force any digit (0–9) to test conditional synthesis. |
| **Preprocessed 28×28 Tensor** | Shows exact centered image the neural networks see (bounding box cropped, 20×20 aspect scaling, center-of-mass aligned). |
| **Confidence Bar Chart** | Interactive 10-class probability distribution $P(y \mid x)$ from SmallCNN. |
| **Side-by-Side Reconstruction** | Visualizes input image vs CVAE reconstructed output $x \to z \to \hat{x}$. |
| **Generated Sample Grid** | Dynamic grid of $N$ synthetic digits generated for the target class. |
| **Generative Hypothesis Testing** | Bar chart of reconstruction loss under each digit hypothesis $0..9$. |
| **Confusion Matrix** | Pre-computed 10×10 test set confusion matrix displayed in the sidebar. |

---

## 6. Screenshots & Submission Evidence

Place exam screenshots below:

### Screenshot 1: Drawing Canvas & Classification Prediction
*(Capture drawn digit, preprocessed 28x28 tensor, and 10-class probability bar chart)*
`[Insert Screenshot Here]`

### Screenshot 2: CVAE Reconstruction & Generated Samples Grid
*(Capture original vs reconstructed digit and grid of samples at temperature 1.0)*
`[Insert Screenshot Here]`

### Screenshot 3: Temperature Variation Comparison
*(Capture samples generated at low temperature 0.3 vs high temperature 1.8)*
`[Insert Screenshot Here]`

### Screenshot 4: Discriminative vs Generative Classification Comparison
*(Capture the side-by-side comparison section showing p(y|x) vs per-class reconstruction loss)*
`[Insert Screenshot Here]`

### Screenshot 5: Sidebar Metrics & Test Set Confusion Matrix
*(Capture sidebar showing >98% accuracy, CVAE reconstruction loss, and confusion matrix)*
`[Insert Screenshot Here]`

---

## 7. Troubleshooting & Edge Cases

- **Empty Canvas:** The preprocessing pipeline detects blank strokes and informs the user via `st.info` instead of throwing an unhandled exception.
- **Inverted Colors:** Uploaded images with dark strokes on light paper are automatically detected and inverted so digits are white on black, as required by MNIST.
- **Off-Center Strokes:** The preprocessing module calculates the center of mass and translates the digit to $(14, 14)$ coordinates inside the 28×28 tensor.
- **Canvas Package Issue:** If `streamlit-drawable-canvas` encounters compatibility issues on newer Streamlit versions, the app provides a graceful notice and allows seamless digit testing via image upload.
