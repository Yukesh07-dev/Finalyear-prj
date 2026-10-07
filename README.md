# Indian Bird Species Classification with Explainable AI & Robustness Analysis

An upgraded deep learning system for **15 Indian Bird Species Classification**, featuring **Explainable AI (Grad-CAM)**, **Real-World Environmental Robustness Testing**, and an **Integrated Species Field Guide Knowledge System**.

---

## 📌 Project Overview

This project enhances an existing deep learning image-classification system trained on 15 Indian bird species using fine-tuned ResNet architectures. The original research compared frozen ResNet-50, fine-tuned ResNet-50, and fine-tuned ResNet-152 models.

The upgraded version adds three core academic and practical contributions:
1. **Explainable AI (Grad-CAM)**: Visualizes model attention maps to reveal the exact visual features (beak, eye rings, plumage, throat patterns) driving the neural network's predictions.
2. **Real-World Environmental Robustness Evaluation**: Empirically measures model performance degrades under realistic photographic conditions (low lighting, glare, lens blur, digital sensor noise, camera tilt, scaling, JPEG compression, foliage occlusion).
3. **Integrated Species Field Guide**: Provides detailed offline ornithological records (scientific name, description, habitat, diet, identification traits, conservation status, interesting facts) for predicted species within the Streamlit UI.

---

## 🎯 Supported Bird Species (15 Target Classes)

1. Asian Green Bee-Eater
2. Brown-Headed Barbet
3. Cattle Egret
4. Common Kingfisher
5. Common Myna
6. Common Rosefinch
7. Common Tailorbird
8. Coppersmith Barbet
9. Forest Wagtail
10. Gray Wagtail
11. Hoopoe
12. House Crow
13. Indian Grey Hornbill
14. Indian Peacock
15. Indian Pitta

---

## 🏗️ Project Architecture & File Structure

```
c:\FinalYearPrj/
├── app/
│   └── app.py                     # Upgraded Streamlit web application
├── checkpoints/
│   ├── finetuned_resnet50/        # Deployed model weights (best_model.pth)
│   └── finetuned_resnet152/       # ResNet-152 model weights
├── data/
│   ├── bird_info.json             # Factual database for all 15 bird species
│   └── splits/
│       ├── train.csv              # Training set split (80%)
│       └── validation.csv         # Validation set split (20%, 4,443 images)
├── experiments/
│   └── robustness_results/        # Output JSON reports & comparison plots
├── src/
│   ├── bird_info.py               # Field guide loader module
│   ├── config.py                  # Dataset paths & hyperparameter configuration
│   ├── dataset.py                 # PyTorch custom Dataset class
│   ├── evaluate.py                # Model evaluation pipeline
│   ├── gradcam.py                 # Pure PyTorch Grad-CAM explainability module
│   ├── models.py                  # ResNet-50 / ResNet-152 model definitions
│   ├── predict.py                 # Command-line single image inference
│   ├── robustness.py              # 10-condition perturbation evaluation engine
│   ├── train.py                   # Model training script
│   ├── trainer.py                 # Training loop controller
│   └── transforms.py              # Preprocessing & robust augmentation pipelines
├── requirements.txt               # Dependencies list
└── README.md                      # Project documentation
```

---

## 🚀 How to Run the Application

### 1. Installation Requirements

Install required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch Streamlit Web Application

Run the Streamlit application:
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Running Robustness Experiments

To execute the empirical 10-condition robustness evaluation on the full validation dataset (4,443 images):

```bash
python -m src.robustness
```

### Empirical Validation Results (Fine-Tuned ResNet-50)

| Environmental Condition | Accuracy (%) | Weighted F1 (%) | Performance Impact |
| :--- | :---: | :---: | :--- |
| **Clean Baseline (Original)** | **97.73%** | **97.73%** | Benchmark Baseline |
| **High Contrast (Glare)** | 96.06% | 96.06% | Minimal (-1.67%) |
| **Low Brightness (Shadows)** | 95.90% | 95.89% | Minor (-1.83%) |
| **15° Rotation (Camera Tilt)** | 94.76% | 94.75% | Minor (-2.97%) |
| **High Brightness (Sunlight)** | 94.62% | 94.62% | Minor (-3.11%) |
| **Scale / Center Crop (80%)** | 93.63% | 93.63% | Moderate (-4.10%) |
| **JPEG Compression (Q=15)** | 91.13% | 91.11% | Moderate (-6.60%) |
| **Low Contrast (Fog/Haze)** | 89.65% | 89.67% | Moderate (-8.08%) |
| **Sensor Noise (High ISO)** | 84.72% | 84.66% | Significant (-13.01%) |
| **Gaussian Blur (Out-of-Focus)** | 81.36% | 81.25% | Severe (-16.37%) |
| **30% Occlusion (Foliage)** | 78.48% | 78.43% | Severe (-19.25%) |

---

## 🧠 Explainable AI (Grad-CAM) Methodology

Grad-CAM computes the gradient of the predicted score for class $c$ ($y^c$) with respect to feature map activations $A^k$ of the final convolutional layer (`model.layer4[-1]`):

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right)$$

- **ReLU filtering**: Retains features that positively correlate with target species.
- **Hook Cleanup**: Hooks are un-registered after each inference to guarantee 0 memory leaks across Streamlit sessions.

---

## 🎓 Viva Questions & Answers ("WHAT I CAN SAY IN MY VIVA")

### Q1: Why was Grad-CAM added to the project?
> **Answer**: Standard deep learning models act as "black boxes" providing only a class name and probability score. Grad-CAM adds explainability by generating visual attention heatmaps that show *why* the model made a prediction, proving that it focuses on actual avian anatomical features (beak shape, crest, plumage texture) rather than background noise.

### Q2: What is the target layer used for Grad-CAM in ResNet-50?
> **Answer**: Grad-CAM targets `model.layer4[-1]`, which is the final residual block in ResNet-50. This layer preserves spatial location information while encoding high-level semantic concepts.

### Q3: Why is real-world robustness testing important?
> **Answer**: Clean validation accuracy (e.g. 97.73%) evaluates ideal photographs, but real-world field conditions involve shadows, out-of-focus lenses, sensor noise, atmospheric haze, and foliage occlusion. Robustness testing measures how gracefully the model degrades under these degradation conditions.

### Q4: What were the most severe degradation conditions identified?
> **Answer**: Partial foliage occlusion (78.48% accuracy) and out-of-focus Gaussian blur (81.36% accuracy) caused the largest drop in performance, showing that spatial continuity of key anatomical regions is critical for CNN feature extraction.

### Q5: Why was the species count kept strictly at 15?
> **Answer**: The project focuses on depth rather than superficial species count. By maintaining the 15 benchmark Indian species, we could rigorously analyze model explainability and environmental robustness while preserving trained weight checkpoints.

### Q6: Why did you retain Fine-Tuned ResNet-50 as the primary model?
> **Answer**: ResNet-50 provides an optimal trade-off between inference speed, memory efficiency, and accuracy (97.73%), outperforming frozen baselines while requiring fewer computational resources than ResNet-152 during web app deployment.
