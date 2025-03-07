# DCGAN on LSUN Bedrooms Dataset

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** following the paper *"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"* by Radford et al. (2015). The model is trained on the **LSUN Bedrooms dataset** to generate realistic bedroom images.

---

## 📌 Objective

The goal is to train a **DCGAN** to generate high-quality bedroom images using adversarial training. The network consists of a **Generator** and a **Discriminator**, both based on convolutional layers without fully connected layers to improve performance and stability.

---

## 📂 Dataset

We use the **LSUN Bedrooms** dataset, loaded from Hugging Face datasets.

### ✅ Dataset Preprocessing
- Resize images to `64x64` pixels.
- Apply **center cropping** for better composition.
- Convert images to tensors and normalize them to `[-1, 1]`.

---

## 🏗️ Model Architecture

### 🔹 Generator
- Takes a **random noise vector (latent space, `nz=100`)** as input.
- Uses **transposed convolutions** to upsample and generate realistic images.
- **Batch normalization** and **ReLU activations** improve training stability.
- Outputs a `3x64x64` RGB image using a **Tanh activation function**.

### 🔹 Discriminator
- Takes a `3x64x64` image as input.
- Uses **convolutions with LeakyReLU activations** to classify real vs. fake images.
- **Batch normalization** stabilizes training.
- Outputs a single **probability value (Sigmoid activation)**.

---

## ⚙️ Training Details

- **Loss Function**: Binary Cross Entropy Loss (**BCELoss**).
- **Optimizers**: Adam Optimizer (`lr=0.0002, β1=0.5`).
- **Batch Size**: `128`.
- **Epochs**: `15`.
- **Fixed Noise**: Used to visualize progress during training.

---
