# Anime Face Generation using Generative Adversarial Networks (GAN)

This repository contains an implementation of a Generative Adversarial Network (GAN) designed to generate synthetic anime-style faces. The project includes a fully customizable GAN architecture for training on custom datasets and generating realistic outputs.

The original dataset can be found in this [GitHub repository](https://github.com/bchao1/Anime-Face-Dataset?tab=readme-ov-file).

---

## Project Files

### **1. model.py**
- Contains a basic generator and discriminator architecture with 5 convolutional layers.
- Parameters such as hidden layer dimensions and `kernel_size` can be customized.

---

### **2. main.py**
- The main script for training the GAN model.
- Key parameters:
  - `--data`: Path to the dataset.
  - `--epochs`: Number of epochs to train (default: 20).
  - `--lr`: Learning rate for both discriminator and generator (default: 0.002). Includes an exponential decay scheduler.
  - `--batch-size`: Batch size for training (default: 64).
  - `--output`: Directory to save the trained model (default: `output_generator`).
  - `--latent-size`: Size of the noise vector for the generator (default: 100).

- **Customizations**:
  - Optimizer: Default is Adam optimizer.
  - Loss function: Default is Binary Cross-Entropy (BCE) loss.

---

### **3. generate_images.py**
- Script to generate synthetic images using the trained generator.
- Key parameters:
  - `--model`: Path to the trained generator model file (in `.pth` format).
  - `--output`: Directory to save generated images.
  - `--number`: Number of images to generate.
  - `--latent-size`: Size of the latent vector the generator was trained with.

---

## Training and Generation Guide

### **Training the GAN**
Run the following command in the directory containing `main.py`:

```bash
python main.py --data <dataset_path> --epochs 40 --batch-size 64 --output model64 --latent-size 200
```

### ***Generating anime faces***
Run the following command in the directory containing `generate_images.py`:

```bash
python generate_images.py --model model64/generator.pth --output generated_images --number 20
```
=> Generates 20 anime faces


### Planned improvements:
Conditional GANs: Introduce conditioning to generate more specific faces based on additional input features
