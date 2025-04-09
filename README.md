# Real-Time-Image-Dehazing-Using-Deep-Learning-PyTorch-
Developed a real-time image dehazing pipeline using a custom U-Net-based deep learning model in PyTorch. Enabled fast inference via pretrained weights without requiring ground truth during deployment. Built with OpenCV, Matplotlib, and Google Colab.
This project implements a real-time image dehazing pipeline using a custom U-Net-style architecture in **PyTorch**. The model is trained on paired hazy and clean images and supports pretrained weight loading for fast and effective deployment — no ground truth required during inference.

## 🚀 Features
- U-Net-based custom model for single image dehazing
- Inference-only mode using pretrained weights (`.pth`)
- Visual comparison of hazy vs dehazed outputs using Matplotlib
- Optional image quality metrics: PSNR and SSIM
- Lightweight and portable: runs in Google Colab or locally

## 🧪 Example
<p align="center">
  <img src="example/hazy.jpg" width="45%"/> ➡️ <img src="example/dehazed.jpg" width="45%"/>
</p>

## 🛠 Tech Stack
- `Python`
- `PyTorch`
- `OpenCV`
- `Matplotlib`
- `Google Colab`

## 📁 Folder Structure
├── proper_final.py # Model architecture ├── best1.pth # Pretrained weights ├── inference.py # Script for testing custom hazy images


## 📦 Usage
```bash
# Load model and run inference
python inference.py --image_path test_hazy.jpg --weights best1.pth

