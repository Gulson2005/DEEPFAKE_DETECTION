🧠 Deepfake Detection using EfficientNetB4 + Attention (Celeb-DF-v2)
A robust deepfake image detection model built on the Celeb-DF-v2 dataset using a fine-tuned EfficientNetB4 backbone enhanced with attention mechanisms, mixed precision training, and advanced augmentation strategies. This project focuses on identifying fake vs real faces in images using state-of-the-art computer vision techniques.

📁 Dataset
Source: Celeb-DF-v2 Dataset

Structure:
Celeb-DF-v2/
└── Celeb_V2/
    ├── Train/
    │   ├── fake/
    │   └── real/
    ├── Val/
    │   ├── fake/
    │   └── real/
    └── Test/
        ├── fake/
        └── real/
🚀 Features
🏗️ Base Model: EfficientNetB4 (pre-trained on ImageNet)

🌌 Custom Architecture Enhancements:

Dual Global Pooling (Average + Max)

Residual Connections

Attention-based Pooling

🎯 Loss Function: Binary Crossentropy with Label Smoothing

🧠 Optimizers: AdamW (with fallback to Adam)

⚖️ Class Weights: To counter dataset imbalance

🧪 Face Detection Preprocessing:

Haar Cascade

Fallback: Skin Detection

Cropped and aligned faces

🔄 Advanced Augmentation:

Horizontal flips, rotations, color jitter, contrast/brightness shift, CutMix

🧬 Training Strategy:

Phase 1: Frozen base model

Phase 2: Fine-tuning all layers

🧼 Data Pipeline:

tf.data.Dataset with caching, prefetching

Optimized generators

🔍 Precision:

Mixed Precision Training for TPU optimization

📉 Callbacks:

Early Stopping

Model Checkpointing

ReduceLROnPlateau

🧪 Evaluation Metrics
✅ Accuracy

🔁 AUC

🔍 Precision / Recall / F1-Score

📉 Confusion Matrix

🔥 ROC & PR Curves

🖼 Sample Inference

Input Image	Prediction
✅ Real
❌ Fake
🛠 Tech Stack
Python

TensorFlow / Keras

OpenCV

NumPy / Pandas

Matplotlib / Seaborn

📦 Setup
Clone Repo

git clone https://github.com/your-username/deepfake-detector.git
cd deepfake-detector
Install Dependencies

pip install -r requirements.txt
Run Training


python train.py
Inference

python predict.py --image path_to_image.jpg
📊 Results
✅ Achieved XX% Accuracy, YY AUC on validation set

💡 Model shows strong generalization across varied facial images

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

📜 License
MIT License

💬 Acknowledgements
Celeb-DF-v2

EfficientNet

TensorFlow Team & Kaggle TPU Runtime

