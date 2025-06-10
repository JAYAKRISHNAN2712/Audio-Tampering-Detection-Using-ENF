# Digital audio tampering detection based on spatio-temporal representation learning of electrical network frequency

This repository contains the implementation of the method proposed in the research paper titled:

**"Digital audio tampering detection based on spatio-temporal representation learning of electrical network frequency"**

## 📖 Abstract

This work proposes a digital audio tampering detection technique by learning the spatio-temporal representation of Electric Network Frequency (ENF) signals embedded in audio recordings. The method extracts phase sequences of ENF using high-precision DFT analysis. Unequal phase sequences are adaptively framed:

- To form fixed-size matrices representing **spatial features**, and
- To capture ENF timing changes representing **temporal features**.

A parallel **CNN-BiLSTM** architecture is used, where:

- **CNN** extracts deep spatial features,
- **BiLSTM** learns temporal dependencies,
- An **attention mechanism** fuses the features,
- A final **MLP** classifies the audio as tampered or authentic.

Experiments on the **Carioca** and **New Spanish** datasets show that our approach outperforms previous methods in detection accuracy.

---

## 🧠 Highlights

- ✅ High-precision ENF phase extraction using DFT
- 🧩 Adaptive framing for uniform spatio-temporal analysis
- 🧠 Deep learning model combining CNN, BiLSTM, and attention mechanism
- 🏆 Superior performance on benchmark datasets
- 📌 Focus on both **tampering detection** and **future localization**

---

## 📂 Project Structure
```
📁 Digital-audio-tampering-detection-based-on-spatio-temporal-representation-learning-of-ENF
├── 📁 codebase               # Source Code
│   ├── 📁 model_store
│   │    ├── model.pth        # Pre-trained model
│   │    ├── params.json      # Hyper parameters
│   ├── inference.py          # Inference pipeline
│   ├── models.py             # CNN + BiLSTM + Attention architecture
│   ├── train.py              # Training pipeline
│   └── utils.py              # DFT-based ENF phase extraction utilities
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```
## Installation
```
pip install -r requirements.txt
```
## Training
```
cd codebase
python3 train.py -i /path/to/audio/files -b batch_size -e epochs
```
## Inference
```
cd codebase
python3 inference.py -i /path/to/test/audio.wav
```
## 📝 Acknowledgments

This implementation is based on the research paper:

> **Digital audio tampering detection based on spatio-temporal representation learning of electrical network frequency**  
> [Authors: Chunyan Zeng, Shuai Kong, Zhifeng Wang, Xiangkui Wan, and Yunfan Chen]    

We thank the authors for their valuable contribution to the field of multimedia forensics.  
All core concepts, including ENF-based phase analysis and the CNN-BiLSTM-attention model, are derived from the ideas presented in this work.

📄 [https://doi.org/10.1007/s11042-024-18887-5]


