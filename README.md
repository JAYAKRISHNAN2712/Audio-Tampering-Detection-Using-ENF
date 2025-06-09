# Digital Audio Tampering Detection Based on ENF Spatio-temporal Features Representation Learning

This repository contains the implementation of the method proposed in the research paper titled:

**"Digital Audio Tampering Detection Based on ENF Spatio-temporal Features Representation Learning"**

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

