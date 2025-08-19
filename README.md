# D4-Net: Detecting Deepfakes using a Dual-branch Deep Learner

![Pipeline](Assets/Architecture.pdf)

## 📌 Overview

The **D4-Net** project introduces a **dual-branch deep neural network** for robust deepfake detection.
It leverages **two complementary pathways**:

* **Semantic Branch (RGB/Xception)** → captures high-level spatial and facial cues.
* **Frequency Branch (MesoNet/FFT)** → extracts subtle frequency irregularities.

The outputs are integrated via **cross-attention** and **adaptive fusion**, leading to highly discriminative features for classification.

We evaluate the model on **FaceForensics++ (FF++)** and **Celeb-DF (V2)** datasets, achieving **97.99% (FF++)** and **96.33% (CeDF)** accuracy.

---

## 📂 Repository Structure

```
Assets/
│── Architecture.pdf        # Paper draft and architecture details
│── Pipeline.png            # Overall workflow of D4-Net
│── FF.png                  # Sample from FaceForensics++
│── CeDF.png                # Sample from Celeb-DF dataset
│── Error.png               # Misclassified sample illustration
│── CeDF_CM.png             # Confusion matrix for CeDF
│── FF++_CM.png             # Confusion matrix for FF++
│
├── dataloaders.py          # Dataset preparation and preprocessing
├── model.py                # D4-Net architecture and training loop
├── README.md               # Project documentation
```

---

## 🚀 Features

* Dual-path design combining **spatial (RGB)** and **frequency (FFT)** features.
* **Cross-attention mechanism** aligns modalities effectively.
* **Adaptive feature fusion** ensures robust decision-making.
* Lightweight yet high-performing (\~26M params, 9 GFLOPs).
* Tested on **benchmark datasets** for generalization.

---

## 📊 Results

| Dataset | Accuracy | F1 Score | AUC    |
| ------- | -------- | -------- | ------ |
| FF++    | 97.99%   | 0.9800   | 0.9948 |
| CeDF    | 96.33%   | 0.9725   | 0.9938 |

🔹 See detailed results and error analysis in `Assets/` plots.

---

## 🛠 Installation

```bash
git clone https://github.com/omkarmakar/D4-Net.git
cd D4-Net
pip install -r requirements.txt
```



## 📑 Reference

If you use this work, please cite:

```
Om Karmakar, Sk Mohiuddin, Asfak Ali, Dmitrii Kaplun, Ram Sarkar.
D4-Net: Detecting Deepfakes using a Dual-branch Deep Learner.
Proceedings of the 33rd ACM International Conference on Multimedia (MM '25), Dublin, Ireland.
DOI : https://doi.org/10.1145/3746270.3760238
```

---

## 📬 Contact

For questions or collaborations:

* Om Karmakar – [omkarmakar07@gmail.com](mailto:omkarmakar07@gmail.com)
* Sk Mohiuddin – [myselfmohiuddin@gmail.com](mailto:myselfmohiuddin@gmail.com)
