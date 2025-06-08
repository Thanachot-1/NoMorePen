# NoMorePen
# GAN-Based Handwritten Thai Character Generation

This repository provides a complete pipeline to train a Generative Adversarial Network (GAN) to generate handwritten Thai characters (ก-ฮ) using the KVIS TOCR dataset. The model is trained one character at a time, and the weights are saved as `.pth` files for later use. It is highly recommended to run the code in **Google Colab** or **Kaggle** for easy access to GPU resources.

---

## 📁 Repository Structure

```bash
your-repo/
├── data/
│   └── data_collection_and_cleaning.py    # Load and prepare character-level data
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # Basic visual inspection and stats
├── src/
│   └── modeling/
│       └── train_and_evaluate.py          # GAN training and validation
├── output/                                # Output images and saved .pth files
└── README.md                              # This file
```

---

## ⚙️ Requirements

* Python 3.8+
* PyTorch 1.12+
* Torchvision 0.13+
* CUDA GPU (optional but recommended)
* matplotlib
* Jupyter Notebook (for local usage)

To install dependencies in Colab or Kaggle:

```bash
pip install torch torchvision matplotlib
```

---

## 📥 Dataset

We use the [KVIS TOCR Dataset](https://www.kvisteach.org/), organized like this:

```
/KVIS_TOCR/
├── ก/
│   ├── img1.png
│   ├── img2.png
│   └── ...
├── ข/
│   └── ...
└── ฮ/
```

Each folder contains handwritten images of a single Thai character (ก-ฮ).

---

## 🚀 How to Run in Colab or Kaggle

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Upload or Mount the Dataset

* **On Colab:**

```python
from google.colab import drive
drive.mount('/content/drive')
!ln -s /content/drive/MyDrive/KVIS_TOCR /content/KVIS_TOCR
```

Then set `root_dir = '/content/KVIS_TOCR/ก'` in your scripts.

* **On Kaggle:**
  Go to *Add Data* in your Notebook settings, upload your `KVIS_TOCR` folder, and use a path like `/kaggle/input/KVIS_TOCR/ก`.

### 3. (Optional) Data Collection & Cleaning

You **do not need this step** if your images are already clean. Run it **only if** you want to preprocess or filter:

```bash
python data/data_collection_and_cleaning.py --root_dir="/content/KVIS_TOCR/ก"
```

### 4. (Optional) Exploratory Data Analysis (EDA)

Run this to visualize and understand the distribution of handwriting styles:

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

You can **skip this step** if you're already confident in your data.

### 5. Train the GAN

Update `root_dir` in `src/modeling/train_and_evaluate.py` to match the character folder, e.g., `root_dir = "/content/KVIS_TOCR/ก"`.

Then run:

```bash
python src/modeling/train_and_evaluate.py
```

Adjust training hyperparameters at the top of the script:

* `target_epochs`
* `batch_size`
* `lr_g`, `lr_d`
* `gen_updates`

Trained models will be saved in `output/` as:

* `generator_epoch_XXX.pth`
* `discriminator_epoch_XXX.pth`

---

## 🧪 Using the Trained Generator

Example code to generate samples:

```python
import torch
from src.modeling.train_and_evaluate import Generator
from torchvision import utils
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 100

G = Generator().to(device)
G.load_state_dict(torch.load('output/generator_epoch_1000.pth', map_location=device))
G.eval()

z = torch.randn(64, latent_dim, 1, 1, device=device)
fake_imgs = G(z).cpu()

grid = utils.make_grid(fake_imgs, nrow=8, normalize=True)
plt.figure(figsize=(4,4))
plt.axis('off')
plt.imshow(grid.permute(1,2,0).squeeze(), cmap='gray')
plt.show()
```

---

## 💡 Tips & Best Practices

* Monitor `D_loss` and `G_loss` to check stability.
* Save checkpoints every few epochs.
* Use image augmentations (like rotation/affine) to increase robustness.
* Train character-by-character and store `.pth` separately for each Thai letter.

---

> 📌 Questions, suggestions, or contributions are welcome! Just open an Issue or Pull Request. 🎉
