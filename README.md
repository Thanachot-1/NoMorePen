# NoMorePen

# GAN-Based Handwritten Thai Character Generation

This repository provides a complete pipeline to train a Generative Adversarial Network (GAN) to generate handwritten Thai characters (ก-ฮ) using the KVIS TOCR dataset. The code is organized into three main stages:

1. **Data Collection & Cleaning** (`data/`)
2. **Exploratory Data Analysis (EDA)** (`notebooks/`)
3. **Modeling, Validation & Error Analysis** (`src/modeling/`)

You can run the pipeline either in Google Colab or on Kaggle Kernels.

---

## 📁 Repository Structure

```bash
your-repo/
├── data/
│   └── data_collection_and_cleaning.py    # จัดการโหลดและเตรียมข้อมูล
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # วิเคราะห์ข้อมูลเบื้องต้น
├── src/
│   └── modeling/
│       └── train_and_evaluate.py          # สคริปต์ฝึก GAN พร้อม validation และ error logging
├── output/                                # ผลลัพธ์: รูปตัวอย่าง และไฟล์ .pth ที่บันทึกโมเดล
└── README.md                              # ไฟล์นี้
```

---

## ⚙️ Prerequisites

* Python 3.8+
* PyTorch 1.12+
* Torchvision 0.13+
* CUDA (optional, สำหรับ GPU)
* Jupyter Notebook (ถ้าใช้ local)

---

## 📥 Dataset

ใช้ [KVIS TOCR Dataset](https://www.kvisteach.org/) โดยดาวน์โหลดไว้ในโฟลเดอร์โครงสร้าง:

```
/KVIS_TOCR/
├── ก/
│   ├── img1.png
│   ├── img2.png
│   └── ...
├── ข/
│   └── ...
└── ...
```

แต่ละโฟลเดอร์ภายในจะเก็บภาพของตัวอักษรเดียวกัน (ก-ฮ) เพื่อให้สคริปต์เลือกเทรนทีละตัวได้ง่าย

---

## 🚀 วิธีใช้งาน บน Colab หรือ Kaggle

### 1. เตรียมโค้ด

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. เตรียม Dataset

* **Colab:**

  1. อัปโหลดโฟลเดอร์ `KVIS_TOCR` ไปที่ Google Drive
  2. รันเซลล์ต่อไปนี้ใน Colab:

     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     !ln -s /content/drive/MyDrive/KVIS_TOCR /content/KVIS_TOCR
     ```
* **Kaggle:**

  1. ไปที่ Settings ของ Kernel
  2. เลือก `Add Data` → KVIS\_TOCR (อัปโหลดเข้าระบบ)
  3. ในโค้ด ให้กำหนด `root_dir = '/kaggle/input/KVIS_TOCR/ก'` (หรือไดเรกทอรีที่เหมาะสม)

### 3. ติดตั้ง Dependencies

```bash
pip install torch torchvision matplotlib
```

### 4. Data Collection & Cleaning

รันสคริปต์โหลดและเตรียมข้อมูล (ปรับ `root_dir` ให้เป็นโฟลเดอร์ตัวอักษรที่ต้องการ):

```bash
python data/data_collection_and_cleaning.py
```

### 5. Exploratory Data Analysis (EDA)

เปิด Jupyter Notebook เพื่อวิเคราะห์ข้อมูลเบื้องต้น:

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

### 6. Training & Validation

รันฝึก GAN ทีละตัวอักษร (ปรับค่า `root_dir` ใน `train_and_evaluate.py` ให้ชี้ไปที่โฟลเดอร์ตัวอักษรนั้น)

```bash
python src/modeling/train_and_evaluate.py
```

* **ปรับพารามิเตอร์** ในหัวไฟล์ได้แก่:

  * `target_epochs`: จำนวน epoch ที่ต้องการเทรน
  * `batch_size`: ขนาด batch
  * `lr_g`, `lr_d`: learning rates
  * `gen_updates`: จำนวนครั้งอัปเดต Generator ต่อ Discriminator 1 รอบ

> เมื่อสคริปต์รันเสร็จไฟล์น้ำหนักจะถูกบันทึกใน `output/` ชื่อ `generator_epoch_XXX.pth` และ `discriminator_epoch_XXX.pth`

---

## 🔍 การใช้โมเดลที่บันทึกไว้

ตัวอย่างโค้ดโหลดโมเดลและสร้างภาพตัวอย่าง:

```python
import torch
from src.modeling.train_and_evaluate import Generator
from torchvision import utils
import matplotlib.pyplot as plt

# กำหนด device และ latent_dim ให้ตรงกับตอนฝึก
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 100

# สร้าง Generator
G = Generator().to(device)

# โหลด weigth ล่าสุด
ckpt_path = 'output/generator_epoch_1000.pth'
G.load_state_dict(torch.load(ckpt_path, map_location=device))
G.eval()

# สุ่ม noise และสร้างภาพ
z = torch.randn(64, latent_dim, 1, 1, device=device)
fake_imgs = G(z).cpu()

# แสดงผล
grid = utils.make_grid(fake_imgs, nrow=8, normalize=True)
plt.figure(figsize=(4,4)); plt.axis('off')
plt.imshow(grid.permute(1,2,0).squeeze(), cmap='gray')
plt.show()
```

---

## 💡 Tips & Best Practices

* **Monitor Loss:** ดูค่า D\_loss และ G\_loss ว่าลดลงตามที่คาดหรือไม่
* **Checkpointing:** เก็บโมเดลทุก 10 epochs เพื่อ rollback เมื่อจำเป็น
* **Data Augmentation:** ลองเพิ่ม `transforms.RandomRotation` หรือ `RandomAffine` เพื่อเพิ่มความหลากหลาย

---

> 📌 หากมีข้อสงสัย เปิด Issue หรือส่ง Pull Request ได้เลย ยินดีรับ feedback ครับ! 🎉
