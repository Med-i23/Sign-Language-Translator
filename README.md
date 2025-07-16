# Sign-Language-Translator

This project was developed as part of my BSc thesis. It consists of two main components:
1. **Application** – a real-time sign language recognition program using webcam input
2. **Model Training** – custom training of neural networks using TensorFlow

Due to environment incompatibilities between TensorFlow versions and some Python packages, the two parts require **separate environments**.

---

## 📦 Repository Structure

```
├── application.py                # Main application file
├── downloader.py                # Script for downloading required assets
├── models/                      # Directory with trainable model scripts
├── requirements_app.txt         # Dependencies for the application
├── requirements_model.txt       # Dependencies for model training
└── README.md
```

---

## 1. Running the Application

### Prerequisites
- Python 3.12
- Webcam
- Git

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/sign-language-recognition.git
   cd sign-language-recognition
   ```

2. **(Optional) Create a virtual environment:**

   ```bash
   python -m venv env_program
   ```

   - On **Linux/macOS**:
     ```bash
     source env_program/bin/activate
     ```
   - On **Windows**:
     ```cmd
     env_program\Scripts\activate
     ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements_app.txt
   ```

4. **Download necessary files:**

   ```bash
   python downloader.py
   ```

5. **Run the application:**

   ```bash
   python application.py
   ```

   ⚠️ *Note: The model may take a few seconds to load on startup.*

### Switching Models

The application uses `ResNet50` by default. To switch to a different model (e.g. your own trained one), change the `model_path` global variable at the top of `application.py`.

---

## 2. Running Model Training

> ⚠️ Requires a **compatible NVIDIA GPU** with **CUDA Compute Capability ≥ 5.0** for GPU acceleration.

### Prerequisites
- Anaconda (recommended for managing the environment)
- Compatible NVIDIA GPU (or skip CUDA steps for CPU-only)
- Python 3.10 (for TensorFlow ≤ 2.10)

### Setup

1. **Install Anaconda** from the [official website](https://www.anaconda.com/).

2. **Create a new environment:**

   ```bash
   conda create -n py310 python=3.10
   conda activate py310
   ```

3. **(Optional) Install CUDA & cuDNN** for GPU acceleration:

   ```bash
   conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
   ```

   > ❗ Skip this step if you do not have an NVIDIA GPU.

4. **Install Python packages:**

   ```bash
   pip install -r requirements_model.txt
   ```

5. **Run model training:**

   ```bash
   python models/{model}.py
   ```

   Replace `{model}` with the script you want to run, e.g. `resnet50`, `mobilenetv2`, etc.

### IDE Integration (Optional)
If you're using an IDE like **PyCharm**, you can set your Anaconda environment as the Python interpreter for easier development.

---

## 🛠 Additional Notes

- **GPU acceleration** with CUDA greatly improves training speed.
- The environments are kept separate to avoid conflicting dependency versions.
- Ensure that your webcam is properly connected before launching the application.

---

## 📄 License

This project is part of an academic thesis and is intended for educational and demonstration purposes.
