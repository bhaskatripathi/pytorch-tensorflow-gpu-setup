
# Setting Up both PyTorch and TensorFlow with GPU Support (CUDA & cuDNN) on Windows 11

This guide provides a practical approach to set up **PyTorch** and **TensorFlow** with GPU support using **CUDA** and **cuDNN** on a Windows 11 system. It’s tailored for beginners in machine learning and deep learning.

---

## Prerequisites

1. A Windows 11 system with an NVIDIA GPU (e.g., RTX 4070).
2. Anaconda installed for environment management.
3. Basic familiarity with Python.

---

## Step-by-Step Setup for PyTorch and TensorFlow

### 1. Install NVIDIA Drivers
- Download the latest drivers for your GPU from NVIDIA.
- After installation, verify the setup by running:
  ```bash
  nvidia-smi
  ```
  This command should display details about your GPU.

---

### 2. Install CUDA Toolkit
- Visit the **CUDA Toolkit Archive** to download the required version:
  [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
- For PyTorch and TensorFlow, it is recommended to use:
  - CUDA 12.4 (latest supported version).
  - CUDA 11.8 (for older setups).
- Install CUDA and ensure it’s added to your environment variables during installation.

---

### 3. Install cuDNN
- Download cuDNN from the following links:
  - [cuDNN Downloads](https://developer.nvidia.com/cudnn-downloads?target_os=Windows)
  - [cuDNN Archive](https://developer.nvidia.com/cudnn-archive)
- Match the cuDNN version with your installed CUDA version.
- Copy the contents of the downloaded `bin`, `include`, and `lib` folders to the corresponding directories in the CUDA installation directory. For example:
  ```
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
  ```

---

### 4. Set Up a Conda Environment
- Open **Anaconda Prompt** and create a new environment:
  ```bash
  conda create -n dl-env python=3.10 -y
  conda activate dl-env
  ```

---

### 5. Install PyTorch
- Install PyTorch with GPU support:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  ```

- Verify PyTorch installation:
  ```python
  import torch
  print(torch.__version__)  # Displays the installed PyTorch version
  print(torch.cuda.is_available())  # Returns True if GPU is detected
  print(torch.cuda.get_device_name(0))  # Displays the name of the GPU
  ```

---

### 6. Install TensorFlow
- Install TensorFlow:
  ```bash
  pip install tensorflow
  ```

- Verify TensorFlow installation:
  ```python
  import tensorflow as tf
  print(tf.__version__)  # Displays the installed TensorFlow version
  print(tf.config.list_physical_devices('GPU'))  # Lists GPUs detected
  ```

---

### 7. Set Up Spyder (Optional)
If you prefer to use **Spyder** as your IDE, follow these steps:

1. Install Spyder in your environment:
   ```bash
   conda install spyder -y
   ```
2. Configure Spyder to use the environment:
   - Open Spyder.
   - Navigate to **Tools > Preferences > Python Interpreter**.
   - Select **"Use the following Python interpreter"** and point it to:
     ```
     C:\Users\<YourUsername>\anaconda3\envs\dl-env\python.exe
     ```

---

## Key Links

1. [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
2. [cuDNN Downloads](https://developer.nvidia.com/cudnn-downloads?target_os=Windows)
3. [cuDNN Archive](https://developer.nvidia.com/cudnn-archive)
4. [PyTorch Official Website](https://pytorch.org/)
5. [TensorFlow GPU Installation Guide](https://www.tensorflow.org/install/source_windows)

---

## Notes

1. **Compatibility:** Ensure the PyTorch and TensorFlow versions match your installed CUDA and cuDNN versions.
2. **Driver Updates:** Use the latest NVIDIA drivers for optimal GPU performance.
3. **Environment Isolation:** Use Conda environments to avoid conflicts between libraries.

This guide provides a no-nonsense approach to setting up PyTorch and TensorFlow with GPU support for deep learning tasks.
