# GPU-Accelerated Image Processing using CUDA

## 🚀 Overview
This project implements a high-performance image processing pipeline using CUDA for GPU acceleration. The system processes multiple images in batch and converts them to grayscale using a custom CUDA kernel.

## ⚡ Key Features
- GPU-based grayscale conversion using CUDA kernels
- Batch processing of multiple images
- Parallel pixel-level computation
- Performance measurement using C++ chrono
- Efficient memory management between CPU and GPU

## 🛠️ Tech Stack
- CUDA (GPU programming)
- OpenCV (image handling)
- C++ (core implementation)

## 📂 Project Structure
src/ → CUDA source code
bin/ → Compiled executable
data/input/ → Input images
data/output/→ Output images


## ⚙️ Build Instructions
```bash
make build
▶️ Run
./bin/project.exe
📊 Output
Grayscale images saved in data/output/
Console displays:
Images processed
Total execution time
🧠 How It Works

Each pixel is processed in parallel using a CUDA kernel. The RGB values are converted to grayscale using the standard weighted formula. The workload is distributed across GPU threads, enabling faster processing compared to CPU execution.

📈 Results

The program successfully demonstrates GPU acceleration for image processing tasks and handles multiple images efficiently through batch processing.

🎯 Learning Outcomes
Understanding of CUDA programming model
GPU memory management (cudaMalloc, cudaMemcpy)
Kernel design for parallel computation
Performance benchmarking
👨‍💻 Author

Vaibhav Yadav
