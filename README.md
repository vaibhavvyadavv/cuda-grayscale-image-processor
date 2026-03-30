# CUDA Image Processing Project

## Overview
This project implements GPU-accelerated image processing using CUDA. The program converts multiple input images into grayscale using a custom CUDA kernel and processes them in batch.

## Features
- GPU-based grayscale conversion using CUDA
- Batch processing of multiple images
- Performance timing using C++ chrono
- Command-line executable

## Project Structure
- `src/` - Source code (main.cu)
- `bin/` - Compiled executable
- `data/input/` - Input images
- `data/output/` - Output images

## Build Instructions
Run the following command: make build

## Run Instructions
./bin/project.exe


## Output
- Processed grayscale images are saved in `data/output/`
- Terminal displays number of images processed and total execution time

## Technologies Used
- CUDA (GPU computing)
- OpenCV (image handling)
- C++ (core logic)

## Results
The program successfully processes multiple images using GPU parallelism and demonstrates efficient batch processing.

## Author
Vaibhav Yadav