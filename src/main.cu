#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>

using namespace std;
using namespace cv;

// CUDA kernel for RGB → Grayscale
__global__ void rgbToGray(unsigned char* input, unsigned char* output,
                          int width, int height, int channels) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;

    unsigned char r = input[idx + 2];
    unsigned char g = input[idx + 1];
    unsigned char b = input[idx + 0];

    output[y * width + x] = (0.299f * r + 0.587f * g + 0.114f * b);
}

int main() {

    // List of input images
    vector<string> images = {
        "data/input/1.jpg",
        "data/input/3.jpg",
        "data/input/4.jpeg",
        "data/input/5.jpg"
    };

    int count = 0;

    // Start timing
    auto start = chrono::high_resolution_clock::now();

    for (string input_path : images) {

        cout << "Processing: " << input_path << endl;

        Mat image = imread(input_path, IMREAD_COLOR);

        if (image.empty()) {
            cout << "Skipping invalid image: " << input_path << endl;
            continue;
        }

        int width = image.cols;
        int height = image.rows;
        int channels = image.channels();

        Mat gray(height, width, CV_8UC1);

        // Allocate GPU memory
        unsigned char *d_input, *d_output;
        cudaMalloc(&d_input, width * height * channels);
        cudaMalloc(&d_output, width * height);

        // Copy input to GPU
        cudaMemcpy(d_input, image.data, width * height * channels, cudaMemcpyHostToDevice);

        // Define block and grid size
        dim3 blockSize(16, 16);
        dim3 gridSize((width + 15) / 16, (height + 15) / 16);

        // Launch kernel
        rgbToGray<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);

        // Copy result back to CPU
        cudaMemcpy(gray.data, d_output, width * height, cudaMemcpyDeviceToHost);

        // Output file name
        string output_path = "data/output/out_" + to_string(count) + ".jpg";

        // Save image
        imwrite(output_path, gray);

        // Free GPU memory
        cudaFree(d_input);
        cudaFree(d_output);

        cout << "Saved: " << output_path << endl;

        count++;
    }

    // End timing
    auto end = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration<double>(end - start).count();

    cout << "-----------------------------------" << endl;
    cout << "Total images processed: " << count << endl;
    cout << "Total time taken: " << time_taken << " seconds" << endl;
    cout << "-----------------------------------" << endl;

    return 0;
}