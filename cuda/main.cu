#include <iostream>
#include <stdio.h>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace cv::cuda;
using namespace std;

VideoCapture openVideo(string filename);
__global__ void processFrame(GpuMat* frames, int* N, double* results);

int main(void)
{
    // Init data and variables
    string filename = "../../samples/sunshine.mp4";
    VideoCapture video = openVideo(filename);

    int frameCount = video.get(CV_CAP_PROP_FRAME_COUNT);
    int samples = 100;
    int interval = frameCount / samples;
    int numBlocks = 4;
    int numThreads = 64;

    printf("[INFO] Setup Data\n");
    int startInitFrames = clock();

    // Get frame samples from VideoCapture
    GpuMat* data = new GpuMat[samples];
    for (int i = 0; i < samples; i++) {
        Mat frame;
        GpuMat gpuFrame;

        video.set(CV_CAP_PROP_POS_FRAMES, i * interval);
        video >> frame;

        gpuFrame.upload(frame);
        data[i] = gpuFrame;
    }

    // Transfer data from Host to Device
    double* results;
    int* N;
    int* d_N;
    N = &samples;

    GpuMat* frames;

    cudaMalloc(&d_N, sizeof(int));
    cudaMalloc(&frames, samples * sizeof(Mat));
    cudaMallocManaged(&results, samples * sizeof(double));

    cudaMemcpy((void*) d_N, N, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) frames, data, samples * sizeof(Mat), cudaMemcpyHostToDevice);

    double timeInitFrames = (clock() - startInitFrames) / CLOCKS_PER_SEC;
    printf("[INFO] Time Init: %.2f ms\n", timeInitFrames * 1000);

    printf("[INFO] Running Device Code\n");
    int startProcessingTime = clock();
    processFrame<<<numBlocks, numThreads>>>(frames, d_N, results);

    cudaDeviceSynchronize();
    double timeProcessing = (double) (clock() - startProcessingTime) / CLOCKS_PER_SEC;

    for (int i = 0; i < samples; i++) {
        printf("Sample #%i: %.3f\n", i, results[i] / 2.55);
    }

    printf("[INFO] Device Synchronized\n");
    printf("[INFO] Time Processing: %.5f ms\n", timeProcessing * 1000);

    cudaFree(data);
    cudaFree(results);
    return 0;
}


VideoCapture openVideo(string filename) {
    VideoCapture capture(filename);

    if (!capture.isOpened()) {
        throw "[ERROR] Cannot open filename";
    }

    return capture;
}


__global__ void processFrame(GpuMat* frames, int* N, double* results) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < *N) {
        int rows = frames[index].rows;
        int cols = frames[index].cols;
        int pixels = rows * cols;

        double illumina = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                uint8_t pixelR = frames[index].data[i * cols + j * 3 + 0];
                uint8_t pixelG = frames[index].data[i * cols + j * 3 + 1];
                uint8_t pixelB = frames[index].data[i * cols + j * 3 + 2];

                illumina += 0.3 * pixelR + 0.59 * pixelG + 0.11  * pixelB;
            }
        }

        results[index] = illumina / pixels;
    }

}