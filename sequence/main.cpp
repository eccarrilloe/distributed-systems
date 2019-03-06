#include <iostream>
#include <cmath>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;
using namespace std;

VideoCapture openVideo(string filename);
double processFrame(Mat frame);

int main(int argc, char** argv)
{
    // Init data and variables
    string filename = "../../samples/sunshine.mp4";
    VideoCapture video = openVideo(filename);

    int frameCount = video.get(CV_CAP_PROP_FRAME_COUNT);
    int samples = 100;
    int interval = frameCount / samples;

    printf("[INFO] Setup Data\n");
    int startInitFrames = clock();

    // Get frame samples from VideoCapture
    Mat* data = new Mat[samples];
    for (int i = 0; i < samples; i++) {
        Mat frame;

        video.set(CV_CAP_PROP_POS_FRAMES, i * interval);
        video >> frame;

        data[i] = frame.clone();
    }

    double timeInitFrames = (double) (clock() - startInitFrames) / CLOCKS_PER_SEC;
    printf("[INFO] Time Init: %.2f ms\n", timeInitFrames * 1000);

    printf("[INFO] Running Sequencial Code\n");

    double* results = new double[samples];
    int startProcessingTime = clock();
    for (int i = 0; i < samples; i++) {
        results[i] = processFrame(data[i]);
    }

    double timeProcessing = (double) (clock() - startProcessingTime) / CLOCKS_PER_SEC;

    for (int i = 0; i < samples; i++) {
        printf("Sample #%i: %.3f\n", i, results[i] / 2.55);
    }
    printf("[INFO] Time Processing: %.3f ms\n", timeProcessing * 1000);

    return 0;
}


VideoCapture openVideo(string filename) {
    VideoCapture capture(filename);


    if (!capture.isOpened()) {
        throw "[ERROR] Cannot open filename";
    }

    return capture;
}

double processFrame(Mat frame) {
    int rows = frame.rows;
    int cols = frame.cols;
    double pixels = rows * cols;

    double illuminance = 0;
    Vec3b pixel;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            pixel = frame.at<Vec3b>(Point(j, i));

            illuminance += 0.3 * pixel[0] + 0.59 * pixel[1] + 0.11  * pixel[2];
        }
    }

    return illuminance / pixels;
}
