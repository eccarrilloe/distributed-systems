#include <iostream>
#include <stdio.h>
#include <cmath>
#include <ctime>
#include <omp.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

VideoCapture openVideo(string filename);
double processFrame(Mat frame);


int main(int argc, char** argv)
{
    string filename = "../../samples/sunshine.mp4";
    VideoCapture video = openVideo(filename);
    Mat frame;

    int currentFrame = 0;
    int currentSample = 1;
    int frameCount = video.get(CV_CAP_PROP_FRAME_COUNT);
    int fps = video.get(CV_CAP_PROP_FPS);
    int time = ceil(frameCount / fps);
    int rate = 2.0;
    int interval = rate * fps;
    int samples = floor(time / rate);
    vector<double> results;
    vector<Mat> frames;

    for (int i = 0; i <= samples; i++) {
        results.push_back(0);
    }

    cout << "[INFO] Video: " << time << " seconds." << endl;
    cout << "[INFO] Taking " << samples + 1 << " samples in intervals of " << rate << "s." << endl;
    cout << endl;

    clock_t startTime = clock();

    for (int i = 0; i <= samples; i++) {
        Mat frame;
        video.set(CV_CAP_PROP_POS_FRAMES, i * interval);
        video >> frame;

        frames.push_back(frame.clone());
    }

    double illuminance;
    #pragma omp parallel for shared(frames, results) private(illuminance)
    for (int i = 0; i <= samples; i++) {
        // printf("%i/%i \n", omp_get_num_threads(), omp_get_thread_num());
        illuminance = processFrame(frames[i]) / 2.55;

        results[i] = illuminance;
    }

    #pragma omp barrier

    for (int i = 0; i <= samples; i++) {
        currentFrame = i * interval + 1;

        printf("[SAMPLE #%i] Frame @%i, Time %.0fs: %.3f%%\n", i + 1, currentFrame, floor(currentFrame / fps), results[i]);
    }

    double timeTaken = (double) (clock() - startTime) / (CLOCKS_PER_SEC);

    cout << endl << "Process finished in " << timeTaken << "s." << endl;

    waitKey(0);
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