#include <iostream>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;
using namespace std;

VideoCapture openVideo(string filename);

int main(int argc, char** argv)
{
    string filename = "../../samples/video3.mp4";
    VideoCapture video = openVideo(filename);
    Mat frame;

    int currentFrame = 0;
    int frameCount = video.get(CV_CAP_PROP_FRAME_COUNT);
    int fps = video.get(CV_CAP_PROP_FPS);
    int rate = 2.0;
    int i = 0;

    int sampleLength = ceil((frameCount / fps) / rate);
    cout << "[INFO] Taking " << sampleLength << " samples in intervals of " << rate << " seconds.";

    while (true) {
        video >> frame;

        i += 1;
        int rows = frame.rows, cols = frame.cols;




        if (frame.empty()) {
            break;
        }

        currentFrame += 1;
    }

    waitKey(0);
}


VideoCapture openVideo(string filename) {
    VideoCapture capture(filename);


    if (!capture.isOpened()) {
        throw "[ERROR] Cannot open filename";
    }

    return capture;
}

