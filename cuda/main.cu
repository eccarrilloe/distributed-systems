#include <iostream>
#include <stdio.h>
#include <cmath>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;


int main (int argc, char* argv[])
{
    try
    {
        cv::Mat src_host = cv::imread("file.png", cv::IMREAD_GRAYSCALE);
        cv::cuda::GpuMat dst, src;
        src.upload(src_host);
        cv::cuda::threshold(src, dst, 128.0, 255.0, cv::THRESH_BINARY);
        cv::Mat result_host(dst);
        cv::imshow("Result", result_host);
        cv::waitKey();
    }
    catch(const cv::Exception& ex)
    {
        std::cout << "Error: " << ex.what() << std::endl;
    }
    return 0;
}