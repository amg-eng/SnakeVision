#ifndef THRESHOLDING
#define THRESHOLDING

#include <opencv2/opencv.hpp>
#include "opencv2/opencv_modules.hpp"

using namespace cv;
using namespace std;

Mat globalThreshold(const Mat& inputImage, int thresholdValue, unsigned char maximum_value, unsigned char minimum_value);

Mat localAdaptiveMeanThreshold(const Mat& inputImage, int kernalSize, int constant, unsigned char maximum_value, unsigned char minimum_value);

Mat adaptivePaddingFunction(const Mat& inputImage, int paddingSize);

Mat localThresholdMeanCalculation(const Mat& inputImage, int kernalSize);


void applyLocalGaussianThreshold(const cv::Mat& inputImage, cv::Mat& outputImage, int blockSize, double C);
Mat gaussianBlur(const Mat& inputImage);
Mat gaussianThreshold(const Mat& inputImage, int thresholdValue);
#endif