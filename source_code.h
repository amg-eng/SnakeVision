#ifndef SOURCE_CODE_H
#define SOURCE_CODE_H

#include <opencv2/core.hpp> 

using namespace std;
using namespace cv;

// Constants
/*
const int GAUSSIAN_KERNEL_SIZE = 5;
const int AVERAGE_KERNEL_SIZE = 3;
const int MEDIAN_KERNEL_SIZE = 3;
const double GAUSSIAN_SIGMA = 1.0;
const double GAUSSIAN_NOISE_STDDEV = 25;
const double UNIFORM_NOISE_LOW = 0;
const double UNIFORM_NOISE_HIGH = 50;
const float SALT_PROBABILITY = 0.01;
const float PEPPER_PROBABILITY = 0.01;
*/

// Function Prototypes

Mat addGaussianNoise(const Mat& image, double mean, double stddev, const string& color);
Mat addUniformNoise(const Mat& image, double low, double high, const string& color);
Mat addSaltAndPepperNoise(const Mat& image, float salt_prob, float pepper_prob, const string& color);
Mat applyGaussianFilter(const Mat& image, int kernel_size, double sigma, const string& color);
Mat applyAverageFilter(const Mat& image, int kernel_size, const string& color);
Mat applyMedianFilter(const Mat& image, int kernel_size, const string& color);


#endif // SOURCE_CODE_H