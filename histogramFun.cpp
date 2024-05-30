#include "histogramFun.h"
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <qmetatype.h>
using namespace cv;
using namespace std;

/**
 * @brief Normalize the input image using histogram equalization technique.
 *
 * @param inputImage The input grayscale image to be normalized.
 * @return The normalized image.
 */
Mat normalizeImage(const Mat& inputImage) {
    // Calculate histogram
    vector<int> histogram(256, 0);
    for (int i = 0; i < inputImage.rows; ++i) {
        for (int j = 0; j < inputImage.cols; ++j) {
            int intensity = inputImage.at<uchar>(i, j);
            histogram[intensity]++;
        }
    }

    // Calculate cumulative distribution function (CDF)
    vector<int> cdf(256, 0);
    cdf[0] = histogram[0];
    for (int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    // Normalize CDF
    vector<int> normalized_cdf(256, 0);
    float factor = 255.0f / (inputImage.rows * inputImage.cols);
    for (int i = 0; i < 256; ++i) {
        normalized_cdf[i] = round(cdf[i] * factor);
    }

    // Apply equalization
    Mat outputImage = inputImage.clone();
    for (int i = 0; i < inputImage.rows; ++i) {
        for (int j = 0; j < inputImage.cols; ++j) {
            int intensity = inputImage.at<uchar>(i, j);
            outputImage.at<uchar>(i, j) = normalized_cdf[intensity];
        }
    }

    return outputImage;
}



/**
 * @brief Equalize the histogram of the input image.
 *
 * @param inputImage The input grayscale image to be equalized.
 * @return The equalized image.
 */
Mat equalizeHistogram(const Mat& inputImage) {
    double minVal, maxVal;
    minMaxLoc(inputImage, &minVal, &maxVal);

    Mat outputImage;
    outputImage = (inputImage - minVal) * (255.0 / (maxVal - minVal));

    return outputImage;
}
