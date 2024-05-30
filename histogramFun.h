#ifndef HISTOGRAM_EQUALIZATION_H
#define HISTOGRAM_EQUALIZATION_H
#include <opencv2/core/base.hpp>
#include <opencv2/opencv.hpp>

cv::Mat equalizeHistogram(const cv::Mat& inputImage);
cv::Mat normalizeImage(const cv::Mat& inputImage);

#endif // HISTOGRAM_EQUALIZATION_H
