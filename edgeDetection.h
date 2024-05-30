#ifndef EDGEDETECTION_H
#define EDGEDETECTION_H
#include <opencv2/opencv.hpp>
#define WEAK_EDGE  40
#define STRONG_EDGE  255
// Function declarations
cv::Mat Phase_Gradient(const cv::Mat& gradient_x, const cv::Mat& gradient_y);
cv::Mat Magnitude_Gradient(const cv::Mat& gradient_x, const cv::Mat& gradient_y);

std::tuple<cv::Mat, cv::Mat, cv::Mat> applySobel(const cv::Mat& input);
std::tuple<cv::Mat, cv::Mat, cv::Mat> applyPrewitt(const cv::Mat& input);
std::tuple<cv::Mat, cv::Mat, cv::Mat> applyRoberts(const cv::Mat& input);


cv::Mat Hysteresis(cv::Mat& thresholded);
cv::Mat DoubleThresholding(cv::Mat& suppressed, float lowThreshold, float highThreshold);
cv::Mat NonMaxSuppression(cv::Mat& magnitude_gradient, cv::Mat& phase_gradient);
void applyCanny(const cv::Mat& input, cv::Mat& output, int lowThreshold = 5, int highThreshold = 20);
// cv::Mat Detect_Edges_Canny(const cv::Mat &src, int lowThreshold, int highThreshold );
std::tuple<cv::Mat, cv::Mat, cv::Mat> Detect_Edges_Canny(const cv::Mat & src, int lowThreshold , int highThreshold );
#endif // EDGEDETECTION_H
