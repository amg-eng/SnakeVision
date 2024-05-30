#ifndef HOUGHLINE_H
#define HOUGHLINE_H

#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


Mat Hough_Line_Transform(const Mat& image, int threshold, double lineResolution);
void boundry_detection(cv::Mat img, std::vector<std::vector<cv::Point>>& contours);
void drawEllipses(cv::Mat& img, const std::vector<cv::Vec6d>& ellipses);
Mat HoughEllipse(Mat img, vector<Vec6d>& ellipses, int threshold, int minRadius, int maxRadius);
Mat Hough_circle_transform(const Mat &image, int threshold, int min_radius /* =10 */, int max_radius /* =200 */, int canny_min_thresold /* = 100 */, int canny_max_thresold /* = 200 */, int thetas);
#endif // HOUGHLINE_H
