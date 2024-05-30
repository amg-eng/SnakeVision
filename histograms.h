#ifndef HISTOGRAMS_H
#define HISTOGRAMS_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

	void saveHistogramImage(const std::string& filename, const cv::Mat& histogramImage);
	cv::Mat normalzeHistogram_n(cv::Mat image);
	cv::Mat calculateCDF(const cv::Mat& hist);
	cv::Mat plotCDF(const cv::Mat& cdf, const cv::Scalar& color); // 	cv::Mat calculateCDF(const cv::Mat& hist);
	cv::Mat convertToGrayScale(const cv::Mat& image);


	cv::Mat calculateGrayscaleCDF(const cv::Mat& grayscaleImage);
	void plotGrayscaleCDF(const std::string& title, const cv::Mat& cdf, const cv::Scalar& lineColor, const cv::Scalar& fillColor);
	cv::Mat displayNormalizedEqualizedImages(const cv::Mat& image);
	cv::Mat normalizeHistogram_n(cv::Mat& histogram);

	cv::Mat calculateHistogram(const cv::Mat& input, int channel);
	cv::Mat plotHistogram(const cv::Mat& histogram, const cv::Scalar color, int histWidth, int histHeight);



	void calculateHistograms(Mat& image, Mat& histR, Mat& histG, Mat& histB);
	void drawHistograms(Mat& histR, Mat& histG, Mat& histB, Mat& histImageR, Mat& histImageG, Mat& histImageB, int histSize, double& maxIntensity);
	void addLabels(Mat& histImageR, Mat& histImageG, Mat& histImageB, int histSize, double maxIntensity, int histHeight, int histWidth);


	cv::Mat addAxesToHistogram(const cv::Mat& histogramImage);










#endif // HISTOGRAMS_H