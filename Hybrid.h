
#ifndef HYBRID_H
#define HYBRID_H
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include<cmath>
using namespace std;
using namespace cv;


Mat image_fourier_filter(Mat image_gray, string type, int d);
Mat hybrid_image(Mat image1, Mat image2 , int d);
Mat image_inverse_fourier_transform2(array < Mat, 4 > images_fourier);
Mat image_inverse_fourier_transform(Mat magnitude, Mat phase);
array<Mat, 2> image_frequancy_filter(Mat magnitude, Mat phase, string type, int d);
array<Mat, 2> image_fourier_transform(Mat image);
void display_img(Mat img);
Mat equalize_image(Mat image);
Mat normalize_image(Mat image);
Mat read_image(String image_path);

#endif