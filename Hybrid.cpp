#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <cmath>
#include"Hybrid.h"

using namespace cv;
using namespace std;


/**
 * Reads an image from the specified image path.
 *
 * @param image_path The path to the image file.
 * @return The loaded image.
 */
Mat read_image(String image_path) {
    Mat img = imread(image_path, IMREAD_COLOR);
    return img;
}



/**
 * Normalizes the image matrix to the range [0, 1].
 *
 * @param image The input image matrix.
 * @return The normalized image matrix.
 */cv::Mat normalize_image(cv::Mat image) {
    cv::Mat norm_img;

    double min_val, max_val;
    cv::minMaxLoc(image, &min_val, &max_val); // Find min and max values

    // Normalize the image to the range [0, 255]
    image.convertTo(norm_img, CV_8U, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));

    return norm_img;
}



 /**
  * Equalizes the intensity histogram of the input image.
  *
  * @param image The input image matrix.
  * @return The equalized image matrix.
  */Mat equalize_image(Mat image) {
    Mat eq_img;
    equalizeHist(image, eq_img);
    return eq_img;
}



  /**
   * Displays the input image.
   *
   * @param img The image matrix to be displayed.
   */
  void display_img(Mat img) {
    imshow("Grey-Scaled Image", img);
    waitKey(0);
}



 /**
 * Computes the Fourier transform of the input image.
 *
 * @param image The input image matrix.
 * @return An array containing the magnitude and phase of the Fourier transform.
 */
array<Mat, 2> image_fourier_transform(Mat image) {

    Mat planes[] = { Mat_<float>(image), Mat::zeros(image.size(), CV_32F) };

    Mat complexI(image.rows, image.cols, CV_32FC2);
    for (int i = 0; i < complexI.rows; ++i) {
        for (int j = 0; j < complexI.cols; ++j) {
            complexI.at<Vec2f>(i, j)[0] = planes[0].at<float>(i, j);
            complexI.at<Vec2f>(i, j)[1] = planes[1].at<float>(i, j);
        }
    }
    dft(complexI, complexI);
    array<Mat, 2> result;
    result[0] = Mat::zeros(image.size(), CV_32F);
    result[1] = Mat::zeros(image.size(), CV_32F);
    for (int i = 0; i < complexI.rows; ++i) {
        for (int j = 0; j < complexI.cols; ++j) {
            result[0].at<float>(i, j) = log(1 + sqrt(complexI.at<Vec2f>(i, j)[0] * complexI.at<Vec2f>(i, j)[0] + complexI.at<Vec2f>(i, j)[1] * complexI.at<Vec2f>(i, j)[1]));
            result[1].at<float>(i, j) = atan2(complexI.at<Vec2f>(i, j)[1],complexI.at<Vec2f>(i, j)[0]);
        }
    }
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = result[0].cols / 2;
    int cy = result[0].rows / 2;
    Mat q0(result[0], Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(result[0], Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(result[0], Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(result[0], Rect(cx, cy, cx, cy)); // Bottom-Right
    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);


    int cx1 = result[1].cols / 2;
    int cy1= result[1].rows / 2;
    Mat q01(result[1], Rect(0, 0, cx1, cy1));   // Top-Left - Create a ROI per quadrant
    Mat q11(result[1], Rect(cx1, 0, cx1, cy1));  // Top-Right
    Mat q21(result[1], Rect(0, cy1, cx1, cy1));  // Bottom-Left
    Mat q31(result[1], Rect(cx1, cy1, cx1, cy1)); // Bottom-Right
    Mat tmp1;                           // swap quadrants (Top-Left with Bottom-Right)
    q01.copyTo(tmp1);
    q31.copyTo(q01);
    tmp1.copyTo(q31);
    q11.copyTo(tmp1);                    // swap quadrant (Top-Right with Bottom-Left)
    q21.copyTo(q11);
    tmp1.copyTo(q21);

    return result;
}



/**
 * Filters the magnitude and phase of the Fourier transform based on a specified type and cutoff frequency.
 *
 * @param magnitude The magnitude of the Fourier transform.
 * @param phase The phase of the Fourier transform.
 * @param type The type of frequency filter ("high" for high-pass, "low" for low-pass).
 * @param d The cutoff frequency for the filter.
 * @return An array containing the filtered magnitude and phase.
 */
array<Mat, 2> image_frequancy_filter(Mat magnitude, Mat phase,string type,int d) {
    array<Mat, 2> result;
    result[0] = Mat::zeros(magnitude.size(), CV_32F);
    result[1] = Mat::zeros(magnitude.size(), CV_32F);
    int cx = result[1].cols / 2;
    int cy = result[1].rows / 2;
    if (type == "high")
    {
        for (int i = 0; i < magnitude.rows; ++i) {
            for (int j = 0; j < magnitude.cols; ++j) {
                if (sqrt((cy - i)* (cy - i) + (cx - j)* (cx - j)) >= d) {
                    result[0].at<float>(i, j) = magnitude.at<float>(i, j);
                    result[1].at<float>(i, j) = phase.at<float>(i, j);
                }
                else {
                    result[0].at<float>(i, j) = 0;
                    result[1].at<float>(i, j) = 0;
                }
            }
        }
    }else if(type == "low"){
    
    
        for (int i = 0; i < magnitude.rows; ++i) {
            for (int j = 0; j < magnitude.cols; ++j) {
                if (sqrt((cy - i) * (cy - i) + (cx - j) * (cx - j)) <= d) {
                    result[0].at<float>(i, j) = magnitude.at<float>(i, j);
                    result[1].at<float>(i, j) = phase.at<float>(i, j);
                }
                else {
                    result[0].at<float>(i, j) = 0;
                    result[1].at<float>(i, j) = 0;
                }
            }
        }
    }

    
    return result;
}



/**
 * Computes the inverse Fourier transform of the magnitude and phase.
 *
 * @param magnitude The magnitude of the Fourier transform.
 * @param phase The phase of the Fourier transform.
 * @return The inverse Fourier transformed image.
 */
Mat image_inverse_fourier_transform(Mat magnitude, Mat phase) {
    int cx = magnitude.cols / 2;
    int cy = magnitude.rows / 2;
    Mat q0(magnitude, Rect(0, 0, cx, cy));   // Top-Left
    Mat q1(magnitude, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magnitude, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magnitude, Rect(cx, cy, cx, cy)); // Bottom-Right

    // Swap quadrants back (Top-Left with Bottom-Right)
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    // Swap quadrants back (Top-Right with Bottom-Left)
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    int cx1 = phase.cols / 2;
    int cy1 = phase.rows / 2;
    Mat q01(phase, Rect(0, 0, cx1, cy1));   // Top-Left
    Mat q11(phase, Rect(cx1, 0, cx1, cy1));  // Top-Right
    Mat q21(phase, Rect(0, cy1, cx1, cy1));  // Bottom-Left
    Mat q31(phase, Rect(cx1, cy1, cx1, cy1)); // Bottom-Right

    // Swap quadrants back (Top-Left with Bottom-Right)
    Mat tmp1;
    q01.copyTo(tmp1);
    q31.copyTo(q01);
    tmp1.copyTo(q31);

    // Swap quadrants back (Top-Right with Bottom-Left)
    q11.copyTo(tmp1);
    q21.copyTo(q11);
    tmp1.copyTo(q21);


    Mat_<complex<float>> result = Mat::zeros(magnitude.size(), CV_32FC2);
    for (int i = 0; i < magnitude.rows; ++i) {
        for (int j = 0; j < magnitude.cols; ++j) {
            result.at<complex<float>>(i, j) = polar(exp(magnitude.at<float>(i, j)) - 1, phase.at<float>(i, j));
        }
    }
    Mat inverseTransform;
    idft(result, inverseTransform, DFT_REAL_OUTPUT | DFT_SCALE);
    inverseTransform.convertTo(inverseTransform, CV_8U);

    return inverseTransform;
}



/**
 * Computes the inverse Fourier transform of multiple magnitude and phase images and combines them into a single image.
 *
 * @param images_fourier An array containing magnitude and phase images for each component.
 * @return The combined inverse Fourier transformed image.
 */
Mat image_inverse_fourier_transform2(array < Mat, 4 > images_fourier) {
    

    for (int i = 0; i < 4; i++) {
        int cx = images_fourier[i].cols / 2;
        int cy = images_fourier[i].rows / 2;
        Mat q0(images_fourier[i], Rect(0, 0, cx, cy));   // Top-Left
        Mat q1(images_fourier[i], Rect(cx, 0, cx, cy));  // Top-Right
        Mat q2(images_fourier[i], Rect(0, cy, cx, cy));  // Bottom-Left
        Mat q3(images_fourier[i], Rect(cx, cy, cx, cy)); // Bottom-Right

        // Swap quadrants back (Top-Left with Bottom-Right)
        Mat tmp;
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        // Swap quadrants back (Top-Right with Bottom-Left)
        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);
    }

    Mat_<complex<float>> result = Mat::zeros(images_fourier[0].size(), CV_32FC2);
    for (int i = 0; i < min(images_fourier[0].rows, images_fourier[2].rows); ++i) {
        for (int j = 0; j < min(images_fourier[0].cols, images_fourier[2].cols); ++j) {
            result.at<complex<float>>(i, j) = polar(exp(images_fourier[0].at<float>(i, j)) - 1, images_fourier[1].at<float>(i, j))+ polar(exp(images_fourier[2].at<float>(i, j)) - 1, images_fourier[3].at<float>(i, j));
        }
    }
    Mat inverseTransform;
    idft(result, inverseTransform, DFT_REAL_OUTPUT | DFT_SCALE);
    inverseTransform.convertTo(inverseTransform, CV_8U);

    return inverseTransform;
}



/**
 * Creates a hybrid image by filtering and combining two input images in the frequency domain.
 *
 * @param image1 The first input image matrix.
 * @param image2 The second input image matrix.
 * @param d The cutoff frequency for the frequency filter.
 * @return The hybrid image created by combining the filtered images.
 */
Mat hybrid_image(Mat image1, Mat image2 , int d) {
    //image should be gray scale
    
    array < Mat, 2 > image1_fourier = image_fourier_transform(image1);
    array < Mat, 2 > image2_fourier = image_fourier_transform(image2);
    array < Mat, 2 > image1_filtered = image_frequancy_filter(image1_fourier[0], image1_fourier[1], "high", d);
    array < Mat, 2 > image2_filtered = image_frequancy_filter(image2_fourier[0], image2_fourier[1], "low", d);
    /*array < Mat, 2 > hybrid_image_fourier;
    hybrid_image_fourier[0] = Mat::zeros(image1_fourier[0].size(), CV_32F);
    hybrid_image_fourier[1] = Mat::zeros(image1_fourier[0].size(), CV_32F);
    
    for (int i = 0; i < image1_fourier[0].rows; ++i) {
        for (int j = 0; j < image1_fourier[0].cols; ++j) {
            hybrid_image_fourier[0].at<float>(i, j) = image1_filtered[0].at<float>(i, j)+ image1_filtered[0].at<float>(i, j);
            hybrid_image_fourier[1].at<float>(i, j) = image2_filtered[1].at<float>(i, j)+ image2_filtered[1].at<float>(i, j);
        }
    }*/
    array < Mat, 4 > hybrid_image_fourier;
    hybrid_image_fourier[0] = image1_filtered[0];
    hybrid_image_fourier[1] = image1_filtered[1];
    hybrid_image_fourier[2] = image2_filtered[0];
    hybrid_image_fourier[3] = image2_filtered[1];
    Mat result = image_inverse_fourier_transform2(hybrid_image_fourier);
    return result;

}



/**
 * Filters the input grayscale image in the frequency domain based on a specified type and cutoff frequency.
 *
 * @param image_gray The input grayscale image matrix.
 * @param type The type of frequency filter ("high" for high-pass, "low" for low-pass).
 * @param d The cutoff frequency for the filter.
 * @return The filtered image in the spatial domain.
 */
Mat image_fourier_filter(Mat image_gray, string type, int d) {
    array < Mat, 2 > fourier_image = image_fourier_transform(image_gray);
    array < Mat, 2 > filter_fourier_image = image_frequancy_filter(fourier_image[0], fourier_image[1], type, d);
    return image_inverse_fourier_transform(filter_fourier_image[0], filter_fourier_image[1]);
}




