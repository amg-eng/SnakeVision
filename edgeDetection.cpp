#include "edgeDetection.h"
#include "source_code.h"
#include <qmath.h>

// Simple convolution function
void conv2D(const cv::Mat& input, const cv::Mat& kernel, cv::Mat& output) {
    cv::Mat flippedKernel;
    cv::flip(kernel, flippedKernel, -1);

    cv::Mat result;
    cv::filter2D(input, result, -1, flippedKernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

    output = result;
}

// void applySobel(const cv::Mat& input, cv::Mat& output) {
//     cv::Mat kernelX = (cv::Mat_<float>(3, 3) <<
//                            -1, 0, 1,
//                        -2, 0, 2,
//                        -1, 0, 1);

//     cv::Mat kernelY = (cv::Mat_<float>(3, 3) <<
//                            -1, -2, -1,
//                        0,  0,  0,
//                        1,  2,  1);

//     cv::Mat gradX, gradY;
//     cv::Mat absGradX, absGradY;
//     cv::Mat grad;

//     conv2D(input, kernelX, gradX);
//     conv2D(input, kernelY, gradY);

//     cv::convertScaleAbs(gradX, absGradX);
//     cv::convertScaleAbs(gradY, absGradY);

//     cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, grad);

//     output = grad;
// }

std::tuple<cv::Mat, cv::Mat, cv::Mat> applySobel(const cv::Mat& input) {
    cv::Mat kernelX = (cv::Mat_<float>(3, 3) <<
                           -1, 0, 1,
                       -2, 0, 2,
                       -1, 0, 1);

    cv::Mat kernelY = (cv::Mat_<float>(3, 3) <<
                           -1, -2, -1,
                       0, 0, 0,
                       1, 2, 1);

    cv::Mat gradX, gradY;
    cv::Mat absGradX, absGradY;
    cv::Mat grad;

    conv2D(input, kernelX, gradX);
    conv2D(input, kernelY, gradY);

    //cv::convertScaleAbs(gradX, absGradX);
    //cv::convertScaleAbs(gradY, absGradY);

    //cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, grad);
    grad = Magnitude_Gradient(gradX, gradY);

    return std::make_tuple(gradX, gradY, grad);
}

std::tuple<cv::Mat, cv::Mat, cv::Mat> applyPrewitt(const cv::Mat& input) {
    cv::Mat kernelX = (cv::Mat_<float>(3, 3) <<
                           -1, 0, 1,
                       -1, 0, 1,
                       -1, 0, 1);

    cv::Mat kernelY = (cv::Mat_<float>(3, 3) <<
                           -1, -1, -1,
                       0, 0, 0,
                       1, 1, 1);

    cv::Mat gradX, gradY;
    cv::Mat absGradX, absGradY;
    cv::Mat grad;

    conv2D(input, kernelX, gradX);
    conv2D(input, kernelY, gradY);

    //cv::convertScaleAbs(gradX, absGradX);
    //cv::convertScaleAbs(gradY, absGradY);

    //cv::addWeighted(absGradX, 1, absGradY, 1, 0, grad);
    grad = Magnitude_Gradient(gradX, gradY);
    return std::make_tuple(gradX, gradY, grad);
}

std::tuple<cv::Mat, cv::Mat, cv::Mat> applyRoberts(const cv::Mat& input) {
    cv::Mat kernelX = (cv::Mat_<float>(2, 2) <<
                           1, 0,
                       0, -1);

    cv::Mat kernelY = (cv::Mat_<float>(2, 2) <<
                           0, 1,
                       -1, 0);

    cv::Mat gradX, gradY;
    cv::Mat absGradX, absGradY;
    cv::Mat grad;
    cv::cvtColor(input, grad, COLOR_BGR2GRAY);

    conv2D(input, kernelX, gradX);
    conv2D(input, kernelY, gradY);

    //cv::convertScaleAbs(gradX, absGradX);
    //cv::convertScaleAbs(gradY, absGradY);

    //cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, grad);
    grad = Magnitude_Gradient(gradX, gradY);

    return std::make_tuple(gradX, gradY, grad);
}





std::tuple<cv::Mat, cv::Mat, cv::Mat> Detect_Edges_Canny(const cv::Mat& src, int lowThreshold = 20, int highThreshold = 50) {
    cv::Mat img = src.clone();
    cv::Mat output;

    // FIRST SMOOTH IMAGE
    cv::Mat grayImage;
    cv::Mat blurredImage;

    cv::cvtColor(img, grayImage, COLOR_BGR2GRAY);
    cv::GaussianBlur(grayImage, blurredImage, Size(5, 5), 1.4);
    cv::Mat sobel_x_kernel = (cv::Mat_<float>(3, 3) <<
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1);

    cv::Mat sobel_y_kernel = (cv::Mat_<float>(3, 3) <<
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1);
    cv::Mat gradX, gradY;
    // std::tuple<cv::Mat, cv::Mat, cv::Mat> result = applySobel(blurredImage);
    // gradX = std::get<0>(result);
    // gradY = std::get<1>(result);
    cv::filter2D(blurredImage, gradX, CV_32F, sobel_x_kernel);

    // Calculate gradient in Y direction
    cv::filter2D(blurredImage, gradY, CV_32F, sobel_y_kernel);
    cv::Mat magnitude_gradient = Magnitude_Gradient(gradX, gradY);
    cv::Mat phase_gradient = Phase_Gradient(gradX, gradY);

    // THEN SUPPRESS NON-MAXIMUM EDGES (with slightly relaxed comparison)
    cv::Mat suppressed = NonMaxSuppression(magnitude_gradient, phase_gradient);

    // THEN APPLY THRESHOLDING (adjusted thresholds)
    cv::Mat thresholded = DoubleThresholding(suppressed, lowThreshold, highThreshold);

    // THEN APPLY HYSTERESIS
    cv::Mat canny_edges = Hysteresis(thresholded);

    // Convert Canny edges to 8-bit
    // canny_edges.convertTo(canny_edges, CV_8U);

    // Return a tuple of the X gradient, Y gradient, and Canny edges
    return std::make_tuple(gradX, gradY, canny_edges);
}
cv::Mat NonMaxSuppression(cv::Mat& magnitude_gradient, cv::Mat& phase_gradient)
{
    cv::Mat suppressed = cv::Mat::zeros(cv::Size(magnitude_gradient.cols, magnitude_gradient.rows), magnitude_gradient.type());
    cv::Mat angles = phase_gradient.clone();
    for (int i = 1; i < angles.rows - 1; i++)
    {
        for (int j = 1; j < angles.cols - 1; j++)
        {
            if (angles.at<float>(i, j) < 0)
            {
                angles.at<float>(i, j) = angles.at<float>(i, j) + 360;
            }
            // # 0 degrees
            if ((angles.at<float>(i, j) >= 337.5 || angles.at<float>(i, j) < 22.5) || (angles.at<float>(i, j) >= 157.5 && angles.at<float>(i, j) < 202.5))
            {
                if (magnitude_gradient.at<float>(i, j) >= magnitude_gradient.at<float>(i, j + 1) && magnitude_gradient.at<float>(i, j) >= magnitude_gradient.at<float>(i, j - 1))
                {
                    suppressed.at<float>(i, j) = magnitude_gradient.at<float>(i, j);
                }
            }
            // # 45 degrees
            if ((angles.at<float>(i, j) >= 22.5 && angles.at<float>(i, j) < 67.5) || (angles.at<float>(i, j) >= 202.5 && angles.at<float>(i, j) < 247.5))
            {
                if (magnitude_gradient.at<float>(i, j) >= magnitude_gradient.at<float>(i - 1, j + 1) && magnitude_gradient.at<float>(i, j) >= magnitude_gradient.at<float>(i + 1, j - 1))
                {
                    suppressed.at<float>(i, j) = magnitude_gradient.at<float>(i, j);
                }
            }
            // # 90 degrees
            if ((angles.at<float>(i, j) >= 67.5 && angles.at<float>(i, j) < 112.5) || (angles.at<float>(i, j) >= 247.5 && angles.at<float>(i, j) < 292.5))
            {
                if (magnitude_gradient.at<float>(i, j) >= magnitude_gradient.at<float>(i - 1, j) && magnitude_gradient.at<float>(i, j) >= magnitude_gradient.at<float>(i + 1, j))
                {
                    suppressed.at<float>(i, j) = magnitude_gradient.at<float>(i, j);
                }
            }
            // # 135 degrees
            if ((angles.at<float>(i, j) >= 112.5 && angles.at<float>(i, j) < 157.5) || (angles.at<float>(i, j) >= 292.5 && angles.at<float>(i, j) < 337.5))
            {
                if (magnitude_gradient.at<float>(i, j) >= magnitude_gradient.at<float>(i - 1, j - 1) && magnitude_gradient.at<float>(i, j) >= magnitude_gradient.at<float>(i + 1, j + 1))
                {
                    suppressed.at<float>(i, j) = magnitude_gradient.at<float>(i, j);
                }
            }
        }
    }
    return suppressed;
}


cv::Mat DoubleThresholding(cv::Mat& suppressed, float lowThreshold, float highThreshold)
{
    cv::Mat thresholded = cv::Mat::zeros(Size(suppressed.cols, suppressed.rows), suppressed.type());

    for (int i = 0; i < suppressed.rows - 1; i++)
    {
        for (int j = 0; j < suppressed.cols - 1; j++)
        {
            if (suppressed.at<float>(i, j) > highThreshold)
            {
                thresholded.at<float>(i, j) = STRONG_EDGE;
            }
            else if ((suppressed.at<float>(i, j) < highThreshold) && (suppressed.at<float>(i, j) > lowThreshold))
            {
                thresholded.at<float>(i, j) = WEAK_EDGE;
            }
            else
            {
                thresholded.at<float>(i, j) = 0;
            }
        }
    }
    return thresholded;
}
cv::Mat Hysteresis(cv::Mat& thresholded)
{
    cv::Mat hysteresis = thresholded.clone();
    for (int i = 1; i < thresholded.rows - 1; i++)
    {
        for (int j = 1; j < thresholded.cols - 1; j++)
        {
            if (thresholded.at<float>(i, j) == WEAK_EDGE)
            {
                if ((thresholded.at<float>(i + 1, j - 1) == STRONG_EDGE) || (thresholded.at<float>(i + 1, j) == STRONG_EDGE) || (thresholded.at<float>(i + 1, j + 1) == STRONG_EDGE) || (thresholded.at<float>(i, j - 1) == STRONG_EDGE) || (thresholded.at<float>(i, j + 1) == STRONG_EDGE) || (thresholded.at<float>(i - 1, j - 1) == STRONG_EDGE) || (thresholded.at<float>(i - 1, j) == STRONG_EDGE) || (thresholded.at<float>(i - 1, j + 1) == STRONG_EDGE))
                {
                    hysteresis.at<float>(i, j) = STRONG_EDGE;
                }
                else
                {
                    hysteresis.at<float>(i, j) = 0;
                }
            }
        }
    }
    return hysteresis;
}
cv::Mat Magnitude_Gradient(const cv::Mat& gradient_x, const cv::Mat& gradient_y)
{
    cv::Mat Magnitude_Gradient = cv::Mat::zeros(cv::Size(gradient_x.cols, gradient_x.rows), gradient_x.type());
    for (int i = 0; i < gradient_x.rows; i++)
    {
        for (int j = 0; j < gradient_x.cols; j++)
        {
            Magnitude_Gradient.at<float>(i, j) = sqrt(pow(gradient_x.at<float>(i, j), 2) + pow(gradient_y.at<float>(i, j), 2));
        }
    }
    return Magnitude_Gradient;
}
cv::Mat Phase_Gradient(const cv::Mat& gradient_x, const cv::Mat& gradient_y)
{
    cv::Mat phase_gradient = cv::Mat::zeros(cv::Size(gradient_x.cols, gradient_y.rows), CV_32FC1);
    for (int i = 0; i < phase_gradient.rows; i++)
    {
        for (int j = 0; j < phase_gradient.cols; j++)
        {
            phase_gradient.at<float>(i, j) = atan2(gradient_y.at<float>(i, j), gradient_x.at<float>(i, j));
        }
    }
    phase_gradient = phase_gradient * 180 / M_PI;
    return phase_gradient;
}
