#include "edgeDetection.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/* ******************************************************************************************************************
 *********************************************   HOUGH LINE   *******************************************************
 ********************************************************************************************************************/
Mat Hough_Line_Transform(const Mat& image, int threshold, double lineResolution)
{
    // Convert the image to grayscale using our function
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Perform edge detection using canny detector using our function
    Mat edges;

    // canny take the arge : input image, output image, min threashold, max threshold
    Canny(grayImage, edges, 150, 200);

    int rows = grayImage.rows;
    int cols = grayImage.cols;

    // Define Hough parameters
    // the maximum possible value of the rho parameter in the Hough space.
    // The maximum value of rho is generally the diagonal length of the image, hence the use of Pythagoras's theorem.
    int max_rho = sqrt(rows * rows + cols * cols);

    // NOTICE : the smaller the steps the better the result
    // represents the distance from the origin to the closest point on a detected line
    int rho_step = 1;

    // This variable sets the step size for discretizing the theta parameter space
    double theta_step = CV_PI / 180;

    // This variable represents the number of bins used to discretize the theta parameter space.
    int theta_bins = 180 * lineResolution;

    // This variable represents the number of bins used to discretize the rho parameter space.
    int rho_bins = 2 * max_rho / rho_step + 1;

    // Initialize accumulator matrix with zeros and add to it the edges and its neighbors
    Mat accumulator(rho_bins, theta_bins, CV_32SC1, Scalar(0));

    // Loop on the image to get dges and accumulate the accumulatir vector
    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            if (edges.at<uchar>(y, x) > 0)
            {
                // Loop over theta values
                for (int t = 0; t < theta_bins; t++)
                {
                    double theta = t * theta_step;
                    // Hough line better parameterization
                    double rho = x * cos(theta) + y * sin(theta);
                    int rho_bin = round((rho + max_rho) / rho_step);
                    accumulator.at<int>(rho_bin, t)++;
                }
            }
        }
    }

    // Vector that represent the edges of the accumulator to draw the lines
    vector<Vec2f> lines;
    for (int r = 1; r < rho_bins - 1; r++)
    {
        for (int t = 1; t < theta_bins - 1; t++)
        {
            // Check For accumulator threshold
            if (accumulator.at<int>(r, t) > threshold)
            {
                // Check the 8-neighbors of the peak point "Local Maximum"
                if (
                    // top-left neighbor
                    accumulator.at<int>(r, t) > accumulator.at<int>(r - 1, t - 1) &&
                    // top neighbor
                    accumulator.at<int>(r, t) > accumulator.at<int>(r - 1, t) &&
                    // top-right neighbor
                    accumulator.at<int>(r, t) > accumulator.at<int>(r - 1, t + 1) &&
                    // left neighbor
                    accumulator.at<int>(r, t) > accumulator.at<int>(r, t - 1) &&
                    // Right neighbor
                    accumulator.at<int>(r, t) > accumulator.at<int>(r, t + 1) &&
                    // bottom-left neighbor
                    accumulator.at<int>(r, t) > accumulator.at<int>(r + 1, t - 1) &&
                    // bottom neighbor
                    accumulator.at<int>(r, t) > accumulator.at<int>(r + 1, t) &&
                    // bottom-right neighbor
                    accumulator.at<int>(r, t) > accumulator.at<int>(r + 1, t + 1)
                    )
                {

                    double rho = (r * rho_step) - max_rho;
                    double theta = t * theta_step;
                    lines.push_back(Vec2f(rho, theta));
                }
            }
        }
    }
    Mat result = image.clone();
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + max_rho * (-b));
        pt1.y = cvRound(y0 + max_rho * (a));
        pt2.x = cvRound(x0 - max_rho * (-b));
        pt2.y = cvRound(y0 - max_rho * (a));
        line(result, pt1, pt2, Scalar(0, 0, 255), 1, LINE_AA);
    }

    return result;

}


/* *******************************************************************************************************************
 *********************************************   HOUGH ELLIPSE *******************************************************
 *********************************************************************************************************************/
void boundry_detection(Mat img, vector<vector<Point>>& contours)
{
    // Clear the existing contours to prepare for new ones
    contours.clear();

    // Apply Canny edge detection to the input image
    cv::Mat edges;
    Canny(img , edges , 100,600);

    // Traverse through the image to find contours
    for (int y = 0; y < edges.rows; y++) {
        for (int x = 0; x < edges.cols; x++) {
            // If the pixel intensity is nonzero (i.e., an edge pixel)
            if (edges.at<uchar>(y, x) != 0) {
                // Create a new contour and initialize it with the current pixel
                std::vector<cv::Point> contour;
                cv::Point current(x, y);
                contour.push_back(current);
                edges.at<uchar>(current) = 0; // Mark pixel as visited

                bool foundNext = true;
                // Traverse through neighboring pixels to form the contour
                while (foundNext) {
                    foundNext = false;
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            // Exclude the central pixel to avoid revisiting it
                            if (dx != 0 || dy != 0) {
                                cv::Point next(current.x + dx, current.y + dy);
                                // Check if the next pixel is within the image bounds
                                if (next.x >= 0 && next.x < edges.cols && next.y >= 0 && next.y < edges.rows) {
                                    // If the next pixel is an edge pixel
                                    if (edges.at<uchar>(next) != 0) {
                                        current = next;
                                        contour.push_back(current); // Add the pixel to the contour
                                        edges.at<uchar>(current) = 0; // Mark pixel as visited
                                        foundNext = true; // Set flag to continue searching for next pixel
                                        break;
                                    }
                                }
                            }
                        }
                        if (foundNext)
                            break; // Exit loop if the next pixel is found
                    }
                }
                // Add the formed contour to the list of contours
                contours.push_back(contour);
            }
        }
    }
}


void drawEllipses(Mat& img, const vector<Vec6d>& ellipses)
{
    if (img.empty()) {
        std::cerr << "Error: Input image is empty." << std::endl;
        return;
    }

    for (const auto& ellipse : ellipses)
    {

        // Ensure ellipse center is within the image bounds
        cv::Point center(ellipse[0], ellipse[1]);
        if (center.x < 0 || center.x >= img.cols || center.y < 0 || center.y >= img.rows) {
            std::cerr << "Error: Ellipse center is outside image bounds." << std::endl;
            continue; // Skip ellipse outside image bounds
        }

        cv::ellipse(img, center,
            cv::Size(ellipse[2], ellipse[3]),
            ellipse[4], 0, 360,
            cv::Scalar(0, 0, 255), 2);
    }
}

Mat HoughEllipse(Mat img, vector<Vec6d>& ellipses, int threshold, int minRadius, int maxRadius)
{
    // Apply the boundry detection to get the edges
    vector<vector<Point>> contours;
    boundry_detection(img, contours);

    for (size_t i = 0; i < contours.size(); i++)
    {
        if (contours[i].size() >= threshold) // Minimum number of points required for fitting an ellipse OR Threshold
        {
            // fit ellipse : calculates the best-fitting ellipse using a least-squares algorithm.
            // It contains information such as the center, size, and orientation of the ellipse.
            RotatedRect ellipse = fitEllipse(contours[i]);
            // vector 6d
            /*
             * ellipse.center.x: X-coordinate of the center of the ellipse.
                ellipse.center.y: Y-coordinate of the center of the ellipse.
                ellipse.size.width / 2.0: Half of the width of the ellipse (major axis).
                ellipse.size.height / 2.0: Half of the height of the ellipse (minor axis).
                ellipse.angle: The angle of rotation of the ellipse in degrees.
                0: This value is not explicitly used here but could be used to store additional information if needed.
             * */
            Vec6d ellipseParams(ellipse.center.x, ellipse.center.y, ellipse.size.width / 2.0, ellipse.size.height / 2.0, ellipse.angle, 0);
            ellipses.push_back(ellipseParams);
        }
    }
    // Mat image = img.clone();
    drawEllipses(img, ellipses);
    return img;
}


/* *******************************************************************************************************************
 *********************************************   HOUGH CIRCLE  *******************************************************
 *********************************************************************************************************************/
Mat Hough_circle_transform(const Mat &image, int threshold, int min_radius /* =10 */, int max_radius /* =200 */, int canny_min_thresold /* = 100 */, int canny_max_thresold /* = 200 */, int thetas)
{
    // Convert the image to grayscale
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Perform edge detection using Canny algorithm
    Mat edges;
    Canny(grayImage, edges, canny_min_thresold, canny_max_thresold);

    // Get image dimensions
    int rows = grayImage.rows;
    int cols = grayImage.cols;

    // Define Hough parameters
    int num_thetas = thetas;
    double dtheta = 360.0 / num_thetas;

    // Initialize 3D Accumulator
    vector<vector<vector<int>>> accumulator(cols, vector<vector<int>>(rows, vector<int>(max_radius - min_radius + 1, 0)));

    // Loop over edge points
    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            // Check if it's an edge point
            if (edges.at<uchar>(y, x) > 0)
            {
                // Loop over radius values
                for (int r = min_radius; r <= max_radius; r++)
                {
                    // Loop over theta values
                    for (int theta = 0; theta < num_thetas; theta++)
                    {
                        // Convert theta to radians
                        double rad = theta * (CV_PI / 180.0);
                        // Calculate center coordinates of the circle
                        int x_center = cvRound(x + r * cos(rad));
                        int y_center = cvRound(y + r * sin(rad));

                        // Check that the center is within image bounds
                        if (x_center >= 0 && x_center < cols && y_center >= 0 && y_center < rows)
                        {
                            // Increment the accumulator
                            accumulator[x_center][y_center][r - min_radius]++;
                        }
                    }
                }
            }
        }
    }

    // Create output image
    Mat circles_img = image.clone();

    // Loop over the accumulator to find circles
    for (int x = 0; x < cols; x++)
    {
        for (int y = 0; y < rows; y++)
        {
            for (int r = min_radius; r <= max_radius; r++)
            {
                // Check if the accumulator value exceeds the threshold
                if (accumulator[x][y][r - min_radius] > threshold)
                {
                    // Draw circle on the output image
                    circle(circles_img, Point(x, y), r, Scalar(0, 0, 255), 1, LINE_AA);
                }
            }
        }
    }

    return circles_img; // Return the image with detected circles
}


// int main() {
//     int threshold = 7;
//     int minRadius = 20;
//     int maxRadius = 80;

//     Mat imageLines = imread("Screenshot 2023-02-28 231508.png", IMREAD_COLOR);
//     double lineResolution = 1;
//     Mat detectedLines = Hough_Line_Transform(imageLines, 100, lineResolution);
//     imshow("Detected Lines", detectedLines);


//     Mat image = imread("ellipses.png"); // Load your image
//     //Mat image = imread("5.jpg"); // Load your image
//     if (image.empty()) {
//         cerr << "Could not read the image!" << endl;
//         return -1;
//     }
//     std::vector<cv::Vec6d> ellipses;
//     HoughEllipse(image, ellipses, threshold, minRadius, maxRadius);
//     imshow("Original Image with Detected Ellipses", image);
    


//     Mat imageCircles = imread("0.png");
//     Mat detected_circls = Hough_circle_transform(imageCircles, 25, 30, 100, 100);
//     imshow("Detected circles", detected_circls);
//     waitKey(0);
//     return 0;
// }
