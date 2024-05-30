#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <random>
#include <cmath>

#include"source_code.h"




/*
    @brief:     Adds Gaussian noise to the given image.
    @param:     image   : A cv::Mat object representing the input image.
                mean    : The mean of the Gaussian distribution (default = 0).
                stddev  : The standard deviation of the Gaussian distribution (default = 25).
    @return:    A cv    ::Mat object representing the noisy image.
*/

Mat addGaussianNoise(const Mat& image, double mean, double stddev, const string& color) {
    Mat noisy_image = image.clone();
    default_random_engine generator;
    normal_distribution<double> distribution(mean, stddev);

    if (color == "c") { // RGB processing
        for (int y = 0; y < noisy_image.rows; ++y) {
            for (int x = 0; x < noisy_image.cols; ++x) {
                Vec3b* pixel = noisy_image.ptr<Vec3b>(y) + x;
                for (int c = 0; c < noisy_image.channels(); ++c) {
                    double noise = distribution(generator);
                    int new_intensity = (*pixel)[c] + noise;
                    (*pixel)[c] = saturate_cast<uchar>(new_intensity);
                }
            }
        }
    }
    else if (color == "g") { // Grayscale processing
        for (int y = 0; y < noisy_image.rows; ++y) {
            for (int x = 0; x < noisy_image.cols; ++x) {
                // Access pixel intensity directly for grayscale images
                uchar& pixel = noisy_image.at<uchar>(y, x);
                double noise = distribution(generator);
                int new_intensity = pixel + noise;
                pixel = saturate_cast<uchar>(new_intensity);
            }
        }
    }
    else {
        // Handle invalid color option
        cerr << "Invalid color option. Please choose 'c' for RGB or 'g' for grayscale." << endl;
    }

    return noisy_image;
}


/*
    @brief:     Adds uniform noise to the given image.
    @param:     image   : A cv::Mat object representing the input image.
                low     : The lower bound of the uniform distribution (default = 0).
                high    : The upper bound of the uniform distribution (default = 50).
    @return:    A cv    ::Mat object representing the noisy image.
*/

Mat addUniformNoise(const Mat& image, double low, double high, const string& color) {
    Mat noisy_image = image.clone();
    default_random_engine generator;
    uniform_real_distribution<double> distribution(low, high);

    if (color == "c") { // RGB processing
        for (int y = 0; y < noisy_image.rows; ++y) {
            for (int x = 0; x < noisy_image.cols; ++x) {
                Vec3b* pixel = noisy_image.ptr<Vec3b>(y) + x;
                for (int c = 0; c < noisy_image.channels(); ++c) {
                    double noise = distribution(generator);
                    int new_intensity = (*pixel)[c] + noise;
                    (*pixel)[c] = saturate_cast<uchar>(new_intensity);
                }
            }
        }
    }
    else if (color == "g") { // Grayscale processing
        for (int y = 0; y < noisy_image.rows; ++y) {
            for (int x = 0; x < noisy_image.cols; ++x) {
                // Access pixel intensity directly for grayscale images
                uchar& pixel = noisy_image.at<uchar>(y, x);
                double noise = distribution(generator);
                int new_intensity = pixel + noise;
                pixel = saturate_cast<uchar>(new_intensity);
            }
        }
    }
    else {
        // Handle invalid color option
        cerr << "Invalid color option. Please choose 'c' for RGB or 'g' for grayscale." << endl;
    }

    return noisy_image;
}


/*
    @brief:     Adds salt and pepper noise to the given image.
    @param:     image           : A cv::Mat object representing the input image.
                salt_prob       : The probability of adding salt noise (default = 0.01).
                pepper_prob     : The probability of adding pepper noise (default = 0.01).
    @return:    A cv::Mat object representing the noisy image.
*/

Mat addSaltAndPepperNoise(const Mat& image, float salt_prob, float pepper_prob, const string& color) {
    Mat noisy_image = image.clone();
    default_random_engine generator;
    uniform_real_distribution<double> distribution(0.0, 1.0);

    if (color == "c") { // RGB processing
        for (int y = 0; y < noisy_image.rows; ++y) {
            for (int x = 0; x < noisy_image.cols; ++x) {
                double random_val = distribution(generator);
                Vec3b* pixel = noisy_image.ptr<Vec3b>(y) + x;
                if (random_val < salt_prob)
                    *pixel = Vec3b(255, 255, 255); // Salt
                else if (random_val > 1 - pepper_prob)
                    *pixel = Vec3b(0, 0, 0);   // Pepper
            }
        }
    }
    else if (color == "g") { // Grayscale processing
        for (int y = 0; y < noisy_image.rows; ++y) {
            for (int x = 0; x < noisy_image.cols; ++x) {
                double random_val = distribution(generator);
                uchar& pixel = noisy_image.at<uchar>(y, x);
                if (random_val < salt_prob)
                    pixel = 255; // Salt
                else if (random_val > 1 - pepper_prob)
                    pixel = 0;   // Pepper
            }
        }
    }
    else {
        // Handle invalid color option
        cerr << "Invalid color option. Please choose 'c' for RGB or 'g' for grayscale." << endl;
    }

    return noisy_image;
}


/*
    @brief:      Applies a Gaussian filter to the given image.
    @param:      image          : A cv::Mat object representing the input image.
                 kernel_size    : The size of the Gaussian kernel (should be an odd number).
                 sigma          : The standard deviation of the Gaussian distribution.
    @return:     A cv::Mat object representing the filtered image.
*/

Mat applyGaussianFilter(const Mat& image, int kernel_size, double sigma, const string& color) {
    Mat filtered_image = image.clone();
    int border = kernel_size / 2;

    // Generate 1D Gaussian kernel
    vector<double> gaussian_kernel;
    for (int i = -border; i <= border; ++i) {
        double value = exp(-(i * i) / (2 * sigma * sigma)) / sqrt(2 * CV_PI * sigma * sigma);
        gaussian_kernel.push_back(value);
    }

    if (color == "c") { // RGB processing
        // Convolve rows with the kernel
        for (int y = 0; y < image.rows; ++y) {
            for (int x = border; x < image.cols - border; ++x) {
                Vec3d sum(0, 0, 0);
                for (int j = -border; j <= border; ++j) {
                    sum += image.at<Vec3b>(y, x + j) * gaussian_kernel[j + border];
                }
                filtered_image.at<Vec3b>(y, x) = sum;
            }
        }

        // Convolve columns with the kernel
        for (int x = 0; x < image.cols; ++x) {
            for (int y = border; y < image.rows - border; ++y) {
                Vec3d sum(0, 0, 0);
                for (int i = -border; i <= border; ++i) {
                    sum += filtered_image.at<Vec3b>(y + i, x) * gaussian_kernel[i + border];
                }
                filtered_image.at<Vec3b>(y, x) = sum;
            }
        }
    }
    else if (color == "g") { // Grayscale processing
        // Convolve rows with the kernel
        for (int y = 0; y < image.rows; ++y) {
            for (int x = border; x < image.cols - border; ++x) {
                double sum = 0.0;
                for (int j = -border; j <= border; ++j) {
                    sum += image.at<uchar>(y, x + j) * gaussian_kernel[j + border];
                }
                filtered_image.at<uchar>(y, x) = saturate_cast<uchar>(sum);
            }
        }

        // Convolve columns with the kernel
        for (int x = 0; x < image.cols; ++x) {
            for (int y = border; y < image.rows - border; ++y) {
                double sum = 0.0;
                for (int i = -border; i <= border; ++i) {
                    sum += filtered_image.at<uchar>(y + i, x) * gaussian_kernel[i + border];
                }
                filtered_image.at<uchar>(y, x) = saturate_cast<uchar>(sum);
            }
        }
    }
    else {
        // Handle invalid color option
        cerr << "Invalid color option. Please choose 'c' for RGB or 'g' for grayscale." << endl;
    }

    return filtered_image;
}


/*
    @brief:      Applies an average filter to the given image.
    @param:      image          : A cv::Mat object representing the input image.
                 kernel_size    : The size of the filter kernel (should be an odd number).
    @return:     A cv::Mat object representing the filtered image.
*/

Mat applyAverageFilter(const Mat& image, int kernel_size, const string& color) {
    Mat filtered_image = image.clone();
    int border = kernel_size / 2;

    if (color == "c") { // RGB processing
        // Iterate over image pixels
        for (int y = border; y < image.rows - border; ++y) {
            for (int x = border; x < image.cols - border; ++x) {
                // Compute average intensity for each channel
                int sum_b = 0, sum_g = 0, sum_r = 0;

                // Iterate over the kernel
                for (int i = -border; i <= border; ++i) {
                    for (int j = -border; j <= border; ++j) {
                        // Access pixel values using pointer arithmetic
                        const uchar* pixel = image.ptr<uchar>(y + i) + (x + j) * image.channels();
                        sum_b += pixel[0]; // Blue channel
                        sum_g += pixel[1]; // Green channel
                        sum_r += pixel[2]; // Red channel
                    }
                }

                // Compute averages
                int avg_b = sum_b / (kernel_size * kernel_size);
                int avg_g = sum_g / (kernel_size * kernel_size);
                int avg_r = sum_r / (kernel_size * kernel_size);

                // Set the center pixel to the average intensity for each channel
                uchar* filtered_pixel = filtered_image.ptr<uchar>(y) + x * image.channels();
                filtered_pixel[0] = avg_b; // Blue channel
                filtered_pixel[1] = avg_g; // Green channel
                filtered_pixel[2] = avg_r; // Red channel
            }
        }
    }
    else if (color == "g") { // Grayscale processing
        // Iterate over image pixels
        for (int y = border; y < image.rows - border; ++y) {
            for (int x = border; x < image.cols - border; ++x) {
                // Compute average intensity
                int sum = 0;

                // Iterate over the kernel
                for (int i = -border; i <= border; ++i) {
                    for (int j = -border; j <= border; ++j) {
                        // Access pixel intensity using at<uchar>
                        sum += image.at<uchar>(y + i, x + j);
                    }
                }

                // Compute average
                int avg = sum / (kernel_size * kernel_size);

                // Set the center pixel to the average intensity
                filtered_image.at<uchar>(y, x) = avg;
            }
        }
    }
    else {
        // Handle invalid color option
        cerr << "Invalid color option. Please choose 'c' for RGB or 'g' for grayscale." << endl;
    }

    return filtered_image;
}


/*
    @brief:      Applies a median filter to the given image.
    @param:      image: A cv::Mat object representing the input image.
                 kernel_size: The size of the filter kernel (should be an odd number).
    @return:     A cv::Mat object representing the filtered image.
*/

Mat applyMedianFilter(const Mat& image, int kernel_size, const string& color) {
    Mat filtered_image = image.clone();
    int border = kernel_size / 2;

    if (color == "c") { // RGB processing
        // Iterate over image pixels
        for (int y = border; y < image.rows - border; ++y) {
            for (int x = border; x < image.cols - border; ++x) {
                // Collect pixel values in the kernel region
                vector<int> blue_values, green_values, red_values;
                for (int i = -border; i <= border; ++i) {
                    for (int j = -border; j <= border; ++j) {
                        const Vec3b& pixel = image.at<Vec3b>(y + i, x + j);
                        blue_values.push_back(pixel[0]);
                        green_values.push_back(pixel[1]);
                        red_values.push_back(pixel[2]);
                    }
                }

                // Sort the pixel values
                sort(blue_values.begin(), blue_values.end());
                sort(green_values.begin(), green_values.end());
                sort(red_values.begin(), red_values.end());

                // Set the center pixel to the median intensity for each channel
                Vec3b& filtered_pixel = filtered_image.at<Vec3b>(y, x);
                filtered_pixel[0] = blue_values[blue_values.size() / 2]; // Blue channel
                filtered_pixel[1] = green_values[green_values.size() / 2]; // Green channel
                filtered_pixel[2] = red_values[red_values.size() / 2]; // Red channel
            }
        }
    }
    else if (color == "g") { // Grayscale processing
        // Iterate over image pixels
        for (int y = border; y < image.rows - border; ++y) {
            for (int x = border; x < image.cols - border; ++x) {
                // Collect pixel values in the kernel region
                vector<int> values;
                for (int i = -border; i <= border; ++i) {
                    for (int j = -border; j <= border; ++j) {
                        values.push_back(image.at<uchar>(y + i, x + j));
                    }
                }

                // Sort the pixel values
                sort(values.begin(), values.end());

                // Set the center pixel to the median intensity
                filtered_image.at<uchar>(y, x) = values[values.size() / 2];
            }
        }
    }
    else {
        // Handle invalid color option
        cerr << "Invalid color option. Please choose 'c' for RGB or 'g' for grayscale." << endl;
    }

    return filtered_image;
}







