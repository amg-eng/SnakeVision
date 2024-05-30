#include "thresholding.h"


/**
 * Applies global thresholding to the input image based on a threshold value.
 * Pixels with intensities greater than the threshold value are set to maximum_value (255),
 * while pixels with intensities less than or equal to the threshold value are set to minimum_value (0).
 *
 * @param inputImage The input image to be thresholded.
 * @param thresholdValue The threshold value for thresholding.
 * @param maximum_value The maximum value to set for pixels above the threshold.
 * @param minimum_value The minimum value to set for pixels below or equal to the threshold.
 * @return The thresholded image.
 */
Mat globalThreshold(const Mat& inputImage, int thresholdValue, unsigned char maximum_value, unsigned char minimum_value) {
    Mat result = inputImage.clone();

    for (int row = 0; row < inputImage.rows; ++row) {
        for (int column = 0; column < inputImage.cols; ++column) {
            result.at<uchar>(row, column) = (inputImage.at<uchar>(row, column) > thresholdValue) ? 255 : 0;
        }
    }

    return result;
}


/**
 * Applies local adaptive mean thresholding to the input image using a specified kernel size and constant.
 * The threshold for each pixel is computed based on the mean intensity of its local neighborhood.
 *
 * @param inputImage The input image to be thresholded.
 * @param kernalSize The size of the local neighborhood kernel.
 * @param constant The constant value subtracted from the mean for threshold calculation.
 * @param maximum_value The maximum value to set for pixels above the local threshold.
 * @param minimum_value The minimum value to set for pixels below or equal to the local threshold.
 * @return The thresholded image using local adaptive mean thresholding.
 */
Mat localAdaptiveMeanThreshold(const Mat& inputImage, int kernalSize, int constant, unsigned char maximum_value, unsigned char minimum_value) {
    Mat result = inputImage.clone();
    int halfKernalSize = kernalSize / 2;

    for (int imageRows = halfKernalSize; imageRows < inputImage.rows - halfKernalSize; ++imageRows) {
        for (int imageColumn = halfKernalSize; imageColumn < inputImage.cols - halfKernalSize; ++imageColumn) {
            int sum = 0;

            for (int kernalFirstIndex = -halfKernalSize; kernalFirstIndex <= halfKernalSize; ++kernalFirstIndex) {
                for (int l = -halfKernalSize; l <= halfKernalSize; ++l) {
                    sum += inputImage.at<uchar>(imageRows + kernalFirstIndex, imageColumn + l);
                }
            }

            int mean = sum / (kernalSize * kernalSize);
            result.at<uchar>(imageRows, imageColumn) = (inputImage.at<uchar>(imageRows, imageColumn) > (mean - constant)) ? 255 : 0;
        }
    }

    return result;
}



/**
 * Adds padding to the input image based on the specified padding size.
 * This function creates a new image with padding around the input image.
 *
 * @param inputImage The input image to which padding will be added.
 * @param paddingSize The size of the padding to be added around the image.
 * @return The padded image.
 */
Mat adaptivePaddingFunction(const Mat& inputImage, int paddingSize) {
    Mat result(inputImage.rows + 2 * paddingSize, inputImage.cols + 2 * paddingSize, CV_8UC1, Scalar(0));
    inputImage.copyTo(result(Rect(paddingSize, paddingSize, inputImage.cols, inputImage.rows)));

    for (int i = 0; i < paddingSize; ++i) {
        result.row(i).copyTo(result.row(i + inputImage.rows));
    }

    for (int j = 0; j < paddingSize; ++j) {
        result.col(j).copyTo(result.col(j + inputImage.cols));
    }

    return result;
}


/**
 * Computes the local mean of pixel intensities in the input image using a specified kernel size.
 * Each pixel in the output image is replaced with the mean intensity of its local neighborhood.
 *
 * @param inputImage The input image for local mean calculation.
 * @param kernalSize The size of the local neighborhood kernel for mean calculation.
 * @return The image with local mean intensity values.
 */
Mat localThresholdMeanCalculation(const Mat& inputImage, int kernalSize) {
    Mat result = inputImage.clone();
    int halfKernalSize = kernalSize / 2;

    for (int imageRows = halfKernalSize; imageRows < inputImage.rows - halfKernalSize; ++imageRows) {
        for (int imageColumn = halfKernalSize; imageColumn < inputImage.cols - halfKernalSize; ++imageColumn) {
            int sum = 0;

            for (int kernalFirstIndex = -halfKernalSize; kernalFirstIndex <= halfKernalSize; ++kernalFirstIndex) {
                for (int kernalSecondIndex = -halfKernalSize; kernalSecondIndex <= halfKernalSize; ++kernalSecondIndex) {
                    sum += inputImage.at<uchar>(imageRows + kernalFirstIndex, imageColumn + kernalSecondIndex);
                }
            }

            result.at<uchar>(imageRows, imageColumn) = sum / (kernalSize * kernalSize);
        }
    }

    return result;
}


/**
 * Applies local Gaussian thresholding to the input image using a specified block size and constant.
 * This function computes a weighted mean within a local neighborhood using Gaussian weighting.
 * Thresholding is then applied based on the weighted mean and the constant value.
 *
 * @param inputImage The input image to be thresholded.
 * @param outputImage The output image where the thresholded result will be stored.
 * @param blockSize The size of the local neighborhood block for Gaussian weighting.
 * @param C The constant value subtracted from the weighted mean for threshold calculation.
 */
void applyLocalGaussianThreshold(const cv::Mat& inputImage, cv::Mat& outputImage, int blockSize, double C) {
    // Convert the input image to grayscale
    cv::Mat grayscaleImage;
    cv::cvtColor(inputImage, grayscaleImage, cv::COLOR_BGR2GRAY);

    // Ensure outputImage has the same size and type as the input grayscaleImage
    outputImage.create(grayscaleImage.size(), grayscaleImage.type());

    // Compute the half size of the block
    int halfBlockSize = blockSize / 2;

    // Create a kernel for Gaussian weighting
    cv::Mat kernel = cv::getGaussianKernel(blockSize, -1, CV_64F);
    cv::Mat kernel2d = kernel * kernel.t();

    // Iterate through each pixel of the image
    for (int y = 0; y < grayscaleImage.rows; ++y) {
        for (int x = 0; x < grayscaleImage.cols; ++x) {
            // Calculate the bounding box for the local neighborhood
            int startX = std::max(0, x - halfBlockSize);
            int endX = std::min(grayscaleImage.cols - 1, x + halfBlockSize);
            int startY = std::max(0, y - halfBlockSize);
            int endY = std::min(grayscaleImage.rows - 1, y + halfBlockSize);

            // Compute the weighted sum of pixel intensities in the local neighborhood
            double sum = 0.0;
            double weightSum = 0.0;
            for (int ny = startY; ny <= endY; ++ny) {
                for (int nx = startX; nx <= endX; ++nx) {
                    double weight = kernel2d.at<double>(ny - startY, nx - startX);
                    sum += static_cast<double>(grayscaleImage.at<uchar>(ny, nx)) * weight;
                    weightSum += weight;
                }
            }
            double mean = sum / weightSum;

            // Apply thresholding
            outputImage.at<uchar>(y, x) = (grayscaleImage.at<uchar>(y, x) > mean - C) ? 255 : 0;
        }
    }

    cv::imshow("gaus", outputImage);
}


/**
 * Applies Gaussian blur to the input grayscale image.
 * This function smooths the image using Gaussian filtering with a specified kernel size.
 *
 * @param inputImage The input grayscale image to be blurred.
 * @return The blurred image.
 */Mat gaussianBlur(const Mat& inputImage) {
    Mat blurredImage;
    GaussianBlur(inputImage, blurredImage, Size(5, 5), 0);
    return blurredImage;
}


 /**
  * Applies Gaussian thresholding to the input grayscale image based on a threshold value.
  * Pixels with intensities greater than the threshold value are set to 255 (white),
  * while pixels with intensities less than or equal to the threshold value are set to 0 (black).
  *
  * @param inputImage The input grayscale image to be thresholded.
  * @param thresholdValue The threshold value for thresholding.
  * @return The thresholded image.
  */Mat gaussianThreshold(const Mat& inputImage, int thresholdValue) {
    Mat thresholded;
    threshold(inputImage, thresholded, thresholdValue, 255, THRESH_BINARY);
    return thresholded;
}