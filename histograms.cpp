
#include "histograms.h"
#include "histogramFun.h"
using namespace std;


void onTrackbar(int, void*) {
    // Empty callback for trackbar
}



/**
 * Normalizes the input histogram matrix to the range [0, 255].
 *
 * @param histogram The input histogram matrix.
 * @return The normalized histogram matrix.
 */
cv::Mat normalizeHistogram_n(cv::Mat& histogram) {
    cv::normalize(histogram, histogram, 0, 255, cv::NORM_MINMAX);
    return histogram;
}



/**
 * Converts the input image to grayscale using the luminance formula.
 *
 * @param input The input image matrix.
 * @return The grayscale image matrix.
 */
cv::Mat convertToGrayScale(const cv::Mat& input) {
    cv::Mat output(input.rows, input.cols, CV_8UC1);

    // Iterate through each pixel
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            // Get the pixel value
            cv::Vec3b pixel = input.at<cv::Vec3b>(y, x);
            // Compute grayscale value using formula: Y = 0.299R + 0.587G + 0.114B
            unsigned char gray = (unsigned char)(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
            // Set the grayscale value for all channels
            output.at<unsigned char>(y, x) = gray;
        }
    }

    return output;
}



/**
 * Saves the histogram image to a file.
 *
 * @param filename The name of the file to save the histogram image.
 * @param histogramImage The histogram image matrix.
 */
void saveHistogramImage(const std::string& filename, const cv::Mat& histogramImage) {
    cv::imwrite(filename, histogramImage);
}






/**
 * Normalizes the input grayscale image to the range [0, 255].
 *
 * @param image The input grayscale image matrix.
 * @return The normalized grayscale image matrix.
 */
cv::Mat normalzeHistogram_n(cv::Mat image) {

    float image_min = image.at<uchar>(0, 0), image_max = 0;
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            if (image.at<uchar>(row, col) <= image_min) {
                image_min = image.at<uchar>(row, col);
            }
            if (image.at<uchar>(row, col) >= image_max) {
                image_max = image.at<uchar>(row, col);
            }
        }
    }
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            image.at<uchar>(row, col) = ((float)(image.at<uchar>(row, col) - image_min) / (image_max - image_min)) * 255;
        }
    }

    return image;
}


/**
 * Calculates the cumulative distribution function (CDF) of the input histogram.
 *
 * @param hist The input histogram matrix.
 * @return The computed CDF matrix.
 */
cv::Mat calculateCDF(const cv::Mat& hist) {
    cv::Mat cdf = cv::Mat::zeros(1, hist.cols, CV_32F);
    cdf.at<float>(0) = hist.at<float>(0);

    for (int i = 1; i < hist.cols; ++i) {
        cdf.at<float>(i) = cdf.at<float>(i - 1) + hist.at<float>(i);
    }

    cdf /= cdf.at<float>(hist.cols - 1);

    return cdf;
}



/**
 * Plots the cumulative distribution function (CDF) using the specified color.
 *
 * @param cdf The CDF matrix to plot.
 * @param color The color for plotting the CDF.
 * @return The CDF image matrix.
 */
cv::Mat plotCDF(const cv::Mat& cdf, const cv::Scalar& color) {
    int cdfWidth = 512;
    int cdfHeight = 400;

    cv::Mat cdfImage(cdfHeight, cdfWidth, CV_8UC3, cv::Scalar(255, 255, 255));

    // Plot the CDF curve
    for (int i = 1; i < cdf.cols; i++) {
        cv::line(cdfImage,
            cv::Point(cdfWidth * (i - 1) / cdf.cols, cdfHeight - cvRound(cdf.at<float>(i - 1) * cdfHeight)),
            cv::Point(cdfWidth * i / cdf.cols, cdfHeight - cvRound(cdf.at<float>(i) * cdfHeight)),
            color, 2, 8, 0);
    }

    // Add labels
    cv::putText(cdfImage, "Intensity", cv::Point(cdfWidth / 2, cdfHeight - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    cv::putText(cdfImage, "Probability", cv::Point(10, cdfHeight / 2), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA, false);

    //// Fill the area under the CDF curve
    //std::vector<cv::Point> curvePoints;
    //for (int i = 1; i < cdf.cols; i++) {
    //    int x1 = cdfWidth * (i - 1) / cdf.cols;
    //    int x2 = cdfWidth * i / cdf.cols;
    //    int y1 = cdfHeight - cvRound(cdf.at<float>(i - 1) * cdfHeight);
    //    int y2 = cdfHeight - cvRound(cdf.at<float>(i) * cdfHeight);

    //    curvePoints.push_back(cv::Point(x1, y1));
    //    curvePoints.push_back(cv::Point(x2, y2));
    //}

    //const cv::Point* curvePointsData = curvePoints.data();
    //int curvePointsCount = (int)curvePoints.size() / 2;
    //const cv::Point* curvePointsArray[1] = { curvePointsData };

    //cv::fillPoly(cdfImage, curvePointsArray, &curvePointsCount, 1, color, cv::LINE_8);

    return cdfImage;
}




/**
 * Calculates the grayscale cumulative distribution function (CDF) of the input image.
 *
 * @param grayscaleImage The input grayscale image matrix.
 * @return The computed grayscale CDF matrix.
 */
cv::Mat calculateGrayscaleCDF(const cv::Mat& grayscaleImage) {
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };

    cv::Mat hist = cv::Mat::zeros(1, histSize, CV_32F);

    for (int y = 0; y < grayscaleImage.rows; ++y) {
        for (int x = 0; x < grayscaleImage.cols; ++x) {
            hist.at<float>(grayscaleImage.at<uchar>(y, x)) += 1.0;
        }
    }

    cv::Mat cdf = cv::Mat::zeros(1, hist.cols, CV_32F);
    cdf.at<float>(0) = hist.at<float>(0);

    for (int i = 1; i < hist.cols; ++i) {
        cdf.at<float>(i) = cdf.at<float>(i - 1) + hist.at<float>(i);
    }

    cdf /= cdf.at<float>(hist.cols - 1);

    return cdf;
}



/**
 * Plots the grayscale cumulative distribution function (CDF) with specified line and fill colors.
 *
 * @param title The title for the CDF plot window.
 * @param cdf The grayscale CDF matrix to plot.
 * @param lineColor The color for plotting the CDF curve.
 * @param fillColor The color for filling the area under the CDF curve.
 */
void plotGrayscaleCDF(const std::string& title, const cv::Mat& cdf, const cv::Scalar& lineColor, const cv::Scalar& fillColor) {
    int cdfWidth = 512;
    int cdfHeight = 400;

    cv::Mat cdfImage(cdfHeight, cdfWidth, CV_8UC3, cv::Scalar(255, 255, 255));

    for (int i = 1; i < cdf.cols; i++) {
        cv::line(cdfImage, cv::Point(cdfWidth * (i - 1) / cdf.cols, cdfHeight - cvRound(cdf.at<float>(i - 1) * cdfHeight)),
            cv::Point(cdfWidth * i / cdf.cols, cdfHeight - cvRound(cdf.at<float>(i) * cdfHeight)), lineColor, 2, 8, 0);

        /////////////////////////////////// color the area under the CDF curve////////////////////////
        // Fill the area under the curve
        cv::rectangle(cdfImage, cv::Point(cdfWidth * (i - 1) / cdf.cols, cdfHeight),
            cv::Point(cdfWidth * i / cdf.cols, cdfHeight - cvRound(cdf.at<float>(i) * cdfHeight)), fillColor, CV_16F);
    }

    cv::imshow(title, cdfImage);
}



/**
 * Displays the normalized and equalized versions of the input image.
 *
 * @param image The input image matrix.
 * @return The normalized and equalized image matrix.
 */
cv::Mat displayNormalizedEqualizedImages(const cv::Mat& image) {
    cv::Mat normalizedImage, equalizedImage;

    cv::normalize(image, normalizedImage, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::equalizeHist(normalizedImage, equalizedImage);

    return normalizedImage;
    
}








/**
 * Calculates the histogram of the input image for the specified channel.
 *
 * @param input The input image matrix.
 * @param channel The channel for which to calculate the histogram.
 * @return The computed histogram matrix.
 */

void calculateHistograms(Mat& image, Mat& histR, Mat& histG, Mat& histB) {
    // Split the image into its channels
    vector<Mat> channels;
    split(image, channels);

    // Calculate histograms for each channel
    int histSize = 256;
    float range[] = { 0, 255 };
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;

    calcHist(&channels[0], 1, 0, Mat(), histB, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&channels[1], 1, 0, Mat(), histG, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&channels[2], 1, 0, Mat(), histR, 1, &histSize, &histRange, uniform, accumulate);
}


/**
 * Plots the histogram using the specified color, width, and height.
 *
 * @param histogram The histogram matrix to plot.
 * @param color The color for plotting the histogram.
 * @param histWidth The width of the histogram plot.
 * @param histHeight The height of the histogram plot.
 * @return The histogram image matrix.
 */

void drawHistograms(Mat& histR, Mat& histG, Mat& histB, Mat& histImageR, Mat& histImageG, Mat& histImageB, int histSize, double& maxIntensity) {
    // Find maximum intensity value among all channels

    minMaxLoc(histR, NULL, &maxIntensity);
    minMaxLoc(histG, NULL, &maxIntensity);
    minMaxLoc(histB, NULL, &maxIntensity);

    // Normalize histograms
    int histWidth = 600, histHeight = 400;
    int binWidth = cvRound((double)histWidth / histR.rows);
    normalize(histR, histR, 0, histSize, NORM_MINMAX, -1, Mat());
    normalize(histG, histG, 0, histSize, NORM_MINMAX, -1, Mat());
    normalize(histB, histB, 0, histSize, NORM_MINMAX, -1, Mat());

    // Draw histograms
    histImageR = Mat(histHeight, histWidth, CV_8UC3, Scalar(255, 255, 255));
    histImageG = Mat(histHeight, histWidth, CV_8UC3, Scalar(255, 255, 255));
    histImageB = Mat(histHeight, histWidth, CV_8UC3, Scalar(255, 255, 255));

    // Draw histograms for R, G, and B channels
    for (int i = 0; i < histSize; i++) {
        line(histImageR, Point((binWidth * i) + 60, histHeight - 40 - cvRound(histR.at<float>(i))),
            Point((binWidth * i) + 60, histHeight - 40),
            Scalar(0, 0, 255), 1, LINE_AA, 0);

        line(histImageG, Point((binWidth * i) + 60, histHeight - 40 - cvRound(histG.at<float>(i))),
            Point((binWidth * i) + 60, histHeight - 40),
            Scalar(0, 255, 0), 1, LINE_AA, 0);

        line(histImageB, Point((binWidth * i) + 60, histHeight - 40 - cvRound(histB.at<float>(i))),
            Point((binWidth * i) + 60, histHeight - 40),
            Scalar(255, 0, 0), 1, LINE_AA, 0);
    }
}


/**
 * Adds axes to the histogram image for better visualization.
 *
 * @param histogramImage The histogram image matrix.
 * @return The histogram image with axes added.
 */

void addLabels(Mat& histImageR, Mat& histImageG, Mat& histImageB, int histSize, double maxIntensity, int histHeight, int histWidth) {
    // Draw X-axis and labels
    line(histImageR, Point(0, histHeight - 40), Point(histWidth - 1, histHeight - 40), Scalar(0, 0, 0));
    line(histImageG, Point(0, histHeight - 40), Point(histWidth - 1, histHeight - 40), Scalar(0, 0, 0));
    line(histImageB, Point(0, histHeight - 40), Point(histWidth - 1, histHeight - 40), Scalar(0, 0, 0));


    putText(histImageR, "Intensity", Point(0.5 * histWidth, histHeight - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1.5, LINE_AA);
    putText(histImageG, "Intensity", Point(0.5 * histWidth, histHeight - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1.5, LINE_AA);
    putText(histImageB, "Intensity", Point(0.5 * histWidth, histHeight - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1.5, LINE_AA);
    // Add labels
    for (int i = 0; i < 6; ++i) {
        int x = i * (histWidth - 1) / 5;
        putText(histImageR, to_string(i * 50), Point(x + 58, histHeight - 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1.5, LINE_AA);
        putText(histImageG, to_string(i * 50), Point(x + 58, histHeight - 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1.5, LINE_AA);
        putText(histImageB, to_string(i * 50), Point(x + 58, histHeight - 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1.5, LINE_AA);
    }

    // Draw Y-axis and labels
    line(histImageR, Point(58, histHeight - 20), Point(58, 0), Scalar(0, 0, 0), 1.5, LINE_AA);
    line(histImageG, Point(58, histHeight - 20), Point(58, 0), Scalar(0, 0, 0), 1.5, LINE_AA);
    line(histImageB, Point(58, histHeight - 20), Point(58, 0), Scalar(0, 0, 0), 1.5, LINE_AA);

    putText(histImageR, "Frequency", Point(10, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1.5, LINE_AA);
    putText(histImageG, "Frequency", Point(10, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1.5, LINE_AA);
    putText(histImageB, "Frequency", Point(10, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1.5, LINE_AA);

    for (int i = 0; i < 6; ++i) {
        int y = i * (histHeight - 20) / 5;
        putText(histImageR, to_string(i * (int)maxIntensity / 5), Point(15, histHeight - 20 - y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1.5, LINE_AA);
        putText(histImageG, to_string(i * (int)maxIntensity / 5), Point(15, histHeight - 20 - y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1.5, LINE_AA);
        putText(histImageB, to_string(i * (int)maxIntensity / 5), Point(15, histHeight - 20 - y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1.5, LINE_AA);
    }
}



/**
 * Calculates the histogram of the input image for the specified channel.
 *
 * @param input The input image matrix.
 * @param channel The channel for which to calculate the histogram.
 * @return The computed histogram matrix.
 */

cv::Mat calculateHistogram(const cv::Mat& input, int channel) {
    int histSize = 256;  // Number of bins
    float range[] = { 0, 256 };
    const float* histRange = { range };

    cv::Mat hist = cv::Mat::zeros(1, histSize, CV_32F);

    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            hist.at<float>(input.at<cv::Vec3b>(y, x)[channel]) += 1.0;
        }
    }

    return hist;
}



/**
 * Plots the histogram using the specified color, width, and height.
 *
 * @param histogram The histogram matrix to plot.
 * @param color The color for plotting the histogram.
 * @param histWidth The width of the histogram plot.
 * @param histHeight The height of the histogram plot.
 * @return The histogram image matrix.
 */
cv::Mat plotHistogram(const cv::Mat& histogram, const cv::Scalar color, int histWidth, int histHeight) {
    // Plot the histogram

    int binWidth = cvRound((double)histWidth / histogram.cols);
    cv::Mat histImage(histHeight, histWidth, CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw histogram lines
    for (int i = 1; i < histogram.cols; i++) {
        cv::line(histImage, cv::Point(binWidth * (i - 1), histHeight - cvRound(histogram.at<float>(i - 1))),
            cv::Point(binWidth * (i), histHeight - cvRound(histogram.at<float>(i))),
            color, 2, 8, 0);
    }

    // Fill the area under the histogram curve
    std::vector<cv::Point> curvePoints;
    for (int i = 0; i < histogram.cols; i++) {
        curvePoints.push_back(cv::Point(binWidth * i, histHeight - cvRound(histogram.at<float>(i))));
    }
    curvePoints.push_back(cv::Point(binWidth * (histogram.cols - 1), histHeight));
    curvePoints.push_back(cv::Point(0, histHeight));
    const cv::Point* curvePointsData = &curvePoints[0];
    int numberOfPoints = (int)curvePoints.size();
    cv::fillPoly(histImage, &curvePointsData, &numberOfPoints, 1, color, 8);


    line(histogram, Point(0, histHeight - 40), Point(histWidth - 1, histHeight - 40), Scalar(0, 0, 0));
   

    putText(histogram, "Intensity", Point(0.5 * histWidth, histHeight - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1.5, LINE_AA);
 

    for (int i = 0; i < 6; ++i) {
        int x = i * (histWidth - 1) / 5;
        putText(histogram, to_string(i * 50), Point(x + 58, histHeight - 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1.5, LINE_AA);
  }

    // Draw Y-axis and labels
    line(histogram, Point(58, histHeight - 20), Point(58, 0), Scalar(0, 0, 0), 1.5, LINE_AA);


    putText(histogram, "Frequency", Point(10, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1.5, LINE_AA);


    for (int i = 0; i < 6; ++i) {
        int y = i * (histHeight - 20) / 5;
        putText(histogram, to_string(i / 5), Point(15, histHeight - 20 - y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1.5, LINE_AA);
   }

    return histImage;
}



cv::Mat addAxesToHistogram(const cv::Mat& histogramImage) {
    // Create a blank canvas to draw the histogram with axes
    cv::Mat canvas(histogramImage.rows + 100, histogramImage.cols + 100, CV_8UC3, cv::Scalar(255, 255, 255));

    // Copy the histogram image onto the canvas
    cv::Rect roi(cv::Point(50, 50), histogramImage.size());
    histogramImage.copyTo(canvas(roi));

    // Add x-axis label and line
    cv::line(canvas, cv::Point(50, canvas.rows - 50), cv::Point(canvas.cols - 50, canvas.rows - 50), cv::Scalar(0, 0, 0), 2);
    cv::putText(canvas, "Intensity", cv::Point(canvas.cols / 2 - 50, canvas.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0, cv::Scalar(0, 0, 0), 1);

    // Add y-axis label and line
    cv::line(canvas, cv::Point(50, canvas.rows - 50), cv::Point(50, 50), cv::Scalar(0, 0, 0), 2);

    // Rotate the "Count" label by 90 degrees
    cv::Point textOrigin(10, canvas.rows / 2);
    cv::Point2f rotatedTextOrigin;
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(textOrigin, 90, 1.0);
    cv::transform(std::vector<cv::Point2f>{textOrigin}, std::vector<cv::Point2f>{rotatedTextOrigin}, rotationMatrix);
    cv::putText(canvas, "Count", rotatedTextOrigin, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

    return canvas;
}