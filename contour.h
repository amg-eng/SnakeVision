
#ifndef CONTOUR_H
#define CONTOUR_H

#include "opencv2/imgcodecs.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
// template<typename T>
// T clamp(T val, T minVal, T maxVal) {
//     return std::min(std::max(val, minVal), maxVal);
// }

enum ChainCode {
    East = 0,
    NorthEast = 1,
    North = 2,
    NorthWest = 3,
    West = 4,
    SouthWest = 5,
    South = 6,
    SouthEast = 7
};


using namespace cv;
using namespace std;

// Function declarations
double calcInternalEnergy(Point pt, Point prevPt, Point nextPt, double alpha);
double calcExternalEnergy(Mat img, Point pt, double beta);
double calcBalloonEnergy(Point pt, Point prevPt, double gamma);
double calcEnergy(Mat img, Point pt, Point prevPt, Point nextPt, double alpha, double beta, double gamma);
void moveCurve(Mat img, vector<Point>& curve, double alpha, double beta, double gamma);
void snake(Mat img, vector<Point>& curve, int numIterations, double alpha, double beta, double gamma);
vector<Point> initial_contour(Point center, int radius);
double points_distance(int x1, int y1, int x2, int y2);
double contour_area(vector<Point> points, int n_points);
double contour_perimeter(vector<Point> points, int n_points);
void display_area_perimeter(vector<Point> curve, Mat &outputImg);
std::vector<cv::Point> active_Contour_Model(cv::Mat inputMat, cv::Mat &outputMat, Mat white, cv::Point center, int radius, int numIterations, double alpha, double beta, double gamma);
Point getNeighbor(const Point &p, ChainCode direction);
int pixelValue(const vector<vector<int>> &image, const Point &p);
ChainCode findNextDirection(ChainCode currentDirection);
vector<ChainCode> chainCode(const vector<vector<int>> &image, const Point &startPoint);
Point findStartingPoint(const vector<vector<int>> &image);
vector<ChainCode> normalizeContour(const vector<ChainCode> &contour);
vector<vector<int>> createImageFromBoundary(const vector<Point> &boundary, int size);



// float elastic_func(cv::Point circlePoints[], int num_of_points, int dx, int dy, int current_index);
// void move_circle_points(cv::Point circlePoints[], cv::Mat edge, int num_of_points, float w_edge, float alpha);

#endif // CONTOUR_H
