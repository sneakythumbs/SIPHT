#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>
#include "sipht.hpp"
#include "pk.hpp"
#include "Harris_Laplace.hpp"
#ifdef __cplusplus
#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#endif

using namespace pk;

void getSize(cv::Mat& img, cv::Mat& trans, cv::Size& size);

int main(int argc, char** argv )
{
    if ( argc < 6 | argc > 7 )
    {
        printf("usage: skewer <Image_Path> <Output_File> <X_Scale> <X_Shear> <Y_Shear> <Y_Scale>\n");
        return -1;
    }

    cv::Mat img1, img2, img1_colour, img2_colour, mask;
    img1 = cv::imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
    img1_colour = cv::imread( argv[1], CV_LOAD_IMAGE_COLOR);

    double xScale = atof(argv[3]);
    double xShear = tan ( atof(argv[4]) );
    double yShear = tan ( atof(argv[5]) );
    double yScale = 1/xScale;
    if (7 == argc) yScale = atof(argv[6]);
    
    cv::Mat transform = (cv::Mat_<double>(2,3) << xScale, xShear, 0, yShear, yScale, 0);
    cv::Mat identity = cv::Mat::eye(2,3,CV_64F);
    
    cv::Size dims;
    
    getSize(img1, transform, dims);
    warpAffine(img1, img2, transform, dims, 
                   CV_INTER_CUBIC ,IPL_BORDER_CONSTANT);
    warpAffine(img1_colour, img2_colour, transform, dims,
                                        CV_INTER_CUBIC ,IPL_BORDER_CONSTANT);

    if ( !img1.data | !img2.data )
    {
        printf("No image data \n");
        return -1;
    }
    
    std::vector<cv::KeyPoint> img1_points, img2_points;
      
    Harris_Laplace harry(img1, 0.04, 0.7);
    harry.detector(img1_points);
    
    cv::Mat output;

    cv::drawKeypoints(img1, img1_points, output, cv::Scalar::all(-1), 4);
//    cv::cornerHarris(img1, output, 3, 1, 0.04);
//    cv::normalize( output, output, 0, 255, 32, CV_32FC1, cv::Mat() );
//    cv::convertScaleAbs( output, output );
//    std::cout << output << std::endl;
    cv::namedWindow("Keypoints", CV_WINDOW_KEEPRATIO );
    cv::imshow("Keypoints", output);
    cv::waitKey(0);
    
    return 0;
}

void getSize(cv::Mat& img, cv::Mat& trans, cv::Size& size)
{
    std::vector<cv::Mat> cnrs;
    cnrs.push_back((cv::Mat_<double>(3,1) << 0, 0, 1));
    cnrs.push_back((cv::Mat_<double>(3,1) << img.cols, 0, 1));
    cnrs.push_back((cv::Mat_<double>(3,1) << 0, img.rows, 1));
    cnrs.push_back((cv::Mat_<double>(3,1) << img.cols, img.rows, 1));
    double max = std::numeric_limits<double>::max();
    double dims[4] {0,0,max,max};
    for (auto&& pt : cnrs)
    {
      pt = trans * pt;
      dims[0] = std::max(pt.at<double>(0,0), dims[0]);
      dims[1] = std::max(pt.at<double>(1,0), dims[1]);
      dims[2] = std::min(pt.at<double>(0,0), dims[2]);
      dims[3] = std::min(pt.at<double>(1,0), dims[3]);
    }
    
    int cols = 2*(static_cast<int>((dims[0] - dims[2])/2));
    int rows = 2*(static_cast<int>((dims[1] - dims[3])/2));
 
    size = cv::Size(cols,rows);

}