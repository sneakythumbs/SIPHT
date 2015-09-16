#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>
#include "sipht.hpp"
#include "pk.hpp"
#include "FAST_Harris.hpp"
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

    cv::Mat img1, img2, img1_colour, img2_colour, mask;
    img1 = cv::imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
    img1_colour = cv::imread( argv[1], CV_LOAD_IMAGE_COLOR);
   
    cv::Mat transform = (cv::Mat_<double>(2,3) << 2, 0, 0, 0, 0.5, 0);
      
    cv::Size dims;
    getSize(img1, transform, dims);
    warpAffine(img1, img2, transform, dims, CV_INTER_CUBIC, IPL_BORDER_CONSTANT);
    
    if ( !img1.data )
    {
        printf("No image data \n");
        return -1;
    }
    
    std::vector<cv::KeyPoint> img1_points, img2_points;
      
    FAST_Harris freddy(img1, 15, 1e10, 0.04, 10, 20);
    freddy.detector(img1_points);
    freddy.detector(img2, img2_points);
    
    std::cout << img1_points.size() << " " << img2_points.size() << std::endl;
    cv::Mat output, snoutput;

    cv::drawKeypoints(img1, img1_points, output, cv::Scalar::all(-1), 4);
    cv::drawKeypoints(img2, img2_points, snoutput, cv::Scalar::all(-1), 4);

    cv::namedWindow("Keypoints", CV_WINDOW_KEEPRATIO );
    cv::imshow("Keypoints", output);
    cv::namedWindow("Points", CV_WINDOW_KEEPRATIO );
    cv::imshow("Points", snoutput);
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
