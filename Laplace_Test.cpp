#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>
#include "sipht.hpp"
#include "pk.hpp"
#include "Laplace.hpp"
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
   

    if ( !img1.data )
    {
        printf("No image data \n");
        return -1;
    }
    
    std::vector<cv::KeyPoint> img1_points, img2_points;
    
    double edge = 10;
    double thresh = 5e0;
    int octaves = 5;
    int intervals = 3;
    
    Laplace larry(img1, edge, thresh, octaves, intervals);
    larry.detector(img1_points);
    
    
    cv::SIFT::CommonParams com;
    cv::SIFT::DetectorParams det;
    cv::SIFT::DescriptorParams des;
	  cv::SIFT sift(com, det, des);
	  
	  SIFT::CommonParams common(octaves, intervals);
	  SIFT::DetectorParams detector(thresh, edge);
	  SIFT::DescriptorParams descriptor(1.0/3, false, false);
  	SIFT sipht(common, detector, descriptor);
    cv::Mat identity = cv::Mat::eye(2,3,CV_64F);

	  cv::Mat img_descriptors;

//    sift(img1, mask, img2_points, img_descriptors);
    sipht(img1, mask, img2_points, img_descriptors, identity);
    cv::Mat output, output2;

    cv::drawKeypoints(img1, img1_points, output, cv::Scalar::all(-1), 4);
    cv::drawKeypoints(img1, img2_points, output2, cv::Scalar::all(-1), 4);    

    cv::namedWindow("Laplace", CV_WINDOW_KEEPRATIO );
    cv::imshow("Laplace", output);
    cv::imwrite("Laplace_graffiti.png", output);
//    cv::namedWindow("sift", CV_WINDOW_KEEPRATIO );
    cv::imshow("sift", output2);
    cv::waitKey(0);
    
    return 0;
}


