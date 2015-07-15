#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>

#include "sipht.hpp"
#include "pk.hpp"

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
    
    int nfeatures=500;
    float scaleFactor=1.4f;
    int nlevels=5;
    int edgeThreshold=31;
    int firstLevel=0;
    int WTA_K=2;
    int scoreType=0; //HARRIS_SCORE
    int patchSize=31;
    
    cv::ORB::CommonParams comm(scaleFactor, nlevels, edgeThreshold, firstLevel);
//    cv::ORB orb(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, 
//                WTA_K, scoreType, patchSize);
    cv::ORB orb(500, comm);
    orb(img1, mask, img1_points);
    
    for (auto& point : img1_points)
    {
      double sigma = std::pow(comm.scale_factor_, point.octave);
      point.size = sigma * sigma * 5;
    }


    cv::Mat output, output2;

    cv::drawKeypoints(img1, img1_points, output, cv::Scalar::all(-1), 4);   

    cv::namedWindow("korb", CV_WINDOW_KEEPRATIO );
    cv::imshow("korb", output);
    cv::waitKey(0);
    
    return 0;
}


