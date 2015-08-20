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
    
    
    std::vector< std::vector<cv::Point> > msers;
    cv::MSER misery;
    misery.minDiversity = 0.4f;
    misery.delta = 8;
    misery.maxVariation = 0.1;
    misery(img1, msers, cv::Mat());

    cv::Mat output = img1, output2;

    std::vector<cv::Mat> mean;
    std::vector<cv::Mat> covar;
    std::vector<double> area;
    
    fitGaussians(msers, mean, covar, area);

  for (int reg = 0; reg < msers.size(); ++reg)
  {
    std::cout << area[reg] << " " << reg << std::endl;
//    cv::circle(output, cv::Point(mean[reg].at<double>(0,0), mean[reg].at<double>(1,0)), sqrt(area[reg]/M_PI), cv::Scalar::all(100));
    cv::KeyPoint keypoint;
    keypoint.pt.x = mean[reg].at<double>(0,0);
    keypoint.pt.y = mean[reg].at<double>(1,0);
    keypoint.size = sqrt(area[reg]/M_PI);
    img1_points.push_back(keypoint);
  }

    cv::drawKeypoints(img1, img1_points, output, cv::Scalar::all(-1), 4);   

    cv::namedWindow("Laplace", CV_WINDOW_KEEPRATIO );
    cv::imshow("Laplace", output);
//    cv::imwrite("mser_graffiti.png", output);
    cv::waitKey(0);
    
    return 0;
}


