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
    misery(img1, msers, cv::Mat());

    cv::Mat output = img1, output2;
//    int grey = 0;
    std::vector<cv::Mat> mean;
    std::vector<cv::Mat> covar;
    std::vector<double> area;    
    for (auto& region : msers)
    {
      int x = 0, y = 0;
      double length = region.size();
      area.push_back(length);
      for (auto& pt : region)
      {
        x += pt.x;
        y += pt.y;

      }
      cv::Mat temp = (cv::Mat_<double>(2,1) << x/length, y/length);
      mean.push_back(temp);
      cv::circle(output, cv::Point(temp.at<double>(0,0), temp.at<double>(1,0)), 10, cv::Scalar::all(100));
      
      double varx = 0, vary = 0, varxy = 0;
      for (auto& pt : region)
      {
        varx += (pt.x - temp.at<double>(0,0)) * (pt.x - temp.at<double>(0,0));
        vary += (pt.y - temp.at<double>(1,0)) * (pt.y - temp.at<double>(1,0));
        varxy += (pt.x - temp.at<double>(0,0)) * (pt.y - temp.at<double>(1,0));
      }
      temp = (cv::Mat_<double>(2,2) << varx/length, varxy/length, varxy/length, vary/length);
      covar.push_back(temp);

    }



//    cv::drawKeypoints(img1, img1_points, output, cv::Scalar::all(-1), 4);   

    cv::namedWindow("Laplace", CV_WINDOW_KEEPRATIO );
    cv::imshow("Laplace", output);
    cv::waitKey(0);
    
    return 0;
}


