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
    
    int nfeatures=1000;
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
    orb(img2, mask, img2_points);
    
    for (auto& point : img1_points)
    {
      double sigma = std::pow(comm.scale_factor_, point.octave);
      point.size = sigma * sigma * 5;    
    }    
    for (auto& point : img2_points)
    {
      double sigma = std::pow(comm.scale_factor_, point.octave);
      point.size = sigma * sigma * 5;
//      std::cout << "response: " << point.response * std::pow(comm.scale_factor_, point.octave) << std::endl;
    }
/*/    
    for ( auto it = img1_points.begin(); it < img1_points.end();)
    {
      if (it->response * std::pow(comm.scale_factor_, it->octave) < 5e-3)
      {
        it = img1_points.erase(it);      
      }
      else
      {
      ++it;
      }
    }
//*/    



    cv::Mat output, output2;

    cv::drawKeypoints(img1, img1_points, output, cv::Scalar::all(-1), 4);   
    cv::drawKeypoints(img2, img2_points, output2, cv::Scalar::all(-1), 4);  
    cv::namedWindow("korb", CV_WINDOW_KEEPRATIO );
    cv::imshow("korb", output);
//    cv::imwrite("orb_graffiti.png", output);
    cv::namedWindow("zorb", CV_WINDOW_KEEPRATIO );
    cv::imshow("zorb", output2);    
    cv::waitKey(0);
    
    return 0;
}

void getSize(cv::Mat& img, cv::Mat& inTrans, cv::Size& size)
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
    pt = inTrans * pt;
    dims[0] = std::max(pt.at<double>(0,0), dims[0]);
    dims[1] = std::max(pt.at<double>(1,0), dims[1]);
    dims[2] = std::min(pt.at<double>(0,0), dims[2]);
    dims[3] = std::min(pt.at<double>(1,0), dims[3]);
  }
  
  int cols = 2*(static_cast<int>((dims[0] - dims[2])/2));
  int rows = 2*(static_cast<int>((dims[1] - dims[3])/2));
  std::cout << img.rows << " " << img.cols << std::endl;
  std::cout << rows << " " << cols << std::endl;
  size = cv::Size(cols,rows);
}
