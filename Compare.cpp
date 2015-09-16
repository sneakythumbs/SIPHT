#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>
#include "pk.hpp"
#include "Comparator.hpp"
#include "Harris_Laplace.hpp"
#include "Hessian_Laplace.hpp"
#include "Laplace.hpp"
#include "sipht.hpp"
#include "FAST_Laplace.hpp"
#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>

using namespace pk;

int main(int argc, char** argv )
{
  if ( argc < 7 | argc > 8 )
  {
    printf("usage: compare <Image_Path> <Output_Path> <Method_Name> <X_Scale> <X_Shear> <Y_Shear> <Y_Scale>\n");
    return -1;
  }
  Comparator check(argv[1], argv[2], argv[3], atof(argv[4]), atof(argv[5]), atof(argv[6]));
  
  std::vector<cv::KeyPoint> img1Points, img2Points;
  std::map <std::string, int> Methods;
  Methods["Harris_Laplace"] = 1;
  Methods["Hessian_Laplace"] = 2;
  Methods["Laplace"] = 3;
  Methods["MSER"] = 4;
  Methods["ORB"] = 5;
  Methods["SIFT"] = 6;
  Methods["SIPHT"] = 7;
  Methods["FAST"] = 8;
  Methods["Harris_Affine"] = 9;
  Methods["Harris_ICS"] = 10;
  Methods["all"] = 1;    
            
  std::string method = std::string(argv[3]);
  
  switch(Methods[method])
  {
//****************************************************************************//
    case 1:  
    {
      check.changeMethod("Harris_Laplace");
  
      Harris_Laplace harry(check.img1, 0, 0.04, 5e-6, 5, 10);
      harry.detector(img1Points);
      harry.detector(check.img2, img2Points);
  
      check.compare(img1Points, img2Points);
  
      img1Points.clear();
      img2Points.clear();
  
      if (method == "Harris_Laplace") break;
    }
//****************************************************************************//  
    case 2:
    {
      check.changeMethod("Hessian_Laplace");
  
      double edge = 10;
      double thresh = 5e-3;
      int octaves = 5;
      int intervals = 3;
    
      Hessian_Laplace hess(check.img1, edge, thresh, octaves, intervals);
      hess.detector(img1Points);
      hess.detector(check.img2, img2Points);
  
      check.compare(img1Points, img2Points);
  
      img1Points.clear();
      img2Points.clear();
      if (method == "Hessian_Laplace") break;
    }
//****************************************************************************//  
    case 3:
    {
      check.changeMethod("Laplace");
      
      double edge = 10;
      double thresh = 3e-1;
      int octaves = 5;
      int intervals = 3; 
      
      Laplace larry(check.img1, edge, thresh, octaves, intervals);
      larry.detector(img1Points);
      larry.detector(check.img2, img2Points);
  
      check.compare(img1Points, img2Points);
  
      img1Points.clear();
      img2Points.clear();
      if (method == "Laplace") break;
    }
//****************************************************************************//  
    case 4:
    {
      check.changeMethod("MSER");
  
      std::vector< std::vector<cv::Point> > mser1, mser2;
      cv::MSER misery;
      misery.minDiversity = 0.4f;
      misery.delta = 5;
      misery.maxVariation = 0.1;
      misery(check.img1, mser1, cv::Mat());

      std::vector<cv::Mat> mean;
      std::vector<cv::Mat> covar;
      std::vector<double> area;    
    
      fitGaussians(mser1, mean, covar, area);
      for (int it = 0; it < mean.size(); ++it)
      {
        cv::KeyPoint keypoint; 
        keypoint.pt.x = mean[it].at<double>(0,0);
        keypoint.pt.y = mean[it].at<double>(1,0);
        keypoint.size = sqrt(area[it]/M_PI);
        img1Points.push_back(keypoint);
      }

      mean.clear();
      covar.clear();
      area.clear();
  
      misery(check.img2, mser2, cv::Mat());
      
      fitGaussians(mser2, mean, covar, area);
      for (int it = 0; it < mean.size(); ++it)
      {
        cv::KeyPoint keypoint; 
        keypoint.pt.x = mean[it].at<double>(0,0);
        keypoint.pt.y = mean[it].at<double>(1,0);
        keypoint.size = sqrt(area[it]/M_PI);
        img2Points.push_back(keypoint);
      }
  
      check.compare(img1Points, img2Points);

/*//  
  cv::Mat output;
  cv::drawKeypoints(check.img2, img2Points, output, cv::Scalar::all(-1), 4);   
  cv::namedWindow("MSER", CV_WINDOW_KEEPRATIO );
  cv::imshow("MSER", output);
  cv::waitKey(0);
//*/  
      img1Points.clear();
      img2Points.clear();
      if (method == "MSER") break;
    }
//****************************************************************************// 
    case 5:
    {
      check.changeMethod("ORB");
  
      int nfeatures=500;
      float scaleFactor=1.4f;
      int nlevels=5;
      int edgeThreshold=31;
      int firstLevel=0;
      int WTA_K=2;
      int scoreType=0; //HARRIS_SCORE
      int patchSize=31;
    
      cv::ORB::CommonParams comm(scaleFactor, nlevels, edgeThreshold, firstLevel);
      cv::ORB orb(1000, comm);
      orb(check.img1, cv::Mat(), img1Points);
      orb(check.img2, cv::Mat(), img2Points);
  
      removeDuplicates(img1Points, 5e-1);
      removeDuplicates(img2Points, 5e-1);
  
      check.compare(img1Points, img2Points);
  
      img1Points.clear();
      img2Points.clear();
      if (method == "ORB") break;
    }
//****************************************************************************//   
    case 6:
    {

      check.changeMethod("SIFT");
  
      cv::SIFT::CommonParams common;
	    cv::SIFT::DetectorParams detector;
	    cv::SIFT::DescriptorParams descriptor;

	    cv::SIFT sift(common, detector, descriptor);

      sift(check.img1, cv::Mat(), img1Points);
      sift(check.img2, cv::Mat(), img2Points);
  
      removeDuplicates(img1Points, 5e-1);
      removeDuplicates(img2Points, 5e-1);
  
      check.compare(img1Points, img2Points);
 
      img1Points.clear();
      img2Points.clear();
      if (method == "SIFT") break;      
    }
//****************************************************************************//
    case 7:
    {
      check.changeMethod("SIPHT");
  
      pk::SIFT::CommonParams common;
	    pk::SIFT::DetectorParams detector(0.04, 10);
//      pk::SIFT::DetectorParams detector(0.04, 15);
	    pk::SIFT::DescriptorParams descriptor;

	    pk::SIFT sipht(common, detector, descriptor, check.transform);
      cv::Mat eye = cv::Mat::eye(2, 3, CV_64FC1);
      sipht(check.img1, cv::Mat(), img1Points, eye);
//  cv::Mat trans;
//  check.transform.convertTo( trans, CV_32FC1 );
      sipht(check.img2, cv::Mat(), img2Points, check.transform);
  
      check.compare(img1Points, img2Points);
 
      img1Points.clear();
      img2Points.clear(); 
     if (method == "SIPHT") break;    
    }
//****************************************************************************//
    case 8:
    {
      check.changeMethod("FAST");
  
      int fastThreshold = 15;
      int octaves = 4;
      int intervals = 5;

      FAST_Laplace freddy(check.img1, fastThreshold, octaves, intervals);
      freddy.detector(img1Points);
      freddy.detector(check.img2, img2Points);
     
      check.compare(img1Points, img2Points);
 
      img1Points.clear();
      img2Points.clear(); 
     if (method == "FAST") break;    
    }
//****************************************************************************//
    case 9:  
    {
      check.changeMethod("Harris_Affine");
  
      Harris_Laplace harry(check.img1, 1, 0.04, 5e-6, 5, 10);
      harry.detector(img1Points);
      harry.detector(check.img2, img2Points);
  
      check.compare(img1Points, img2Points);
  
      img1Points.clear();
      img2Points.clear();
  
      if (method == "Harris_Affine") break;
    }
//****************************************************************************//
    case 10:  
    {
      check.changeMethod("Harris_ICS");
  
      Harris_Laplace harry(check.img1, 2, 0.04, 5e-6, 5, 10);
      harry.detector(img1Points);
      harry.detector(check.img2, img2Points);
  
      check.compare(img1Points, img2Points);
  
      img1Points.clear();
      img2Points.clear();
  
      if (method == "Harris_ICS") break;
    }
//****************************************************************************//
  }
  return 0;
}
  
