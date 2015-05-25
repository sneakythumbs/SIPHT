#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <cmath>
#include "sipht.hpp"

using namespace cv;

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: skewer <Image_Path> \n");
        return -1;
    }

    Mat img, mask;
    img = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );


    if ( !img.data )
    {
        printf("No image data \n");
        return -1;
    }

    std::vector<KeyPoint> cv_points, pk_points;

  	SIFT::CommonParams common;
  	SIFT::DetectorParams detector;
	  SIFT::DescriptorParams descriptor;

  	SIFT sift(common, detector, descriptor);
	
    pk::SIFT::CommonParams pk_common;
  	pk::SIFT::DetectorParams pk_detector;
  	pk::SIFT::DescriptorParams pk_descriptor;

  	pk::SIFT sipht(pk_common, pk_detector, pk_descriptor);

	  cv::Mat cv_descriptors, pk_descriptors;

    cv::Mat transform = cv::Mat::eye(2,3,CV_64F);
    
    sift(img, mask, cv_points, cv_descriptors);
    sipht(img, mask, pk_points, pk_descriptors, transform);
    
    long cv_found = 0, pk_found = 0, matching = 0;
    double tol = 1;
    for (auto j : pk_points)
      ++pk_found;
    for (auto i : cv_points)
    {
        ++cv_found;              
        for (auto j : pk_points)
        {
            Point3f orig(i.pt.x, i.pt.y, i.angle);
            Point3f modif(j.pt.x, j.pt.y, j.angle);
            if (norm(orig - modif) < tol)
                ++matching;
        }
    } 
    std::cout << "original method found:\t" << cv_found << std::endl;
    std::cout << "modified method found:\t" << pk_found << std::endl;
    std::cout << "number of matching points:\t" << matching << std::endl;
    
    Mat output;

    drawKeypoints(img, cv_points, output, Scalar::all(-1), 4);
    namedWindow("Keypoints", CV_WINDOW_KEEPRATIO );
    imshow("Keypoints", output);
//    waitKey(0);
    
    drawKeypoints(img, pk_points, output, Scalar::all(-1), 4);
    namedWindow("PKeypoints", CV_WINDOW_KEEPRATIO );
    imshow("PKeypoints", output);
    waitKey(0);

    
    return 0;
}
