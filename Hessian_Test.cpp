#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>

#include "sipht.hpp"
#include "pk.hpp"
#include "Hessian_Laplace.hpp"

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
    double thresh = 5e-2;
    int octaves = 5;
    int intervals = 3;
    
    Hessian_Laplace hess(img1, edge, thresh, octaves, intervals);
    hess.detector(img1_points);
    

    cv::Mat output, output2;

    cv::drawKeypoints(img1, img1_points, output, cv::Scalar::all(-1), 4);   

    cv::namedWindow("Hessian_Laplace", CV_WINDOW_KEEPRATIO );
    cv::imshow("Hessian_Laplace", output);
//    cv::imwrite("Hessian_Laplace_graffiti.png", output);
    cv::waitKey(0);
    
    return 0;
}


