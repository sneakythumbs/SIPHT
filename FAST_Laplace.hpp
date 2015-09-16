#ifndef __FAST_LAPLACE_HPP__
#define __FAST_LAPLACE_HPP__

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/core/internal.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"

#include "pk.hpp"

#include <cstdio>
#include <limits>
#include <vector>
#include <cstdlib>
#include <iostream>

namespace pk
{

	class  FAST_Laplace
	{
	public:

    float threshold;
    int octaves;
    int intervals;
    int border;
    std::vector<double> sigma;
    int recursion;
    
    FAST_Laplace( float thresh = 0.0, int oct = 4, int inter = 3, double _sigma = 1.6 );
    FAST_Laplace( const cv::Mat& img, float thresh = 0.0, int oct = 4, int inter = 3, double _sigma = 1.6 );
    
    void initialise ( const cv::Mat& img );
    void detector( std::vector<cv::KeyPoint>& keypoints );
    void detector( cv::Mat& img, std::vector<cv::KeyPoint>& keypoints);
	
	protected:
	
	  void buildGaussPyr( const cv::Mat& img );
	  bool isScaleExtremum(int row, int col, int octave, int interval);
	  double getPixelLaplacian(const cv::Mat& img, int row, int col);
	  void createBaseImage(const cv::Mat& src, cv::Mat& dst,
					               const bool img_dbl, const double sigma);

	  
	  std::vector< std::vector<cv::Mat> > gaussPyramid;
	  

	};

} /* namespace pk */ 

#endif
