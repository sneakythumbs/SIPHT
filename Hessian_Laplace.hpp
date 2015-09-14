#ifndef __HESSIAN_LAPLACE_HPP__
#define __HESSIAN_LAPLACE_HPP__

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

	class  Hessian_Laplace
	{
	public:

    double k;
    float threshold;
    int octaves;
    int intervals;
    int border;
    std::vector<double> sigma;
    int recursion;
    
    Hessian_Laplace( float _k = 0.04, float thresh = 0.0, int oct = 4, int inter = 3, double _sigma = 1.6 );
    Hessian_Laplace( const cv::Mat& img, float _k = 0.04, float thresh = 0.0, int oct = 4, int inter = 3, double _sigma = 1.6 );
    
    void initialise ( const cv::Mat& img );
    void detector( std::vector<cv::KeyPoint>& keypoints );
    void detector( cv::Mat& img, std::vector<cv::KeyPoint>& keypoints);
	
	protected:
	
	  void buildGaussPyr( const cv::Mat& img );
	  void buildHessianPyr();
	  bool isSpacialExtremum(int row, int col, int octave, int interval);
	  bool isScaleExtremum(int row, int col, int octave, int interval);
	  double getPixelLaplacian(const cv::Mat& img, int row, int col);
	  void createBaseImage(const cv::Mat& src, cv::Mat& dst,
					               const bool img_dbl, const double sigma);
    void deriv_2D(const cv::Mat& scale_img, cv::Mat& dI, int row, int col);					               
    void hessian_2D(const cv::Mat& scale_img, cv::Mat& H, int row, int col);					               
    void interp_step(const cv::Mat& scale_img, int row, int col, double& xr, double& xc);		               
		double interpContrast(const cv::Mat& scale_img, int row, int col, double& xr, double& xc);
		int interpSpacialExtremum(const cv::Mat& scale_img, int oct, int row, int col, cv::Point2f& coords);
		
	  
	  std::vector< std::vector<cv::Mat> > gaussPyramid;
	  std::vector< std::vector<cv::Mat> > hessianPyramid;

	};

} /* namespace pk */ 

#endif
