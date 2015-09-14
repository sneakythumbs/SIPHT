#ifndef __LAPLACE_HPP__
#define __LAPLACE_HPP__

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

	class  Laplace
	{
	public:

    int curveThresh;
    float threshold;
    int octaves;
    int intervals;
    int border;
    std::vector<double> sigma;
    int recursion;
    
    Laplace( int _curve = 10, float thresh = 0.0, int oct = 4, int inter = 3, double _sigma = 1.6 );
    Laplace( const cv::Mat& img, int _curve = 10, float thresh = 0.0, int oct = 4, int inter = 3, double _sigma = 1.6 );
    
    void initialise ( const cv::Mat& img );
    void detector( std::vector<cv::KeyPoint>& keypoints );
    void detector( cv::Mat& img, std::vector<cv::KeyPoint>& keypoints);
	
	protected:
	
	  void buildGaussPyr( const cv::Mat& img );
	  void buildLaplacePyr();
	  bool isLocalExtremum(int row, int col, int octave, int interval);
	  void createBaseImage(const cv::Mat& src, cv::Mat& dst,
					               const bool img_dbl, const double sigma);
    void deriv_2D(const cv::Mat& scale_img, cv::Mat& dI, int row, int col);					               
    void hessian_2D(cv::Mat& H, int oct, int inter, int row, int col);
    void deriv_3D( cv::Mat& dI, int oct, int inter, int row, int col);					               
    void hessian_3D(cv::Mat& H, int oct, int inter, int row, int col);    					               
    void interp_step(int oct, int inter, int row, int col, double& xi, double& xr, double& xc);		               
		double interpContrast(int oct, int inter, int row, int col, double& xi, double& xr, double& xc);
		int interpExtremum(int oct, int inter, int row, int col, cv::KeyPoint& point, cv::Point3f& coords);
		bool isTooEdgeLike(int oct, int inter, int row, int col);
		
	  
	  std::vector< std::vector<cv::Mat> > gaussPyramid;
	  std::vector< std::vector<cv::Mat> > laplacePyramid;

	};

} /* namespace pk */ 

#endif
