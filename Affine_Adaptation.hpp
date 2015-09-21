#ifndef __AFFINE_ADAPTATION_HPP__
#define __AFFINE_ADAPTATION_HPP__

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/core/internal.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"

#include "pk.hpp"
#include "ics.h"

#include <algorithm>
#include <string>

namespace pk
{

  /*
   * Elliptic region around interest cv::Point
   */
  class Elliptic_KeyPoint : public cv::KeyPoint 
  {
    public:

	  cv::Point2f		    centre;
	  cv::Size_<float> 	axes;

	  double 			      phi;
	  float 			      size;
	  float 			      si;

	  cv::Mat 			    transf;

  	Elliptic_KeyPoint();
	  Elliptic_KeyPoint(cv::Point centre, double phi, cv::Size axes, float size, float si);

	  virtual ~Elliptic_KeyPoint();
  };

  class Affine_Adaptation
  {
    public:
    
    int method;
    std::string icsScriptLocation;
    std::string tempLocation;
    
    Affine_Adaptation(int method = 0);
    Affine_Adaptation(int method, std::string scriptLocation, std::string outputLocation);
    bool calcAffineAdaptation(const cv::Mat & image, Elliptic_KeyPoint& keyPoint, int iterations);
    void keyPointToElliptic(cv::KeyPoint & keypoint, Elliptic_KeyPoint & ellipticKeypoint);
    void ellipticToKeyPoint(Elliptic_KeyPoint & ellipticKeypoint, cv::KeyPoint & keypoint);
    
    protected:
    
    void calcSecondMomentMatrix(const cv::Mat & dx2, const cv::Mat & dxy, const cv::Mat & dy2, cv::Point p, cv::Mat& M);
    float selIntegrationScale(const cv::Mat & image, float si, cv::Point c);
    float selDifferentiationScale(const cv::Mat & image, cv::Mat & Lxm2smooth, cv::Mat & Lxmysmooth, cv::Mat & Lym2smooth, float si, cv::Point c);
    float calcSecondMomentSqrt(const cv::Mat & dx2, const cv::Mat & dxy, const cv::Mat & dy2, cv::Point p, cv::Mat& Mk);
    float normMaxEval(cv::Mat & U, cv::Mat& uVal, cv::Mat& uVect);
    void deriv_2D(const cv::Mat& img, cv::Mat& dI, int row, int col);
    void hessian_2D( const cv::Mat& img, cv::Mat& H, int row, int col);
    
    void svdInv(cv::Mat &inp);
    void normalizeTransformationMatrix(cv::Mat &src, cv::Mat &dst);
    float executeICSscript(cv::Mat &ics);
    void calculateImageGradients( cv::Mat& img, double integrationScale, std::vector<cv::Point2d>& gradients );
    void writeOutGradients(std::vector<cv::Point2d>& gradients);    
    float calculateUnmixingMatrix( cv::Mat& warpedImagePatch, cv::Mat& ics, double integrationScale, double derivativeScale);
  };
};



#endif
