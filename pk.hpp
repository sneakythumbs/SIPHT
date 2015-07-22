#ifndef __PK_HPP__
#define __PK_HPP__

#include <cstdlib>
#include <limits>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/core/internal.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"

/* width of border in which to ignore keypoints */
#define SIFT_IMG_BORDER 5

/* maximum steps of keypoint interpolation before failure */
#define SIFT_MAX_INTERP_STEPS 5

/** max feature descriptor length */
#define FEATURE_MAX_D 128


  /** holds feature data relevant to detection */
  struct detection_data
  {
      int r; // row
      int c; // col
      int octv;
      int intvl;
      double subintvl;
      double scl_octv;
  };

/**
   Structure to represent an affine invariant image feature.  The fields
   x, y, a, b, c represent the affine region around the feature:

   a(x-u)(x-u) + 2b(x-u)(y-v) + c(y-v)(y-v) = 1
*/
  struct feature
  {
      double x;                      /**< x coord */
      double y;                      /**< y coord */

      double scl;                    /**< scale of a Lowe-style feature */
      double ori;                    /**< orientation of a Lowe-style feature */

      int d;                         /**< descriptor length */
      double descr[FEATURE_MAX_D];   /**< descriptor */

      detection_data* feature_data;            /**< user-definable data */

      int class_id;
      float response;
  };

namespace pk
{
  bool is_extremum(const cv::Mat& scale_img, int row, int col);
  void deriv_2D(const cv::Mat& scale_img, cv::Mat& dI, int row, int col);
  void hessian_2D( const cv::Mat& scale_img, cv::Mat& H, int row, int col);
  void interp_step( const cv::Mat& scale_img, int row, int col, double& xr, double& xc );
  double interp_contr( const cv::Mat& scale_img, int row, int col, double& xr, double& xc );
  struct feature* new_feature( void );
  struct feature* interp_extremum(const cv::Mat& scale_img, int octv, int row, int col, int intvls, double contr_thr);
  bool is_too_edge_like(const cv::Mat& scale_img, int row, int col, int curv_thr);
  void scale_space_extrema(const cv::Mat& scale_img, int intvls, double contr_thr, int curv_thr, std::vector<cv::KeyPoint>& keypoints);
  void harrisCorner(cv::Mat& src, cv::Mat& dst, double sigma, double k);
  void hessianDeterminant(cv::Mat& src, cv::Mat& dst, double sigma);
  void fitGaussians(std::vector< std::vector<cv::Point> >& msers, std::vector<cv::Mat>& mean,
                    std::vector<cv::Mat>& covar, std::vector<double>& area);
  double GaussianDerivative1D(double x, double sigma, int order);
  void GaussDerivKernel1D(double sigma, int order, cv::Mat& kernel);
  void removeDuplicates(std::vector<cv::KeyPoint>& keypoints, double tol);
} /* End Namespce pk */

#endif
