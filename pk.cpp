#include "pk.hpp"
#include <cstdlib>
#include <limits>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/core/internal.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"

namespace pk
{
/*
  Determines whether a pixel is a local extremum by comparing it to it's
  3x3 pixel neighborhood.

  @param scale_img one level of the DoG pyramid
  @param row pixel row
  @param col pixel's image column

  @return Returns 1 if the specified pixel is an extremum (max or min) among
    it's 3x3 pixel neighborhood.
*/
  bool is_extremum(const cv::Mat& scale_img, int row, int col)
  {
    double val = scale_img.at<float>(row,col);
  
    /* check for maximum */
    if( val > 0 )
      {
        for (int y = -1; y <= 1; ++y )
          for (int x = -1; x <= 1; ++x )
            if (val < scale_img.at<float>(row + y, col + x))
              return false;
      }

    /* check for minimum */
    else
      {
        for (int y = -1; y <= 1; ++y )
          for (int x = -1; x <= 1; ++x )
  				if (val > scale_img.at<float>(row + y, col + x))
              return false;
      }
  
    return true;
  }

/*
  Computes the partial derivatives in x, and y of a pixel in one DoG
  scale space level

  @param scale_img one level of the DoG pyramid
  @param row pixel's image row
  @param col pixel's image column

  @return Returns the vector of partial derivatives for pixel I
    { dI/dx, dI/dy }^T as a CvMat&
*/
  void deriv_2D(const cv::Mat& scale_img, cv::Mat& dI, int row, int col)
  {
    double dx, dy;
  
    dx = ( scale_img.at<float>(row,col+1) -
           scale_img.at<float>(row,col-1) ) / 2.0;
  	dy = ( scale_img.at<float>(row+1,col) -
  	       scale_img.at<float>(row-1,col) ) / 2.0;
  
    dI = cv::Mat( 2, 1, CV_64FC1 );
    dI.at<double>(0,0) = dx;
    dI.at<double>(1,0) = dy;
  
    return;
  }
  
/*
  Computes the 2D Hessian matrix for a pixel in an image.

  @param scale_img one level of the DoG pyramid
  @param row pixel's image row
  @param col pixel's image column

  @return Returns the Hessian matrix (below) for pixel I as a CvMat&

  / Ixx  Ixy \ <BR>
  \ Ixy  Iyy /
*/
  void hessian_2D( const cv::Mat& scale_img, cv::Mat& H, int row, int col)
  {
    double val, dxx, dyy, dxy;
  
    val = scale_img.at<float>(row,col);
    dxx = ( scale_img.at<float>(row,col+1) +
            scale_img.at<float>(row,col-1) - 2*val );
    dyy = ( scale_img.at<float>(row+1,col) +
            scale_img.at<float>(row-1,col) - 2*val );
    dxy = ( scale_img.at<float>(row+1,col+1) -
            scale_img.at<float>(row+1,col-1) -
            scale_img.at<float>(row-1,col+1) + 
            scale_img.at<float>(row-1,col-1) ) / 4.0;
            
    H = cv::Mat(2,2,CV_64FC1);
    H.at<double>(0,0) = dxx;
    H.at<double>(0,1) = dxy;
    H.at<double>(1,0) = dxy;
    H.at<double>(1,1) = dyy;
    
    return;

  }

/*
  Performs one step of extremum interpolation.  Based on Eqn. (3) in Lowe's
  paper.

  @param scale_img one scale level of DoG
  @param row row being interpolated
  @param col column being interpolated
  @param xr output as interpolated subpixel increment to row
  @param xc output as interpolated subpixel increment to col
*/
  void interp_step( const cv::Mat& scale_img, int row, int col, double& xr, double& xc )
  {
    cv::Mat dD, H, X;

    deriv_2D( scale_img, dD, row, col);
    hessian_2D( scale_img, H, row, col);
    cv::gemm( H.inv(), dD, -1, cv::Mat(), 0, X);
    xr = X.at<double>(1);
    xc = X.at<double>(0);
  }

/*
  Calculates interpolated pixel contrast.  Based on Eqn. (3) in Lowe's
  paper.

  @param scale_img one scale level of DoG
  @param row pixel row
  @param col pixel column
  @param xr interpolated subpixel increment to row
  @param xc interpolated subpixel increment to col

  @param Returns interpolated contrast.
*/
  double interp_contr( const cv::Mat& scale_img, int row, int col, double& xr, double& xc )
  {
    cv::Mat dD, X, T;
    X.push_back(xc);
    X.push_back(xr);
  
    deriv_2D( scale_img, dD, row, col);
    T = dD.t() * X;
      return scale_img.at<float>(row,col) + T.at<double>(0) * 0.5;
  }

/*
  Allocates and initializes a new feature

  @return Returns a pointer to the new feature
*/
  struct feature* new_feature( void )
  {
    struct feature* feat;
    struct detection_data* ddata;
  
    feat = (feature*) malloc( sizeof( struct feature ) );
    memset( feat, 0, sizeof( struct feature ) );
    ddata = (detection_data*) malloc( sizeof( struct detection_data ) );
    memset( ddata, 0, sizeof( struct detection_data ) );
    feat->feature_data = ddata;

    return feat;
  }

/*
  Interpolates a scale-space extremum's location to subpixel
  precision to form an image feature.  Rejects features with low contrast.
  Based on Section 4 of Lowe's paper.

  @param scale_img one scale level of DoG
  @param octv feature's octave of scale space
  @param row feature's image row
  @param col feature's image column
  @param intvls total intervals per octave
  @param contr_thr threshold on feature contrast

  @return Returns the feature resulting from interpolation of the given
    parameters or NULL if the given location could not be interpolated or
    if contrast at the interpolated loation was too low.  If a feature is
    returned, its scale, orientation, and descriptor are yet to be determined.
*/
  struct feature* interp_extremum(const cv::Mat& scale_img, 
                                         int octv, int row, int col, 
                                         int intvls, double contr_thr)
  {
    struct feature* feat;
    struct detection_data* ddata;
    double xr=0, xc=0, contr;
    int i = 0;

    while( i < SIFT_MAX_INTERP_STEPS )
      {
        interp_step( scale_img, row, col, xr, xc );
        if( std::abs( xr ) < 0.5  &&  std::abs( xc ) < 0.5 )
          break;

        col += cvRound( xc );
        row += cvRound( xr );
  
        if( col < SIFT_IMG_BORDER  ||
            row < SIFT_IMG_BORDER  ||
            col >= scale_img.cols - SIFT_IMG_BORDER  ||
            row >= scale_img.rows - SIFT_IMG_BORDER )
          {
            return NULL;
          }
  
        i++;
      }
  
    /* ensure convergence of interpolation */
    if( i >= SIFT_MAX_INTERP_STEPS )
      return NULL;
  
    contr = interp_contr( scale_img, row, col, xr, xc );
    if( std::abs( contr ) < contr_thr / intvls )
      return NULL;
  
    feat = new_feature();
    ddata = feat->feature_data;
    feat->x = ( col + xc ) * pow( 2.0, octv );
    feat->y = ( row + xr ) * pow( 2.0, octv );
    ddata->r = row;
    ddata->c = col;
    ddata->octv = octv;

    return feat;
  }

/*
  Determines whether a feature is too edge like to be stable by computing the
  ratio of principal curvatures at that feature.  Based on Section 4.1 of
  Lowe's paper.

  @param dog_img image from the DoG pyramid in which feature was detected
  @param r feature row
  @param c feature col
  @param curv_thr high threshold on ratio of principal curvatures

  @return Returns 0 if the feature at (r,c) in dog_img is sufficiently
    corner-like or 1 otherwise.
*/
  bool is_too_edge_like(const cv::Mat& scale_img, int row, int col, int curv_thr)
  {

    /* principal curvatures are computed using the trace and det of Hessian */
    double val = scale_img.at<float>(row,col);
    double dxx = scale_img.at<float>(row,col+1) + scale_img.at<float>(row,col-1) - 2*val;
    double dyy = scale_img.at<float>(row+1,col) + scale_img.at<float>(row-1,col) - 2*val;
    double dxy = ( scale_img.at<float>(row+1,col+1) - scale_img.at<float>(row+1,col-1) -
                 scale_img.at<float>(row-1,col+1) + scale_img.at<float>(row-1,col-1) ) / 4.0;
    double trace = dxx + dyy;
    double det = dxx*dyy - dxy*dxy;
    /* negative determinant -> curvatures have different signs; reject feature */
    if( det <= 0 )
      return true;

    if( trace * trace / det < ( curv_thr + 1.0 )*( curv_thr + 1.0 ) / curv_thr )
      return false;
    return true;
  }

/*
  Detects features at extrema in DoG scale space.  Bad features are discarded
  based on contrast and ratio of principal curvatures.

  @param scale_img one scale level of DoG
  @param octvs octaves of scale space represented by dog_pyr
  @param intvls intervals per octave
  @param contr_thr low threshold on feature contrast
  @param curv_thr high threshold on feature ratio of principal curvatures
  @param storage memory storage in which to store detected features

  @return Returns an array of detected features whose scales, orientations,
    and descriptors are yet to be determined.
*/
  void scale_space_extrema(const cv::Mat& scale_img, int intvls, 
                                    double contr_thr, int curv_thr,
                                    std::vector<cv::KeyPoint>& keypoints)
  {
    double prelim_contr_thr = 0.5 * contr_thr / intvls;
    struct feature* feat;
    struct detection_data* ddata;
    int o, i, r, c;

    for(r = SIFT_IMG_BORDER; r < scale_img.rows-SIFT_IMG_BORDER; r++)
      for(c = SIFT_IMG_BORDER; c < scale_img.cols-SIFT_IMG_BORDER; c++)
        /* perform preliminary check on contrast */
        if( std::abs( scale_img.at<float>( r, c ) ) > prelim_contr_thr )
          if( is_extremum( scale_img, r, c ) )
          {
            feat = interp_extremum(scale_img, o, r, c, intvls, contr_thr);
            if( feat )
            {
              ddata = feat->feature_data;
              if( ! is_too_edge_like( scale_img, ddata->r, ddata->c, curv_thr ) )
              {
                cv::KeyPoint point(ddata->c, ddata->r, 1);
                keypoints.push_back(point);
              }
              else
                free( ddata );
              free( feat );
            }
         }

    return;
  }
  
  void harrisCorner(cv::Mat& src, cv::Mat& dst, double sigma, double k)
  {
    dst = cv::Mat(src.size(), CV_32FC1);
    cv::Mat Lxx, Lyy, Lxy, temp;
    int ksize = 3;
    int dx = 1, dy = 0;
    cv::Sobel(src, Lxx, -1, dx, dy, ksize);
    dx = 0, dy = 1;
    cv::Sobel(src, Lyy, -1, dx, dy, ksize);  
    
    temp = Lxx.mul(Lyy);
    temp *= sigma*sigma;
    cv::GaussianBlur(temp, Lxy, cv::Size(), sigma);
    
    temp = Lxx.mul(Lxx);
    temp *= sigma*sigma;
    cv::GaussianBlur(temp, Lxx, cv::Size(), sigma);
  
    temp = Lyy.mul(Lyy);
    temp *= sigma*sigma;
    cv::GaussianBlur(temp, Lyy, cv::Size(), sigma);

      
    for (int row = 0; row < dst.rows; ++row)
      for (int col = 0; col < dst.cols; ++col)
      {
        dst.at<float>(row, col) = Lxx.at<float>(row, col) * Lyy.at<float>(row, col)
                                - Lxy.at<float>(row, col) * Lxy.at<float>(row, col)
                           - k * (Lxx.at<float>(row, col) + Lyy.at<float>(row, col))
                               * (Lxx.at<float>(row, col) + Lyy.at<float>(row, col));
      } 
  }
  
  void hessianDeterminant(cv::Mat& src, cv::Mat& dst, double sigma)
  {
    dst = cv::Mat(src.size(), CV_32FC1);
    cv::Mat Lxx, Lyy, Lxy, temp;
    int ksize = 3;
    int dx = 2, dy = 0;
    cv::Sobel(src, Lxx, -1, dx, dy, ksize);
    dx = 0, dy = 2;
    cv::Sobel(src, Lyy, -1, dx, dy, ksize);
    dx = 1, dy = 1;
    cv::Sobel(src, Lxy, -1, dx, dy, ksize);  
    
    temp = Lxy * pow(sigma, 4);
    cv::GaussianBlur(temp, Lxy, cv::Size(), sigma);
    
    temp = Lxx * pow(sigma, 4);
    cv::GaussianBlur(temp, Lxx, cv::Size(), sigma);
  
    temp = Lyy * pow(sigma, 4);
    cv::GaussianBlur(temp, Lyy, cv::Size(), sigma);

      
    for (int row = 0; row < dst.rows; ++row)
      for (int col = 0; col < dst.cols; ++col)
      {
        dst.at<float>(row, col) = Lxx.at<float>(row, col) * Lyy.at<float>(row, col)
                                - Lxy.at<float>(row, col) * Lxy.at<float>(row, col);
      } 
  }
  
  void fitGaussians(std::vector< std::vector<cv::Point> >& msers, std::vector<cv::Mat>& mean,
                    std::vector<cv::Mat>& covar, std::vector<double>& area)
  {
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
  }
  
  double GaussianDerivative1D(double x, double sigma, int order)
  {
    double zeroth = exp(-x*x/(2*sigma*sigma)) / (sigma * sqrt(2*M_PI));
    switch (order)
    {
      case 0:
        return zeroth;
    
      case 1:
        return zeroth * -x/(sigma*sigma);
      
      case 2:
        return zeroth * (x*x - sigma*sigma) / pow(sigma, 4);
      
      case 3:
        return zeroth * -(pow(x, order) -3*x*sigma*sigma) / pow(sigma, 6);
      
      case 4:
        return zeroth * (pow(x, order) - 6*x*x*sigma*sigma + 3*pow(sigma, 4)) / pow(sigma, 8);
      
      default:
        return -1;
    }
  }
  
  void GaussDerivKernel1D(double sigma, int order, cv::Mat& kernel)
  {
    int size = 6*cvRound(sigma) + 1;
    kernel = cv::Mat(1,size,CV_32FC1);
    for (int frag = 0; frag < size; ++frag)
      kernel.at<float>(0,frag) = GaussianDerivative1D(frag -(size+1)/2, sigma, order);
    kernel /= cv::sum(kernel)[0];
  }
          
  void removeDuplicates(std::vector<cv::KeyPoint>& keypoints, double tol)
  {
    for (auto i = keypoints.begin(); i != keypoints.end(); ++i)
      for (auto j = keypoints.begin(); j != keypoints.end(); ++j)
      {
        if (i == j) continue;
  
        if (sqrt(pow((*i).pt.x - (*j).pt.x, 2) + 
                 pow((*i).pt.y - (*j).pt.y, 2)) < tol)
        {
          keypoints.erase(j);
          --i;
          break;
        }
    
      }
  }          
} /* End Namespace pk */
