#include "Harris_Laplace.hpp"
#include "Affine_Adaptation.hpp"

namespace pk
{

  Harris_Laplace::Harris_Laplace(int _method, float _k, float thresh, int oct, int inter, double _sigma)
  {
    k = _k;
    threshold = thresh;
    octaves = oct;
    intervals = inter;
    sigma.resize(intervals + 3);
    border = 5;
    recursion = 0;
    method = _method;
    /*
      precompute Gaussian sigmas using the following formula:
      \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    */
    sigma[0] = _sigma;
    double p = pow( 2.0, 1.0 / intervals );
    for (int inter = 1; inter < intervals + 3; ++inter )
    {
        double sig_prev = pow( p, inter - 1 ) * _sigma;
        double sig_total = sig_prev * p;
        sigma[inter] = sqrt( sig_total * sig_total - sig_prev * sig_prev );
    }
  }
  
  Harris_Laplace::Harris_Laplace(const cv::Mat& img, int _method, float _k, float thresh, int oct, int inter, double _sigma)
  {
    k = _k;
    threshold = thresh;
    octaves = oct;
    intervals = inter;
    sigma.resize(intervals + 3);
    border = 5;
    recursion = 0;
    method = _method;
    /*
      precompute Gaussian sigmas using the following formula:
      \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    */
    sigma[0] = _sigma;
    double p = pow( 2.0, 1.0 / intervals );
    for (int inter = 1; inter < intervals + 3; ++inter )
    {
        double sig_prev = pow( p, inter - 1 ) * _sigma;
        double sig_total = sig_prev * p;
        sigma[inter] = sqrt( sig_total * sig_total - sig_prev * sig_prev );
    }
        
    buildGaussPyr(img);
    buildHarrisPyr();
  }

  void Harris_Laplace::initialise(const cv::Mat& img)
  {
    buildGaussPyr(img);
    buildHarrisPyr();
  }

  void Harris_Laplace::detector(std::vector<cv::KeyPoint>& keypoints)
  {
    for (int oct = 0; oct < octaves; ++oct )
      for (int inter = 1; inter <= intervals; ++inter )
        for (int row = border; row < harrisPyramid[oct][0].rows - border; ++row)
          for (int col = border; col < harrisPyramid[oct][0].cols - border; ++col)
          {
            /* perform preliminary check on contrast */
            if(  harrisPyramid[oct][inter].at<float>(row, col)  > this->threshold )
            {
              if( isSpacialExtremum(row, col, oct, inter) & isScaleExtremum(row, col, oct, inter) )
              {
                cv::Point2f coords;
                int unfound = interpSpacialExtremum(harrisPyramid[oct][inter], oct, row, col, coords);
                //TODO interpolate scale
                if (!unfound)
                {             
                  double size = sigma[0] * pow(2.0, oct + (double)inter / intervals);
//                  cv::KeyPoint point(coords * 0.5, size);
                  cv::KeyPoint point(coords, size);
//                  int method = 1;
                  if (method)
                  {
                    Affine_Adaptation adaptor(method, ".", ".");
                    Elliptic_KeyPoint EPoint;
                    adaptor.keyPointToElliptic(point, EPoint);
                    bool isDivergent = adaptor.calcAffineAdaptation(gaussPyramid[0][0], EPoint, 3);
                    if (!isDivergent)
                    {
                      adaptor.ellipticToKeyPoint(EPoint, point);
                      point.pt *= 0.5;
                      keypoints.push_back(point);
                    }
                  }


//                  int code = findScale(row, col, oct, inter, point);
//                  if (0 == code)
                  if (!method)
                    keypoints.push_back(point);
                }
              }  
            }        
          }
    return;
  }

  void Harris_Laplace::detector(cv::Mat& img, 
                                std::vector<cv::KeyPoint>& keypoints)
  {
    buildGaussPyr(img);
    buildHarrisPyr();
    for (int oct = 0; oct < octaves; ++oct )
      for (int inter = 1; inter <= intervals; ++inter )
        for (int row = border; row < harrisPyramid[oct][0].rows - border; ++row)
          for (int col = border; col < harrisPyramid[oct][0].cols - border; ++col)
          {
            /* perform preliminary check on contrast */
            if(  harrisPyramid[oct][inter].at<float>(row, col)  > this->threshold )
            {
              if( isSpacialExtremum(row, col, oct, inter) & isScaleExtremum(row, col, oct, inter) )
              {
                cv::Point2f coords;
                int unfound = interpSpacialExtremum(harrisPyramid[oct][inter], oct, row, col, coords);
                //TODO interpolate scale
                if (!unfound)
                {             
                  double size = sigma[0] * pow(2.0, oct + (double)inter / intervals);
//                  cv::KeyPoint point(coords * 0.5, size);
                  cv::KeyPoint point(coords, size);
//                  int method = 2;
                  if (method)
                  {
                    Affine_Adaptation adaptor(method, ".", ".");
                    Elliptic_KeyPoint EPoint;
                    adaptor.keyPointToElliptic(point, EPoint);
                    bool isDivergent = adaptor.calcAffineAdaptation(gaussPyramid[0][0], EPoint, 3);
                    if (!isDivergent)
                    {
                      adaptor.ellipticToKeyPoint(EPoint, point);
                      point.pt *= 0.5;
                      keypoints.push_back(point);
                    }
                  }


//                  int code = findScale(row, col, oct, inter, point);
//                  if (0 == code)
                  if (!method)
                    keypoints.push_back(point);
                }
              }  
            }        
          }
    return;
  }
  
  /*
     Converts an image to 8-bit grayscale and Gaussian-smooths it.  The image is
     optionally doubled in size prior to smoothing.

     @param img input image
     @param img_dbl if true, image is doubled in size prior to smoothing
     @param sig total std of Gaussian smoothing
   */
  void Harris_Laplace::createBaseImage(const cv::Mat& src, cv::Mat& dst,
					                             const bool img_dbl, const double sig)
  {
    cv::Mat grey, dbl;
    double sig_diff;
    cv::Mat fimg;
    src.convertTo( fimg, CV_64FC1 );
    if (src.channels() == 3)
      cv::cvtColor(fimg * 1.0 / 255, grey, CV_BGR2GRAY);
    else if (src.channels() == 1)
      grey = fimg * 1.0 / 255;
    else
    {
	    std::cout << "not an rgb or greyscale image\n";
	    exit (EXIT_FAILURE);
    }

    if (img_dbl)
    {
	    sig_diff = sqrt(sig * sig - 0.5 * 0.5 * 4);

    	cv::resize(grey, dbl, cv::Size(), 2, 2, CV_INTER_CUBIC);
	    cv::GaussianBlur(dbl, dbl, cv::Size(), sig_diff);
	    dbl.convertTo(dst, CV_32FC1);
	    return;
    }
    else
    {
	    sig_diff = sqrt(sig * sig - 0.5 * 0.5);
	    cv::GaussianBlur(grey, grey, cv::Size(), sig_diff);
	    grey.convertTo(dst, CV_32FC1);
	    return;
    }
  }

/*
  Builds Gaussian scale space pyramid from an image

  @param img base image of the pyramid

*/
  void Harris_Laplace::buildGaussPyr(const cv::Mat& img)
  {
  
    gaussPyramid.resize(octaves);
    for (int oct = 0; oct < octaves; ++oct )
      gaussPyramid[oct].resize(intervals + 3);

    for (int oct = 0; oct < octaves; ++oct )
      for (int inter = 0; inter < intervals + 3; ++inter )
      {
  	    if ( oct == 0  &&  inter == 0 )
  	      createBaseImage(img, gaussPyramid[oct][inter], true, sigma[0]);

        /* base of new octave is halved image from end of previous octave */
        else if ( inter == 0 )
          cv::resize(gaussPyramid[oct-1][intervals], gaussPyramid[oct][inter], cv::Size(), 0.5, 0.5, CV_INTER_NN);

        /* blur the current octave's last image to create the next one */
        else
         	cv::GaussianBlur(gaussPyramid[oct][inter-1], gaussPyramid[oct][inter], cv::Size(), sigma[inter]);
//        cv::namedWindow("garry", CV_WINDOW_AUTOSIZE );
//        cv::imshow("garry", gaussPyramid[oct][inter]);
//        cv::waitKey(0);         	
        }

    return;
  }
  
  void Harris_Laplace::buildHarrisPyr()
  {
    harrisPyramid.resize(octaves);
    for (int oct = 0; oct < octaves; ++oct )
      harrisPyramid[oct].resize(intervals + 2);
    
    for (int oct = 0; oct < octaves; ++oct )
      for (int inter = 0; inter < intervals + 2; ++inter )
      {  
        double size = sigma[0] * pow(2.0, oct + (double)inter / intervals);
//        cv::cornerHarris(gaussPyramid[oct][inter], harrisPyramid[oct][inter], cvRound(sigma[inter]), 1, k);
        harrisCorner(gaussPyramid[oct][inter], harrisPyramid[oct][inter], sigma[inter], k);
/*        
        cv::Mat img;
        cv::normalize(harrisPyramid[oct][inter], img, 0, 1, 32);
        cv::namedWindow("harry", CV_WINDOW_AUTOSIZE );
        cv::imshow("harry", img);
        cv::waitKey(0);
//*/
      }
  }

	bool Harris_Laplace::isSpacialExtremum(int row, int col, int octave, int interval)
	{
	  double val = harrisPyramid[octave][interval].at<float>(row,col);
	  
	  /* check for maximum */  
	  if( val > 0 )
    {
      for (int y = -1; y <= 1; ++y )
        for (int x = -1; x <= 1; ++x )
          if (val < harrisPyramid[octave][interval].at<float>(row + y, col + x))
            return false;
    }

    /* check for minimum */ // Response in monotonic, this is never called
    else
    {
      for (int y = -1; y <= 1; ++y )
        for (int x = -1; x <= 1; ++x )
			  	if (val > harrisPyramid[octave][interval].at<float>(row + y, col + x))
            return false;
    }

    return true;
	}
	  
	bool Harris_Laplace::isScaleExtremum(int row, int col, int octave, int interval)
	{
	  double size = sigma[0] * pow(2.0, octave + (double)interval / intervals);
	  //double size = sigma[interval];// * pow(2.0, octave);
	  double val = getPixelLaplacian(gaussPyramid[octave][interval + 1], row, col) * size * size;
    if (std::abs(val) < 1e-1) return false;
	  if (val > 0)
	    for (int scale = -1; scale <= 1; ++ scale)
	    {
	      double size = sigma[0] * pow(2.0, octave + (double)(interval + scale) / intervals);
//	      double size = sigma[interval];
	      double neighbour = getPixelLaplacian(gaussPyramid[octave][interval + scale + 1], row, col) * size * size;
	      if (val < neighbour)
	        return false;
	    }
	  else
	    for (int scale = -1; scale <= 1; ++ scale)
	    {
	      double size = sigma[0] * pow(2.0, octave + (double)(interval + scale) / intervals);
//	      double size = sigma[interval];
	      double neighbour = getPixelLaplacian(gaussPyramid[octave][interval + scale + 1], row, col) * size * size;
	      if (val > neighbour)
	        return false;
	    }
	  return true;
	}
	
	double Harris_Laplace::getPixelLaplacian(const cv::Mat& img, int row, int col)
	{
	  return ( img.at<float>(row - 1, col) +
	           img.at<float>(row, col - 1) +
	           img.at<float>(row + 1, col) +
	           img.at<float>(row, col + 1) -
	           img.at<float>(row, col) * 4 );
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
    void Harris_Laplace::deriv_2D(const cv::Mat& scale_img, cv::Mat& dI, int row, int col)
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
    void Harris_Laplace::hessian_2D( const cv::Mat& scale_img, cv::Mat& H, int row, int col)
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
    void Harris_Laplace::interp_step( const cv::Mat& scale_img, int row, int col, double& xr, double& xc )
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
  
    @param scale_img one scale level of Harris Pyramid
    @param row pixel row
    @param col pixel column
    @param xr interpolated subpixel increment to row
    @param xc interpolated subpixel increment to col
  
    @param Returns interpolated contrast.
  */
    double Harris_Laplace::interpContrast( const cv::Mat& scale_img, int row, int col, double& xr, double& xc )
    {
      cv::Mat dD, X, T;
      X.push_back(xc);
      X.push_back(xr);
    
      deriv_2D( scale_img, dD, row, col);
      T = dD.t() * X;
        return scale_img.at<float>(row,col) + T.at<double>(0) * 0.5;
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
  int Harris_Laplace::interpSpacialExtremum(const cv::Mat& scale_img, 
                                            int oct, int row, int col, 
                                            cv::Point2f& coords)
  {
    double xr=0, xc=0, contr;
    int i = 0;
    const int MAX_STEPS = 5;
    while( i < MAX_STEPS )
      {
        interp_step( scale_img, row, col, xr, xc );
        if( std::abs( xr ) < 0.5  &&  std::abs( xc ) < 0.5 )
          break;

        col += cvRound( xc );
        row += cvRound( xr );
  
        if( col < border  ||
            row < border  ||
            col >= scale_img.cols - border  ||
            row >= scale_img.rows - border )
          {
            return 1;
          }
  
        i++;
      }
  
    /* ensure convergence of interpolation */
    if( i >= MAX_STEPS )
      return 2;
  
    contr = interpContrast( scale_img, row, col, xr, xc );
    if( std::abs( contr ) < this->threshold / intervals )
      return 3;
    coords.x = ( col + xc ) * pow( 2.0, oct );
    coords.y = ( row + xr ) * pow( 2.0, oct );

    return 0;
  }  
  
  int Harris_Laplace::findScale(int row, int col, int oct, int inter, cv::KeyPoint& result)
  {
    if (!isSpacialExtremum(row, col, oct, inter)) return 1;
    int interval = findMaxInterval(row, col, oct);
    if (-1 == interval) return 2;
    if (interval == inter)
    {
      cv::Point2f coords;
      double size = sigma[0] * pow(2.0, oct + static_cast<double>(inter) / intervals) * 2;
      interpSpacialExtremum(harrisPyramid[oct][inter], oct, row, col, coords);
      result = cv::KeyPoint(coords * 0.5, size, -1, -1, oct, inter);
      return 0;
    }
    int code = searchNeighbours(row, col, oct, inter, result);
      if (0 == code) return 0;
    return 3;
  }
 
  int Harris_Laplace::findMaxInterval(int row, int col, int oct)
  {
  	double lap0 = 0;
	  for (int inter = 1; inter <= intervals; ++inter)
	  {
	    double size = sigma[0] * pow(2.0, oct + static_cast<double>(inter) / intervals) * 2;
	    double lap = getPixelLaplacian(gaussPyramid[oct][inter], row, col) * size * size;
	    if (std::abs(lap) < std::abs(lap0))
	      return inter - 1;
	    lap0 = lap;
	  }
	  return -1;
 
  }
    
  int Harris_Laplace::searchNeighbours(int row, int col, int oct, int inter, cv::KeyPoint& result)
  {
 
    for (int y = -1; y <= 1; ++y )
      for (int x = -1; x <= 1; ++x )
      {
//        if (0 == y & 0 == x) continue;
        if ((col + x) < border  ||
            (row + y) < border  ||
            (col + x) >= harrisPyramid[oct][inter].cols - border  ||
            (row + y) >= harrisPyramid[oct][inter].rows - border )
         {
           continue;
         }
        ++recursion;
        int code = findScale(row + y, col + x, oct, inter, result);
        --recursion;
        if (recursion > 10) return 1;

        std::cout << row << " " << col << " " << oct << " " << inter << std::endl;
//        std::cout << code << std::endl;
        if (0 == code)  return 0;
      }
    return 2;
  
  } 
  
}				/* End Namespace pk */
