#include "Laplace.hpp"

namespace pk
{

  Laplace::Laplace(int _curve, float thresh, int oct, int inter, double _sigma)
  {
    curveThresh = _curve;
    threshold = thresh;
    octaves = oct;
    intervals = inter;
    sigma.resize(intervals + 2);
    border = 5;
    recursion = 0;
    /*
      precompute Gaussian sigmas using the following formula:
      \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    */
    sigma[0] = _sigma;
    double p = pow( 2.0, 1.0 / intervals );
    for (int inter = 1; inter < intervals + 2; ++inter )
    {
        double sig_prev = pow( p, inter - 1 ) * _sigma;
        double sig_total = sig_prev * p;
        sigma[inter] = sqrt( sig_total * sig_total - sig_prev * sig_prev );
    }
  }
  
  Laplace::Laplace(const cv::Mat& img, int _curve, float thresh, int oct, int inter, double _sigma)
  {
    curveThresh = _curve;
    threshold = thresh;
    octaves = oct;
    intervals = inter;
    sigma.resize(intervals + 2);
    border = 5;
    recursion = 0;
    /*
      precompute Gaussian sigmas using the following formula:
      \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    */
    sigma[0] = _sigma;
    double p = pow( 2.0, 1.0 / intervals );
    for (int inter = 1; inter < intervals + 2; ++inter )
    {
        double sig_prev = pow( p, inter - 1 ) * _sigma;
        double sig_total = sig_prev * p;
        sigma[inter] = sqrt( sig_total * sig_total - sig_prev * sig_prev );
    }    
    buildGaussPyr(img);
    buildLaplacePyr();
  }

  void Laplace::initialise(const cv::Mat& img)
  {
    buildGaussPyr(img);
    buildLaplacePyr();
  }

  void Laplace::detector(std::vector<cv::KeyPoint>& keypoints)
  {
    double prelimThresh = 0.5 * this->threshold / intervals;
    for (int oct = 0; oct < octaves; ++oct )
      for (int inter = 1; inter <= intervals; ++inter )
        for (int row = border; row < laplacePyramid[oct][0].rows - border; ++row)
          for (int col = border; col < laplacePyramid[oct][0].cols - border; ++col)
            /* perform preliminary check on contrast */
            if( std::abs( laplacePyramid[oct][inter].at<float>(row, col) )  > prelimThresh )
              if( isLocalExtremum(row, col, oct, inter) )// & !isTooEdgeLike(oct, inter, row, col) )
              {
                cv::KeyPoint point;
                cv::Point3f coords;
                int code = interpExtremum( oct, inter, row, col, point, coords);
                if (0 == code)
                  if (!isTooEdgeLike(oct, coords.z, coords.y, coords.x))       
                    keypoints.push_back(point); 
              }        
    return;

  }

  void Laplace::detector(cv::Mat& img, 
                                std::vector<cv::KeyPoint>& keypoints)
  {
    buildGaussPyr(img);
    buildLaplacePyr();
    double prelimThresh = 0.5 * this->threshold / intervals;
    for (int oct = 0; oct < octaves; ++oct )
      for (int inter = 1; inter <= intervals; ++inter )
        for (int row = border; row < laplacePyramid[oct][0].rows - border; ++row)
          for (int col = border; col < laplacePyramid[oct][0].cols - border; ++col)
            /* perform preliminary check on contrast */
            if( std::abs( laplacePyramid[oct][inter].at<float>(row, col) )  > prelimThresh )
              if( isLocalExtremum(row, col, oct, inter) & !isTooEdgeLike(oct, inter, row, col) )
              {
                cv::KeyPoint point;
                cv::Point3f coords;
                int code = interpExtremum( oct, inter, row, col, point, coords);
                if (0 == code)       
                  if (!isTooEdgeLike(oct, coords.z, coords.y, coords.x))       
                    keypoints.push_back(point); 
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
  void Laplace::createBaseImage(const cv::Mat& src, cv::Mat& dst,
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
  void Laplace::buildGaussPyr(const cv::Mat& img)
  {
  
    gaussPyramid.resize(octaves);
    for (int oct = 0; oct < octaves; ++oct )
      gaussPyramid[oct].resize(intervals + 2);

    for (int oct = 0; oct < octaves; ++oct )
      for (int inter = 0; inter < intervals + 2; ++inter )
      {
  	    if ( oct == 0  &&  inter == 0 )
  	      createBaseImage(img, gaussPyramid[oct][inter], true, sigma[0]);

        /* base of new octave is halved image from end of previous octave */
        else if ( inter == 0 )
          cv::resize(gaussPyramid[oct-1][intervals], gaussPyramid[oct][inter], cv::Size(), 0.5, 0.5, CV_INTER_AREA);

        /* blur the current octave's last image to create the next one */
        else
         	cv::GaussianBlur(gaussPyramid[oct][inter-1], gaussPyramid[oct][inter], cv::Size(), sigma[inter]);
//        cv::namedWindow("garry", CV_WINDOW_AUTOSIZE );
//        cv::imshow("garry", gaussPyramid[oct][inter]);
//        cv::waitKey(0);         	
        }

    return;
  }
  
  void Laplace::buildLaplacePyr()
  {
    laplacePyramid.resize(octaves);
    for (int oct = 0; oct < octaves; ++oct )
      laplacePyramid[oct].resize(intervals + 2);
    
    for (int oct = 0; oct < octaves; ++oct )
      for (int inter = 0; inter < intervals + 2; ++inter )
      {  
        double size = sigma[0] * pow(2.0, oct + (double)inter / intervals) * 0.5;
        Laplacian(gaussPyramid[oct][inter], laplacePyramid[oct][inter], gaussPyramid[oct][inter].depth(), 3, size * size);
/*        
        cv::Mat img;
        cv::normalize(laplacePyramid[oct][inter], img, 0, 1, 32);
        cv::namedWindow("larry", CV_WINDOW_AUTOSIZE );
        cv::imshow("larry", img);
        cv::waitKey(0);
//*/
      }
  }

	bool Laplace::isLocalExtremum(int row, int col, int octave, int interval)
	{
	  double val = laplacePyramid[octave][interval].at<float>(row,col);
	  
	  /* check for maximum */  
	  if( val > 0 )
    {
      for (int scale = -1; scale <= 1; ++scale)
        for (int y = -1; y <= 1; ++y )
          for (int x = -1; x <= 1; ++x )
            if (val < laplacePyramid[octave][interval + scale].at<float>(row + y, col + x))
              return false;
    }

    /* check for minimum */ 
    else
    {
      for (int scale = -1; scale <= 1; ++scale)
        for (int y = -1; y <= 1; ++y )
          for (int x = -1; x <= 1; ++x )
	  		  	if (val > laplacePyramid[octave][interval + scale].at<float>(row + y, col + x))
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
    void Laplace::deriv_2D(const cv::Mat& scale_img, cv::Mat& dI, int row, int col)
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
    void Laplace::hessian_2D( cv::Mat& H,int oct, int inter, int row, int col)
    {
      double val, dxx, dyy, dxy;
    
      val = laplacePyramid[oct][inter].at<float>(row,col);
      dxx = ( laplacePyramid[oct][inter].at<float>(row,col+1) +
              laplacePyramid[oct][inter].at<float>(row,col-1) - 2*val );
      dyy = ( laplacePyramid[oct][inter].at<float>(row+1,col) +
              laplacePyramid[oct][inter].at<float>(row-1,col) - 2*val );
      dxy = ( laplacePyramid[oct][inter].at<float>(row+1,col+1) -
              laplacePyramid[oct][inter].at<float>(row+1,col-1) -
              laplacePyramid[oct][inter].at<float>(row-1,col+1) + 
              laplacePyramid[oct][inter].at<float>(row-1,col-1) ) / 4.0;
              
      H = cv::Mat(2,2,CV_64FC1);
      H.at<double>(0,0) = dxx;
      H.at<double>(0,1) = dxy;
      H.at<double>(1,0) = dxy;
      H.at<double>(1,1) = dyy;
      
      return;
  
    }
    
    

  /*
    Computes the partial derivatives in x, y, and scale of a pixel in the DoG
    scale space pyramid.
  
    @param dog_pyr DoG scale space pyramid
    @param octv pixel's octave in dog_pyr
    @param intvl pixel's interval in octv
    @param r pixel's image row
    @param c pixel's image col

    @return Returns the vector of partial derivatives for pixel I
      { dI/dx, dI/dy, dI/ds }^T as a CvMat*
  */
  void Laplace::deriv_3D( cv::Mat& dI, int oct, int inter, int row, int col)
  {
    double dx, dy, ds;
  
    dx = ( laplacePyramid[oct][inter].at<float>(row,col+1) -
           laplacePyramid[oct][inter].at<float>(row,col-1) ) / 2.0;
  	dy = ( laplacePyramid[oct][inter].at<float>(row+1,col) -
  	       laplacePyramid[oct][inter].at<float>(row-1,col) ) / 2.0;
    ds = ( laplacePyramid[oct][inter+1].at<float>(row,col) -
           laplacePyramid[oct][inter-1].at<float>(row,col) ) / 2.0;
  
    dI = cv::Mat( 3, 1, CV_64FC1 );
    dI.at<double>(0,0) = dx;
    dI.at<double>(1,0) = dy;
    dI.at<double>(2,0) = ds;
  
    return;
  }

  /*
    Computes the 3D Hessian matrix for a pixel in the DoG scale space pyramid.
  
    @param dog_pyr DoG scale space pyramid
    @param octv pixel's octave in dog_pyr
    @param intvl pixel's interval in octv
    @param r pixel's image row
    @param c pixel's image col

    @return Returns the Hessian matrix (below) for pixel I as a CvMat*
  
    / Ixx  Ixy  Ixs \ <BR>
    | Ixy  Iyy  Iys | <BR>
    \ Ixs  Iys  Iss /
  */

  void Laplace::hessian_3D( cv::Mat& H, int oct, int inter, int row, int col)
  {
    double val, dxx, dyy, dss, dxy, dxs, dys;


    val = laplacePyramid[oct][inter].at<float>(row,col);
    dxx = ( laplacePyramid[oct][inter].at<float>(row,col+1) +
            laplacePyramid[oct][inter].at<float>(row,col-1) - 2*val );
    dyy = ( laplacePyramid[oct][inter].at<float>(row+1,col) +
            laplacePyramid[oct][inter].at<float>(row-1,col) - 2*val );
    dss = ( laplacePyramid[oct][inter+1].at<float>(row,col) +
            laplacePyramid[oct][inter-1].at<float>(row,col) - 2*val );
    dxy = ( laplacePyramid[oct][inter].at<float>(row+1,col+1) -
            laplacePyramid[oct][inter].at<float>(row+1,col-1) -
            laplacePyramid[oct][inter].at<float>(row-1,col+1) + 
            laplacePyramid[oct][inter].at<float>(row-1,col-1) ) / 4.0;
    dxs = ( laplacePyramid[oct][inter+1].at<float>(row,col+1) -
            laplacePyramid[oct][inter+1].at<float>(row,col-1) -
            laplacePyramid[oct][inter-1].at<float>(row,col+1) + 
            laplacePyramid[oct][inter-1].at<float>(row,col-1) ) / 4.0;
    dys = ( laplacePyramid[oct][inter+1].at<float>(row+1,col) -
            laplacePyramid[oct][inter+1].at<float>(row-1,col) -
            laplacePyramid[oct][inter-1].at<float>(row+1,col) + 
            laplacePyramid[oct][inter-1].at<float>(row-1,col) ) / 4.0;
          
    H = cv::Mat(3,3,CV_64FC1);
    H.at<double>(0,0) = dxx;
    H.at<double>(0,1) = dxy;
    H.at<double>(0,2) = dxs;
    H.at<double>(1,0) = dxy;
    H.at<double>(1,1) = dyy;
    H.at<double>(1,2) = dys;
    H.at<double>(2,0) = dxs;
    H.at<double>(2,1) = dys;
    H.at<double>(2,2) = dss;
    
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
    void Laplace::interp_step( int oct, int inter, int row, int col, double& xi, double& xr, double& xc )
    {
      cv::Mat dD, H, X;
  
      deriv_3D( dD, oct, inter, row, col);
      hessian_3D( H, oct, inter, row, col);
      cv::gemm( H.inv(), dD, -1, cv::Mat(), 0, X);
      xi = X.at<double>(2);
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
    double Laplace::interpContrast( int oct, int inter, int row, int col, double& xi,  double& xr, double& xc )
    {
      cv::Mat dD, X, T;
      X.push_back(xc);
      X.push_back(xr);
      X.push_back(xi);

      deriv_3D( dD, oct, inter, row, col);
    
      T = dD.t() * X;
  
      return laplacePyramid[oct][inter].at<float>(row,col) + T.at<double>(0) * 0.5;
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
  int Laplace::interpExtremum( int oct, int inter, int row, int col, cv::KeyPoint& point, cv::Point3f& coords)
  {
    double xi=0, xr=0, xc=0, contr;
    int i = 0;
    const int MAX_STEPS = 5;
    while( i < MAX_STEPS )
    {
      interp_step( oct, inter, row, col, xi, xr, xc );
      if( std::abs( xi ) < 0.5  &&  std::abs( xr ) < 0.5  &&  std::abs( xc ) < 0.5 )
        break;

      col += cvRound( xc );
      row += cvRound( xr );
      inter += cvRound( xi );

      if( inter < 1  ||
          inter > intervals  ||
          col < border  ||
          row < border  ||
          col >= laplacePyramid[oct][0].cols - border  ||
          row >= laplacePyramid[oct][0].rows - border )
        {
          return 1;
        }

      i++;
    }

  
    /* ensure convergence of interpolation */
    if( i >= MAX_STEPS )
      return 2;
  
    contr = interpContrast( oct, inter, row, col, xi, xr, xc );
    if( std::abs( contr ) < threshold / intervals )
      return 3;
    point.pt.x = ( col + xc ) * pow( 2.0, oct ) * 0.5;
    point.pt.y = ( row + xr ) * pow( 2.0, oct ) * 0.5;
    double size = sigma[0] * pow(2.0, oct + static_cast<double>(inter + xi) / intervals) * 0.5;
    point.size = size * 4;
    
    coords.x = col;
    coords.y = row;
    coords.z = inter;

    return 0;
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

  bool Laplace::isTooEdgeLike(int oct, int inter, int row, int col)
  { 
    /* principal curvatures are computed using the trace and det of Hessian */
    
/*    double val = laplacePyramid[oct][inter].at<float>(row,col);
    double dxx = laplacePyramid[oct][inter].at<float>(row,col+1) + laplacePyramid[oct][inter].at<float>(row,col-1) - 2*val;
    double dyy = laplacePyramid[oct][inter].at<float>(row+1,col) + laplacePyramid[oct][inter].at<float>(row-1,col) - 2*val;
    double dxy = ( laplacePyramid[oct][inter].at<float>(row+1,col+1) - laplacePyramid[oct][inter].at<float>(row+1,col-1) -
                 laplacePyramid[oct][inter].at<float>(row-1,col+1) + laplacePyramid[oct][inter].at<float>(row-1,col-1) ) / 4.0;
    double trace = dxx + dyy;
    double det = dxx*dyy - dxy*dxy;
//*/
    cv::Mat Hess;
    hessian_2D( Hess, oct, inter, row, col);
    double trace = cv::trace(Hess)[0];
    double det = cv::determinant(Hess);

    /* negative determinant -> curvatures have different signs; reject feature */
    if( det <= 0 )
      return true;

    if( trace * trace / det < ( curveThresh + 1.0 )*( curveThresh + 1.0 ) / curveThresh )
      return false;
    return true;
  }

}				/* End Namespace pk */
