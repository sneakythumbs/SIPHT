#include "Harris_Laplace.hpp"

namespace pk
{

  Harris_Laplace::Harris_Laplace(float _k, float thresh, int oct, int inter, double _sigma)
  {
    k = _k;
    threshold = thresh;
    octaves = oct;
    intervals = inter;
    sigma.resize(intervals + 2);
    border = 5;
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
  
  Harris_Laplace::Harris_Laplace(const cv::Mat& img, float _k, float thresh, int oct, int inter, double _sigma)
  {
    k = _k;
    threshold = thresh;
    octaves = oct;
    intervals = inter;
    sigma.resize(intervals + 2);
    border = 5;
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
                //TODO interpolate keypoint
                double size = sigma[0] * pow(2.0, oct + (double)inter / intervals) * 2;
                double dcol = 0, drow = 0, scale = pow(2.0, oct);
                cv::KeyPoint point((col + dcol) * scale * 0.5, (row + drow) * scale * 0.5, size);
                keypoints.push_back(point);    
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
            /* perform preliminary check on contrast */
            if(  harrisPyramid[oct][inter].at<float>(row, col)  > this->threshold )
            {
            	std::cout << true << std::endl;
              if( isSpacialExtremum(row, col, oct, inter) & isScaleExtremum(row, col, oct, inter) )
              {
                //TODO interpolate keypoint
                double size = sigma[0] * pow(2.0, oct + (double)inter / intervals);
                double dcol = 0, drow = 0, scale = pow(2.0, oct);
                cv::KeyPoint point((col + dcol) * scale, (row + drow) * scale, size);
                keypoints.push_back(point);    
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
      gaussPyramid[oct].resize(intervals + 2);

    for (int oct = 0; oct < octaves; ++oct )
      for (int inter = 0; inter < intervals + 2; ++inter )
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
//        cv::Mat temp;
//        cv::normalize( gaussPyramid[oct][inter], temp, 0, 255, 32, CV_32FC1, cv::Mat() );
//        cv::convertScaleAbs( temp, temp );

//        cv::cornerHarris(temp, harrisPyramid[oct][inter], 3, 1, k);
        harrisCorner(gaussPyramid[oct][inter], harrisPyramid[oct][inter], sigma[inter], k);
//        harrisPyramid[oct][inter] *= -1;
//        cv::normalize( harrisPyramid[oct][inter], harrisPyramid[oct][inter], 0.0, 255.0, 32, CV_32FC1, cv::Mat() );
//        double max, min;
//        cv::minMaxLoc(harrisPyramid[oct][inter], &min, &max);
//        cv::convertScaleAbs( harrisPyramid[oct][inter], harrisPyramid[oct][inter] );
//        std::cout << harrisPyramid[oct][inter] << std::endl;   
//        std::cout << max << std::endl;     
        cv::namedWindow("harry", CV_WINDOW_AUTOSIZE );
        cv::imshow("harry", harrisPyramid[oct][inter]);
        cv::waitKey(0);
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
	  double val = getPixelLaplacian(harrisPyramid[octave][interval], row, col) * size * size;
	  
	  if (val > 0)
	    for (int scale = -1; scale <= 1; ++ scale)
	    {
	      double size = sigma[0] * pow(2.0, octave + (double)(interval + scale) / intervals);
	      double neighbour = getPixelLaplacian(harrisPyramid[octave][interval + scale], row, col) * size * size;
	      if (val < neighbour)
	        return false;
	    }
	  else
	    for (int scale = -1; scale <= 1; ++ scale)
	    {
	      double size = sigma[0] * pow(2.0, octave + (double)(interval + scale) / intervals);
	      double neighbour = getPixelLaplacian(harrisPyramid[octave][interval + scale], row, col) * size * size;
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

}				/* End Namespace pk */
