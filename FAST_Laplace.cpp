#include "FAST_Laplace.hpp"

namespace pk
{

  FAST_Laplace::FAST_Laplace(float thresh, int oct, int inter, double _sigma)
  {
    threshold = thresh;
    octaves = oct;
    intervals = inter;
    sigma.resize(intervals + 3);
    border = 5;
    recursion = 0;
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
  
  FAST_Laplace::FAST_Laplace(const cv::Mat& img, float thresh, int oct, int inter, double _sigma)
  {
    threshold = thresh;
    octaves = oct;
    intervals = inter;
    sigma.resize(intervals + 3);
    border = 5;
    recursion= 0;
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
  }

  void FAST_Laplace::initialise(const cv::Mat& img)
  {
    buildGaussPyr(img);
  }

  void FAST_Laplace::detector(std::vector<cv::KeyPoint>& keypoints)
  {
    std::vector< std::vector< std::vector<cv::KeyPoint> > > fastPoints;
    fastPoints.resize(octaves);
    for (int oct = 0; oct < octaves; ++oct )
      fastPoints[oct].resize(intervals + 1);
    
    for (int oct = 0; oct < octaves; ++oct )
      for (int inter = 0; inter < intervals + 1; ++inter )
      {  
        cv::Mat img8bit, temp;
        temp = gaussPyramid[oct][inter] * 255;
        temp.convertTo(img8bit, CV_8UC1);
        double size = sigma[0] * pow(2.0, oct + (double)inter / intervals);
        cv::FAST(img8bit, fastPoints[oct][inter], this->threshold);
        for (auto& point : fastPoints[oct][inter])
          if (isScaleExtremum(point.pt.y, point.pt.x, oct, inter))
          {
            point.pt.x *= pow( 2.0, oct-1 );
            point.pt.y *= pow( 2.0, oct-1 );          
            point.size = size * 2;
            point.octave = oct;
            point.class_id = inter;
            keypoints.push_back(point);
          }
      }        
    return;

  }

  void FAST_Laplace::detector(cv::Mat& img, 
                                std::vector<cv::KeyPoint>& keypoints)
  {
    buildGaussPyr(img);
    
    std::vector< std::vector< std::vector<cv::KeyPoint> > > fastPoints;
    fastPoints.resize(octaves);
    for (int oct = 0; oct < octaves; ++oct )
      fastPoints[oct].resize(intervals + 1);
    
    for (int oct = 0; oct < octaves; ++oct )
      for (int inter = 0; inter < intervals + 1; ++inter )
      {  
        cv::Mat img8bit, temp;
        temp = gaussPyramid[oct][inter] * 255;
        temp.convertTo(img8bit, CV_8UC1);
        double size = sigma[0] * pow(2.0, oct + (double)inter / intervals);
        cv::FAST(img8bit, fastPoints[oct][inter], this->threshold);
        for (auto& point : fastPoints[oct][inter])
          if (isScaleExtremum(point.pt.y, point.pt.x, oct, inter))
          {
            point.pt.x *= pow( 2.0, oct-1 );
            point.pt.y *= pow( 2.0, oct-1 );
            point.size = size * 2;
            point.octave = oct;
            point.class_id = inter;
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
  void FAST_Laplace::createBaseImage(const cv::Mat& src, cv::Mat& dst,
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
  void FAST_Laplace::buildGaussPyr(const cv::Mat& img)
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
        else{std::cout << sigma[inter] << std::endl;
         	cv::GaussianBlur(gaussPyramid[oct][inter-1], gaussPyramid[oct][inter], cv::Size(), sigma[inter]);       	
}

        }

    return;
  }




	  
	bool FAST_Laplace::isScaleExtremum(int row, int col, int octave, int interval)
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
	
	double FAST_Laplace::getPixelLaplacian(const cv::Mat& img, int row, int col)
	{
	  return ( img.at<float>(row - 1, col) +
	           img.at<float>(row, col - 1) +
	           img.at<float>(row + 1, col) +
	           img.at<float>(row, col + 1) -
	           img.at<float>(row, col) * 4 );
	}


  
}				/* End Namespace pk */
