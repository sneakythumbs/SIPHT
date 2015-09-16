#include "FAST_Harris.hpp"

namespace pk
{

  /** Function that computes the Harris response in a 9 x 9 patch at a given point in an image
  * @param patch the 9 x 9 patch
  * @param k the k in the Harris formula
  * @param dX_offsets pre-computed offset to get all the interesting dX values
  * @param dY_offsets pre-computed offset to get all the interesting dY values
  * @return
  */
  template<typename PatchType, typename SumType>
    inline float harris(const cv::Mat& patch, float k, const std::vector<int> &dX_offsets,
                        const std::vector<int> &dY_offsets)
    {
      float a = 0, b = 0, c = 0;

      static cv::Mat_<SumType> dX(9, 7), dY(7, 9);
      SumType * dX_data = reinterpret_cast<SumType*> (dX.data), *dY_data = reinterpret_cast<SumType*> (dY.data);
      SumType * dX_data_end = dX_data + 9 * 7;
      PatchType * patch_data = reinterpret_cast<PatchType*> (patch.data);
      int two_row_offset = (int)(2 * patch.step1());
      std::vector<int>::const_iterator dX_offset = dX_offsets.begin(), dY_offset = dY_offsets.begin();
      // Compute the differences
      for (; dX_data != dX_data_end; ++dX_data, ++dY_data, ++dX_offset, ++dY_offset)
      {
        *dX_data = (SumType)(*(patch_data + *dX_offset)) - (SumType)(*(patch_data + *dX_offset - 2));
        *dY_data = (SumType)(*(patch_data + *dY_offset)) - (SumType)(*(patch_data + *dY_offset - two_row_offset));
      }

      // Compute the Scharr result
      dX_data = reinterpret_cast<SumType*> (dX.data);
      dY_data = reinterpret_cast<SumType*> (dY.data);
      for (size_t v = 0; v <= 6; v++, dY_data += 2)
      {
        for (size_t u = 0; u <= 6; u++, ++dX_data, ++dY_data)
        {
          // 1, 2 for Sobel, 3 and 10 for Scharr
          float Ix = (float)(1 * (*dX_data + *(dX_data + 14)) + 2 * (*(dX_data + 7)));
          float Iy = (float)(1 * (*dY_data + *(dY_data + 2)) + 2 * (*(dY_data + 1)));

          a += Ix * Ix;
          b += Iy * Iy;
          c += Ix * Iy;
        }
      }

      return ((a * b - c * c) - (k * ((a + b) * (a + b))));
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /** Class used to compute the cornerness of specific points in an image */
  struct HarrisResponse
  {
    /** Constructor
     * @param image the image on which the cornerness will be computed (only its step is used
     * @param k the k in the Harris formula
     */
    explicit HarrisResponse(const cv::Mat& image, double k = 0.04);

    /** Compute the cornerness for given keypoints
     * @param kpts points at which the cornerness is computed and stored
     */
    void operator()(std::vector<cv::KeyPoint>& kpts) const;
  private:
    /** The cached image to analyze */
    cv::Mat image_;

    /** The k factor in the Harris corner detection */
    double k_;

    /** The offset in X to compute the differences */
    std::vector<int> dX_offsets_;

    /** The offset in Y to compute the differences */
    std::vector<int> dY_offsets_;
  };

  /** Constructor
   * @param image the image on which the cornerness will be computed (only its step is used
   * @param k the k in the Harris formula
   */
  HarrisResponse::HarrisResponse(const cv::Mat& image, double k) :
    image_(image), k_(k)
  {
    // Compute the offsets for the Harris corners once and for all
    dX_offsets_.resize(7 * 9);
    dY_offsets_.resize(7 * 9);
    std::vector<int>::iterator dX_offsets = dX_offsets_.begin(), dY_offsets = dY_offsets_.begin();
    int x, y, image_step = (int)image.step1();
    for (y = 0; y <= 6 * image_step; y += image_step)
    {
      int dX_offset = y + 2, dY_offset = y + 2 * image_step;
      for (x = 0; x <= 6; ++x)
      {
        *(dX_offsets++) = dX_offset++;
        *(dY_offsets++) = dY_offset++;
      }
      for (size_t x = 7; x <= 8; ++x)
        *(dY_offsets++) = dY_offset++;
    }
  
    for (y = 7 * image_step; y <= 8 * image_step; y += image_step)
    {
      int dX_offset = y + 2;
      for (x = 0; x <= 6; ++x)
        *(dX_offsets++) = dX_offset++;
    }
  }

  /** Compute the cornerness for given keypoints
   * @param kpts points at which the cornerness is computed and stored
   */
  void HarrisResponse::operator()(std::vector<cv::KeyPoint>& kpts) const
  {
    // Those parameters are used to match the OpenCV computation of Harris corners
    float scale = (1 << 2) * 7.0f * 255.0f;
    scale = 1.0f / scale;
    float scale_sq_sq = scale * scale * scale * scale;
  
    // define it to 1 if you want to compare to what OpenCV computes
  
    for (std::vector<cv::KeyPoint>::iterator kpt = kpts.begin(), kpt_end = kpts.end(); kpt != kpt_end; ++kpt)
    {
      if (kpt->pt.x - 4 <= 0    ||
          kpt->pt.x + 4 >= image_.cols ||
          kpt->pt.y - 4 <= 0    ||
          kpt->pt.y + 4 >= image_.rows )
      { kpt->response = 0; }
      else
      {
        cv::Mat patch = image_(cv::Rect(cvRound(kpt->pt.x) - 4, cvRound(kpt->pt.y) - 4, 9, 9));
  
        // Compute the response
        kpt->response = harris<uchar, int> (patch, (float)k_, dX_offsets_, dY_offsets_) * scale_sq_sq;
      }
    }
  }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  FAST_Harris::FAST_Harris( float fThresh, float hThresh, int _k, int oct, int inter, double _sigma )
  {
    fastThresh = fThresh;
    harrisThresh = hThresh;
    k = _k;
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
  FAST_Harris::FAST_Harris( const cv::Mat& img, float fThresh, float hThresh, int _k, int oct, int inter, double _sigma )
  {
    fastThresh = fThresh;
    harrisThresh = hThresh;
    k = _k;
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
    buildGaussPyr(img);
  }

  void FAST_Harris::initialise(const cv::Mat& img)
  {
    buildGaussPyr(img);
  }

  void FAST_Harris::detector(std::vector<cv::KeyPoint>& keypoints)
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
//        cv::FastFeatureDetector fd(this->fastThresh, true);
//        fd.detect(img8bit, fastPoints[oct][inter], cv::Mat());

        cv::FAST(img8bit, fastPoints[oct][inter], this->fastThresh);
//        HarrisResponse harry(gaussPyramid[oct][inter]);
//        harry(fastPoints[oct][inter]);
        for (auto& point : fastPoints[oct][inter])
          if (isScaleExtremum(point.pt.y, point.pt.x, oct, inter))
//          if(point.response > this->harrisThresh)
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

  void FAST_Harris::detector(cv::Mat& img, 
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
        cv::FAST(img8bit, fastPoints[oct][inter], this->fastThresh);
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
  void FAST_Harris::createBaseImage(const cv::Mat& src, cv::Mat& dst,
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
  void FAST_Harris::buildGaussPyr(const cv::Mat& img)
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
          cv::resize(gaussPyramid[oct-1][intervals], gaussPyramid[oct][inter], cv::Size(), 0.5, 0.5, CV_INTER_AREA);

        /* blur the current octave's last image to create the next one */
        else
         	cv::GaussianBlur(gaussPyramid[oct][inter-1], gaussPyramid[oct][inter], cv::Size(), sigma[inter]);       	
        }

    return;
  }

  
	bool FAST_Harris::isScaleExtremum(int row, int col, int octave, int interval)
	{
//*/	
    double val = getPixelHarrisScore(row, col, octave, interval);
    std::cout << val << std::endl;
    if (val > this->harrisThresh)
	    return true;
	  return false;
//*/
	}
  double FAST_Harris::getPixelHarrisScore(int row, int col, int octave, int interval)
  {
//*/    
    cv::Mat zeroth, first, xKern, yKern;
    GaussDerivKernel1D(sigma[interval], 0, zeroth);
    GaussDerivKernel1D(sigma[interval], 1, first);
    
    int xmax = gaussPyramid[octave][interval].cols - 1;
    int ymax = gaussPyramid[octave][interval].rows - 1;
    int size = first.cols/2;
    if (col - size <= 0    ||
        col + size >= xmax ||
        row - size <= 0    ||
        row + size >= ymax )
     { return 0; }
        
    xKern = zeroth.t() * first;
    yKern = first.t() * zeroth;
    
    cv::Mat Lxx, Lyy, Lxy;
    int depth = gaussPyramid[octave][interval].depth();
    cv::filter2D(gaussPyramid[octave][interval](cv::Rect(col-size, row-size, 2*size+1, 2*size+1)), Lxx, depth, xKern);
    cv::filter2D(gaussPyramid[octave][interval](cv::Rect(col-size, row-size, 2*size+1, 2*size+1)), Lyy, depth, yKern); 
    
    double stddev = sigma[0] * pow(2.0, octave + (double)interval / intervals);
    double scale = stddev*stddev;
    
    Lxy = Lxx.mul(Lyy) * scale;
    
    Lxx = Lxx.mul(Lxx) * scale;
    
    Lyy = Lyy.mul(Lyy) * scale;
    
    return Lxx.at<float>(size+1,size+1) * Lyy.at<float>(size+1,size+1)
         - Lxy.at<float>(size+1,size+1) * Lxy.at<float>(size+1,size+1)
    - k * (Lxx.at<float>(size+1,size+1) + Lyy.at<float>(size+1,size+1))
        * (Lxx.at<float>(size+1,size+1) + Lyy.at<float>(size+1,size+1));
//*/
/*/  
    cv::Mat harris;
    harrisCorner(gaussPyramid[octave][interval], harris, sigma[interval], k);
    return harris.at<float>(row, col);
//*/
  }
    
}


