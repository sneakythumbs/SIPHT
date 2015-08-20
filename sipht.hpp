/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __SIPHT_HPP__
#define __SIPHT_HPP__



#include "opencv2/core/core.hpp"
#include "opencv2/core/internal.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"

#include <stdio.h>
#ifdef __cplusplus
#include <limits>
#endif


namespace pk
{

	class  SIFT
	{
	public:
		  cv::Mat transformation;
    	struct  CommonParams
    	{
    	    static const int DEFAULT_NOCTAVES = 4;
    	    static const int DEFAULT_NOCTAVE_LAYERS = 3;
    	    static const int DEFAULT_FIRST_OCTAVE = -1;
    	    enum { FIRST_ANGLE = 0, AVERAGE_ANGLE = 1 };
	
    	    CommonParams();
    	    CommonParams( int _nOctaves, int _nOctaveLayers, int /*_firstOctave*/, int /*_angleMode*/ );
    	    CommonParams( int _nOctaves, int _nOctaveLayers );
    	    int nOctaves, nOctaveLayers;
    	    int firstOctave; // it is not used now (firstOctave == 0 always)
    	    int angleMode;   // it is not used now
    	    int scaleType;
    	};
	
    	struct  DetectorParams
    	{
    	    static double GET_DEFAULT_THRESHOLD() { return 0.04; }
    	    static double GET_DEFAULT_EDGE_THRESHOLD() { return 10.0; }
	
    	    DetectorParams();
    	    DetectorParams( double _threshold, double _edgeThreshold );
    	    double threshold, edgeThreshold;
    	};
	
    	struct  DescriptorParams
    	{
    	    static double GET_DEFAULT_MAGNIFICATION() { return 3.0; }
    	    static const bool DEFAULT_IS_NORMALIZE = true;
    	    static const int DESCRIPTOR_SIZE = 128;
	
    	    DescriptorParams();
    	    DescriptorParams( double _magnification, bool /*_isNormalize*/, bool _recalculateAngles );
    	    DescriptorParams( bool _recalculateAngles );
    	    double magnification;
    	    bool isNormalize; // it is not used now (true always)
    	    bool recalculateAngles;
    	};
	
	
    	SIFT();
    	//! sift-detector constructor
    	SIFT( double _threshold, double _edgeThreshold,
    	      int _nOctaves=CommonParams::DEFAULT_NOCTAVES,
    	      int _nOctaveLayers=CommonParams::DEFAULT_NOCTAVE_LAYERS,
    	      int _firstOctave=CommonParams::DEFAULT_FIRST_OCTAVE,
    	      int _angleMode=CommonParams::FIRST_ANGLE );
    	//! sift-descriptor constructor
    	SIFT( double _magnification, bool _isNormalize=true,
    	      bool _recalculateAngles = true,
    	      int _nOctaves=CommonParams::DEFAULT_NOCTAVES,
    	      int _nOctaveLayers=CommonParams::DEFAULT_NOCTAVE_LAYERS,
    	      int _firstOctave=CommonParams::DEFAULT_FIRST_OCTAVE,
    	      int _angleMode=CommonParams::FIRST_ANGLE );
    	SIFT( const CommonParams& _commParams,
    	      const DetectorParams& _detectorParams = DetectorParams(),
    	      const DescriptorParams& _descriptorParams = DescriptorParams() );
    	SIFT( const CommonParams& _commParams,
            const DetectorParams& _detectorParams,
            const DescriptorParams& _descriptorParams,
            cv::Mat& transform );
	
    	//! returns the descriptor size in floats (128)
    	int descriptorSize() const;
    	//! finds the keypoints using SIFT algorithm
    	void operator()(const cv::Mat& img, const cv::Mat& mask,
    	                std::vector<cv::KeyPoint>& keypoints,
    	                cv::Mat& transform) const;
    	//! finds the keypoints and computes descriptors for them using SIFT algorithm.
    	//! Optionally it can compute descriptors for the user-provided keypoints
    	void operator()(const cv::Mat& img, const cv::Mat& mask,
    	                std::vector<cv::KeyPoint>& keypoints,
    	                cv::Mat& descriptors,
    	                cv::Mat& transform,
    	                bool useProvidedKeypoints=false) const;
	
    	CommonParams getCommonParams () const;
    	DetectorParams getDetectorParams () const;
    	DescriptorParams getDescriptorParams () const;
    	void setScaleType(int scaleType);
    	std::vector< std::vector<cv::Mat> > getScalePyramid(cv::Mat& img, cv::Mat& transform);
	
	protected:
    	CommonParams commParams;
    	DetectorParams detectorParams;
    	DescriptorParams descriptorParams;
	};

} /* namespace pk */    

#endif
/* End of file. */
