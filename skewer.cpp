#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>
#include "sipht.hpp"
#include "pk.hpp"
#ifdef __cplusplus
#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#endif

using namespace pk;

void getSize(cv::Mat& img, cv::Mat& inTrans, cv::Mat& outTrans, cv::Size& size);
//void removeDuplicates(std::vector<cv::KeyPoint>& keypoints, double tol);

int main(int argc, char** argv )
{
    if ( argc < 6 | argc > 7 )
    {
        printf("usage: skewer <Image_Path> <Output_File> <X_Scale> <X_Shear> <Y_Shear> <Y_Scale>\n");
        return -1;
    }

    cv::Mat img1, img2, img1_colour, img2_colour, mask;
    img1 = cv::imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
    img1_colour = cv::imread( argv[1], CV_LOAD_IMAGE_COLOR);

    double xScale = atof(argv[3]);
    double xShear = tan ( atof(argv[4]) );
    double yShear = tan ( atof(argv[5]) );
    double yScale = 1/xScale;
    if (7 == argc) yScale = atof(argv[6]);
    
    cv::Mat transform = (cv::Mat_<double>(2,3) << xScale, xShear, 0, yShear, yScale, 0);
    cv::Mat identity = cv::Mat::eye(2,3,CV_64F);
    
    cv::Size dims;
    cv::Mat trans;
    
    getSize(img1, transform, trans, dims);
    warpAffine(img1, img2, trans, dims, 
                   CV_INTER_CUBIC ,IPL_BORDER_CONSTANT);
    warpAffine(img1_colour, img2_colour, trans, dims,
                                        CV_INTER_CUBIC ,IPL_BORDER_CONSTANT);
    cv::namedWindow("original", CV_WINDOW_AUTOSIZE );
    cv::imshow("original", img1_colour);  
    cv::waitKey(0);
    cv::namedWindow("warped", CV_WINDOW_AUTOSIZE );
    cv::imshow("warped", img2_colour);   
    cv::waitKey(0);
//    imwrite("graffiti_thumb_yscale.png", img2_colour);

//    cv::Size dims = img2.size();
//    int type = img2.type();
//    mask = Mat(dims, type, Scalar::all(0));
//    mask(Rect(0,dims.height/4,dims.width/4,(3*dims.height)/4)) = 1; 

    if ( !img1.data | !img2.data )
    {
        printf("No image data \n");
        return -1;
    }

  std::vector<cv::KeyPoint> img1_points, img2_points;

	SIFT::CommonParams common;
	SIFT::DetectorParams detector;
	SIFT::DescriptorParams descriptor;

	SIFT sipht(common, detector, descriptor);
/*/
  cv::SIFT::CommonParams com;
  cv::SIFT::DetectorParams det;
  cv::SIFT::DescriptorParams des;
	cv::SIFT sift(com, det, des);
//*/
	cv::Mat img1_descriptors, img2_descriptors;

//    sipht.setScaleType(atoi(argv[4]));

    sipht(img1, mask, img1_points, img1_descriptors, identity);
    sipht(img2, mask, img2_points, img2_descriptors, transform);
/*/
    for ( int thresh = 2; thresh <= 20; thresh += 2)
    {
//    cv:SIFT sift(0.04, (double)thresh);
//    sift(img1, mask, img1_points, img1_descriptors);
//    sift(img2, mask, img2_points, img2_descriptors);
    SIFT sipht(0.04, (double)thresh);
    sipht(img1, mask, img1_points, img1_descriptors, identity);
    sipht(img2, mask, img2_points, img2_descriptors, transform);
    removeDuplicates(img1_points, 0.5);
    removeDuplicates(img2_points, 0.5);
    int orig = 0, warp = 0;

    for (auto&& i : img1_points)  ++orig;
    for (auto&& j : img2_points)  ++warp;

    std::ofstream file;
    std::string name = std::string("SIPHT-") + std::to_string(thresh);
    file.open(name, std::ios::out );
    file << "# " << argv[1] << "\t[" << xScale << "," << xShear
         << ";" << yShear << "," << yScale << "]\n"
         << "# unwarped points: " << orig << "\twarped points: " << warp
         << "\n# fraction found\t" << "fraction correct\n";
    

    for (double tol = 0.01; tol < 4.01; tol += 0.1)
    {
      int count = 0;

      for (auto&& i : img1_points)
      {
         
          cv::Mat original = (cv::Mat_<double>(3,1) << i.pt.x, i.pt.y, 1);
          original = transform * original;              
//          cv::Point3f original(i.pt.x, i.pt.y, i.angle);                
          for (auto&& j : img2_points)
          {
              cv::Mat warped = (cv::Mat_<double>(2,1) << j.pt.x, j.pt.y);
              if (cv::norm(warped - original) < tol)
                  ++count;
          }
      }
      std::cout << "tolerance:\t" << tol << std::endl; 
      std::cout << "original method found:\t" << orig << std::endl;
      std::cout << "modified method found:\t" << warp << std::endl;
      std::cout << "number of matching points:\t" << count << std::endl;
      file  << tol << "\t" << double(warp)/orig << "\t" << double(count)/orig << "\n";
      if (tol == 0.01) tol = 0;
    }
    }
//*/
    std::vector< std::vector<cv::Mat> > DoG = sipht.getScalePyramid(img2, transform);
    std::vector< std::vector<cv::Mat> > CaT = sipht.getScalePyramid(img2, identity);
    cv::namedWindow("ScaleSpace", CV_WINDOW_AUTOSIZE );
    std::vector<cv::KeyPoint> localPoints;
    
    for (auto octave = DoG.begin(); octave != DoG.end(); ++octave)
      for (auto interval = octave->begin(); interval != octave->end(); ++interval) 
      {
//        std::cout << "Total " << cv::sum(*interval)[0] << std::endl; 
        cv::Mat lol = (*interval) * 1e1; 
        cv::imshow("ScaleSpace", lol);   
        cv::waitKey(0);
          scale_space_extrema((*interval), common.nOctaveLayers, 
                              detector.threshold, detector.edgeThreshold,
                              localPoints);
          std::cout << "number of local maxes: " << localPoints.size() << std::endl;
      }
//*
    cv::Mat output;

    cv::drawKeypoints(img1, img1_points, output, cv::Scalar::all(-1), 4);
    cv::namedWindow("Keypoints", CV_WINDOW_KEEPRATIO );
    cv::imshow("Keypoints", output);
 
    cv::drawKeypoints(img2, img2_points, output, cv::Scalar::all(-1), 4);
    cv::namedWindow("Warppoints", CV_WINDOW_KEEPRATIO );
    cv::imshow("Warppoints", output);   
    cv::waitKey(0);
// */
/*
	cv::Ptr<cv::DescriptorMatcher> catcher = cv::DescriptorMatcher::create("BruteForce");
	std::vector< std::vector<cv::DMatch> > kmatches;
	catcher->knnMatch(img1_descriptors, img2_descriptors, kmatches, 2);
	
	std::vector<cv::DMatch> matches;
	unsigned len = kmatches.size();

	for (unsigned i = 0; i < len; ++i)
	{
		if (kmatches[i][0].distance / kmatches[i][1].distance < 0.7)
			matches.push_back(kmatches[i][1]);
	}
		
    cv::drawMatches(img1_colour, img1_points, img2_colour, img2_points,
    			matches, output, cv::Scalar::all(-1), cv::Scalar::all(2), std::vector<char>()
    			, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    cv::namedWindow("Matches", CV_WINDOW_KEEPRATIO );
    cv::imshow("Matches", output);
    
    cv::waitKey(0);
*/
    return 0;
}

void getSize(cv::Mat& img, cv::Mat& inTrans, cv::Mat& outTrans, cv::Size& size)
{
    std::vector<cv::Mat> cnrs;
    cnrs.push_back((cv::Mat_<double>(3,1) << 0, 0, 1));
    cnrs.push_back((cv::Mat_<double>(3,1) << img.cols, 0, 1));
    cnrs.push_back((cv::Mat_<double>(3,1) << 0, img.rows, 1));
    cnrs.push_back((cv::Mat_<double>(3,1) << img.cols, img.rows, 1));
    double max = std::numeric_limits<double>::max();
    double dims[4] {0,0,max,max};
    for (auto&& pt : cnrs)
    {
      pt = inTrans * pt;
      dims[0] = std::max(pt.at<double>(0,0), dims[0]);
      dims[1] = std::max(pt.at<double>(1,0), dims[1]);
      dims[2] = std::min(pt.at<double>(0,0), dims[2]);
      dims[3] = std::min(pt.at<double>(1,0), dims[3]);
    }
    
    int cols = 2*(static_cast<int>((dims[0] - dims[2])/2));
    int rows = 2*(static_cast<int>((dims[1] - dims[3])/2));
 
    outTrans = inTrans.clone();
    
//    outTrans.at<double>(1,2) = 
//    cols/2 + 1 - (cnrs[3].at<double>(1,0) - cnrs[0].at<double>(1,0) + 1)/2;
    
//    outTrans.at<double>(0,2) = 
//    rows/2 + 1 - (cnrs[3].at<double>(0,0) - cnrs[0].at<double>(0,0) + 1)/2;

    std::cout << img.rows << " " << img.cols << std::endl;
    std::cout << rows << " " << cols << std::endl;
    size = cv::Size(cols,rows);

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

