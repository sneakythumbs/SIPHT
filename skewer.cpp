#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <cmath>
#include "sipht.hpp"
#ifdef __cplusplus
#include <limits>
#endif

using namespace pk;

void getSize(cv::Mat& img, cv::Mat& inTrans, cv::Mat& outTrans, cv::Size& size);
void removeDuplicates(std::vector<cv::KeyPoint>& keypoints, double tol);

int main(int argc, char** argv )
{
    if ( argc != 4 )
    {
        printf("usage: skewer <Image_Path> <X_Skew> <Y_Skew>\n");
        return -1;
    }

    cv::Mat img1, img2, img1_colour, img2_colour, mask;
    img1 = cv::imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
    img1_colour = cv::imread( argv[1], CV_LOAD_IMAGE_COLOR);

    double xSkew = tan ( atof(argv[2]) );
    double ySkew = tan ( atof(argv[3]) );
    
    cv::Mat transform = (cv::Mat_<double>(2,3) << 1, xSkew, 0, ySkew, 1, 0);
    cv::Mat identity = cv::Mat::eye(2,3,CV_64F);
    
    cv::Size dims;
    cv::Mat trans;
    getSize(img1, transform, trans, dims);
    
    warpAffine(img1, img2, trans, dims, 
                   CV_INTER_LINEAR ,IPL_BORDER_CONSTANT);
    warpAffine(img1_colour, img2_colour, trans, dims,
                                        CV_INTER_LINEAR ,IPL_BORDER_CONSTANT);


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

	cv::Mat img1_descriptors, img2_descriptors;

//    sipht.setScaleType(atoi(argv[4]));
    sipht(img1, mask, img1_points, img1_descriptors, identity);
    sipht(img2, mask, img2_points, img2_descriptors, transform);
    int orig = 0, warp = 0, count = 0;
    double tol = 3E-0;
    removeDuplicates(img1_points, 0.5);
    removeDuplicates(img2_points, 0.5);
    for (auto&& j : img2_points)
      ++warp;
    for (auto&& i : img1_points)
    {
        ++orig;
        cv::Mat original = (cv::Mat_<double>(3,1) << i.pt.x, i.pt.y, 1);
        original = transform * original;              
//        cv::Point3f original(i.pt.x, i.pt.y, i.angle);                
        for (auto&& j : img2_points)
        {
            cv::Mat warped = (cv::Mat_<double>(2,1) << j.pt.x, j.pt.y);
            if (cv::norm(warped - original) < tol)
                ++count;
        }
    } 
    std::cout << "original method found:\t" << orig << std::endl;
    std::cout << "modified method found:\t" << warp << std::endl;
    std::cout << "number of matching points:\t" << count << std::endl;
    cv::Mat output;
//*
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
    cnrs.push_back((cv::Mat_<double>(3,1) << img.rows, 0, 1));
    cnrs.push_back((cv::Mat_<double>(3,1) << 0, img.cols, 1));
    cnrs.push_back((cv::Mat_<double>(3,1) << img.rows, img.cols, 1));
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
    
    int rows = 2*(static_cast<int>((dims[0] - dims[2])/2)) + 1;
    int cols = 2*(static_cast<int>((dims[1] - dims[3])/2)) + 1;
 
    outTrans = inTrans.clone();
    
//    outTrans.at<double>(1,2) = 
//    cols/2 + 1 - (cnrs[3].at<double>(1,0) - cnrs[0].at<double>(1,0) + 1)/2;
    
//    outTrans.at<double>(0,2) = 
//    rows/2 + 1 - (cnrs[3].at<double>(0,0) - cnrs[0].at<double>(0,0) + 1)/2;

    size = cv::Size(rows,cols);

}

void removeDuplicates(std::vector<cv::KeyPoint>& keypoints, double tol)
{
  for (auto i = keypoints.begin(); i != keypoints.end(); ++i)
    for (auto j = keypoints.begin(); j != keypoints.end(); ++j)
    {
      if (i == j) continue;
      if (sqrt(pow((*i).pt.x - (*j).pt.x, 2) + 
               pow((*i).pt.y - (*j).pt.y, 2)) < tol)
        keypoints.erase(j);
    
    }



}
