#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <cmath>
#include "sipht.hpp"

using namespace pk;

int main(int argc, char** argv )
{
    if ( argc != 5 )
    {
        printf("usage: skewer <Image_Path> <X_Skew> <Y_Skew> <detector_type>\n");
        return -1;
    }

    cv::Mat img1, img2, img1_colour, img2_colour, mask;
    img1 = cv::imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
    img1_colour = cv::imread( argv[1], CV_LOAD_IMAGE_COLOR);

    double xSkew = tan ( atof(argv[2]) );
    double ySkew = tan ( atof(argv[3]) );
    cv::Mat transform = (cv::Mat_<double>(2,3) << 1, xSkew, 0, ySkew, 1, 0);
    warpAffine(img1, img2, transform, img1.size(), 
                   CV_INTER_LINEAR ,IPL_BORDER_CONSTANT);
    warpAffine(img1_colour, img2_colour, transform, img1_colour.size(),
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

    sipht.setScaleType(atoi(argv[4]));
    sipht(img1, mask, img1_points, img1_descriptors);
    sipht(img2, mask, img2_points, img2_descriptors);
    long count = 0;
    double tol = 5E-1;
    for (auto i : img1_points)
    {
        cv::Point2f warped;
        warped.x = i.pt.x * transform.at<double>(0,0) 
        		 + i.pt.y * transform.at<double>(0,1);
        warped.y = i.pt.x * transform.at<double>(1,0) 
                 + i.pt.y * transform.at<double>(1,1);                
        for (auto j : img2_points)
        {
//        std::cout << warped << "  " << j.pt << std::endl; 
            if (norm(warped - j.pt) < tol)
                ++count;
        }
    } 
    std::cout << count << std::endl;
    cv::Mat output;
//*
    cv::drawKeypoints(img1, img1_points, output, cv::Scalar::all(-1), 4);
    
    cv::namedWindow("Keypoints", CV_WINDOW_KEEPRATIO );
    cv::imshow("Keypoints", output);
    
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
