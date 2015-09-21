#ifndef  _ICS_
#define  _ICS_


#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

class ICS
{
	public:
		ICS(){};
		~ICS(){};

		void calculateNormalizationMat(vector<Point2d> &inp, Mat &ics);

	private:
		void calculateFirstScatter(vector<Point2d> &inp, Mat &S1, Mat &S1_invSqrt);
		//void rotateWhitenedData(vector<Point2d> &inp, vector<Point2d> &outp, Mat &U2, Mat &U2t);
		bool calculateSecondScatter(vector<Point2d> &inp, Mat &S2);

		double 	frobeniusNorm(Mat &inp);
		void 	sumOfSignOuters(Mat inpData, Mat &res);

		Mat calcMatSqrt(Mat &inp);

		Point2d calcMedian(cv::Mat Input);
		Point2d calcMean(cv::Mat Input);
};


#endif
