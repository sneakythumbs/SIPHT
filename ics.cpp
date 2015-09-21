#include "ics.h"


void
ICS::calculateNormalizationMat(vector<Point2d> &inp, Mat &ics)
{
	Mat S1, B1,
		S2;

	calculateFirstScatter( inp, S1, B1);
	calculateSecondScatter( inp, S2);

	Mat B2 = B1 * S2 * B1;

	Mat V, D;
	eigen(B2, D, V);

	if(D.at<double>(0,0) < D.at<double>(0,1))
	{
		Mat tmp = (Mat_<double>(2,2) << V.at<double>(1,0), V.at<double>(0,0), V.at<double>(1,1), V.at<double>(0,1));
		V = tmp;
	}
	else
	{
		Mat tmp = (Mat_<double>(2,2) << V.at<double>(0,0), V.at<double>(1,0), V.at<double>(0,1), V.at<double>(1,1));
		V = tmp;
	}

	Mat U2_tr;
	transpose(V, U2_tr);

	Mat B = U2_tr * B1;

	Mat B_tr;
	transpose(B, B_tr);

	Mat data = Mat(inp.size(), 2, CV_64FC1);
	MatIterator_<double> it_data = data.begin<double>();
	for( vector<Point2d>::iterator it_inp = inp.begin(); it_inp != inp.end(); it_inp++ )
	{
		*it_data++ = it_inp->x;
		*it_data++ = it_inp->y;
	}
	Mat Z1 = data * B_tr;

	//Scalar meanZ1 = mean(Z1);
	Point2d mean = calcMean(Z1);
	Point2d median = calcMedian(Z1);

	if( mean.x - median.x < 0)
	{
		B.at<double>(0,0) *= -1;
		B.at<double>(0,1) *= -1;
	}
	if( mean.y - median.y < 0)
	{
		B.at<double>(1,0) *= -1;
		B.at<double>(1,1) *= -1;
	}

	ics = B.clone();
}


Point2d
ICS::calcMean(cv::Mat Input)
{
	MatConstIterator_<double> it = Input.begin<double>();

	double mx = 0.0,
		   my = 0.0;
	while( it < Input.end<double>() )
	{
		mx += *it++;
		my += *it++;
	}

	Point2d res(mx/Input.rows, my/Input.rows);

	return res;
}


void
ICS::calculateFirstScatter(vector<Point2d> &inp, Mat &S1, Mat &S1_invSqrt)
{
	vector<Point2d>::iterator it;


	// calculate average gradient
	Point2d avgGradient = Point2d(0.0,0.0);
	int cnt = 0;
	for(it = inp.begin() ; it != inp.end(); it++ )
	{
		avgGradient.x += it->x;
		avgGradient.y += it->y;
	}
	avgGradient.x /= double(inp.size());
	avgGradient.y /= double(inp.size());


	// calculate covariance matrix
	Mat cov = Mat::zeros( 2, 2, CV_64FC1);
	double normGradX = 0.0,
		   normGradY = 0.0;
	for(it = inp.begin() ; it != inp.end(); it++ )
	{
		normGradX = it->x - avgGradient.x;
		normGradY = it->y - avgGradient.y;
		cov.at<double>(0,0) += (normGradX * normGradX);
		cov.at<double>(0,1) += (normGradY * normGradX);
		cov.at<double>(1,1) += (normGradY * normGradY);
	}
	cov.at<double>(1,0) = cov.at<double>(0,1);
	cov /= double(inp.size());

	// write output
	S1 = cov.clone();

	// calculate INVSqrt of covariance matrix
	Mat w, u, vt;
	SVDecomp(cov, w, u, vt);
	Mat invSqrtW = ( Mat_<double>(2, 2) << 1/sqrt(w.at<double>(0,0)), 0, 0, 1/sqrt(w.at<double>(0,1)) );
	cov = u * invSqrtW * vt;

	// write output
	S1_invSqrt = cov.clone();
}


double
ICS::frobeniusNorm(Mat &inp)
{
	assert(inp.type() == CV_64F);
	Mat tmp = inp.t() * inp;

	return sqrt(tmp.at<double>(0,0) * tmp.at<double>(1,1));
}


void
ICS::sumOfSignOuters(Mat inpData, Mat &res)
{
   Mat signs = inpData;
   res = Mat::zeros( 2, 2, CV_64FC1 );

   int n= signs.rows;
   int k= signs.cols;
   int i;
   int j;
   int m;
   int p=0;

   double *r;
   r = new double[n];

   //compute norms:
   for (i=0; i<n; i++)
	   r[i]=0.0;

   for (i=0; i<n; i++)
   {
	   for(j=0; j<k; j++)
		   r[i] += signs.at<double>(i,j)*signs.at<double>(i,j);
	   r[i]=sqrt(r[i]);
   }

   //compute signs:
   for(i=0; i<n; i++)
	   for(j=0; j<k; j++)
		   signs.at<double>(i,j) /= r[i];

  //compute the sum of outer products:
  for(j=0;j<k;j++)
	  for(m=0;m<k;m++)
      {
		  for( i=0; i<n; i++ )
			  res.at<double>(j,m) += signs.at<double>(i,j)*signs.at<double>(i,m);
		  p++;
      }

  delete [] r;
}


bool
ICS::calculateSecondScatter(vector<Point2d> &inp, Mat &S2)
{
	int steps = 100;
	double eps = 0.000001;

	Point2d mean = Point2d(0,0);

	for( vector<Point2d>::iterator it = inp.begin(); it != inp.end(); it++ )
	{
		mean.x += it->x;
		mean.y += it->y;
	}
	mean.x /= inp.size();
	mean.y /= inp.size();

	for( vector<Point2d>::iterator it = inp.begin(); it != inp.end(); it++ )
	{
		it->x -= mean.x;
		it->y -= mean.y;
	}

	Mat inpMat = Mat(inp.size(), 2, CV_64FC1);
	int cnt = 0;
	for( vector<Point2d>::iterator it = inp.begin(); it != inp.end(); it++, cnt++)
	{
		inpMat.at<double>(cnt, 0) = it->x;
		inpMat.at<double>(cnt, 1) = it->y;
	}

	Mat V0 = inpMat.t() * inpMat;
	V0 = V0.inv();

	int iter = 0;
	double differ = 1.0;

	Mat res, sqrtV, data,
		Vnew, diffMat;
	while( iter <= steps && differ > eps )
	{
		sqrtV = calcMatSqrt(V0);
		data = inpMat * sqrtV;

		sumOfSignOuters( data, res );

		Vnew = sqrtV * res.inv() * sqrtV;
		Vnew /= (Vnew.at<double>(0,0) + Vnew.at<double>(1,1));

		diffMat = Vnew - V0;
		differ = frobeniusNorm(diffMat);

		V0 = Vnew.clone();
		iter++;
	}

	Mat Vshape = Vnew.inv();

	double detVshape = Vshape.at<double>(0,0)*Vshape.at<double>(1,1) - Vshape.at<double>(1,0)*Vshape.at<double>(0,1);

	if( detVshape <= 0 )
		return false;
	S2 = Vshape / sqrt(detVshape);

	return true;
}


Mat
ICS::calcMatSqrt(Mat &inp)
{
	Mat w, u, vt;
	SVDecomp(inp, w, u, vt);

	Mat W =( Mat_<double>(2,2) << sqrt(w.at<double>(0,0)), 0, 0, sqrt(w.at<double>(0,1)) );
	Mat out = u * W * vt;
	return out.clone();
}


Point2d
ICS::calcMedian(cv::Mat Input)
{
	Point2d res;


	std::vector<double> vecFromMatCol0,
						vecFromMatCol1;
	vecFromMatCol0.reserve(Input.rows);
	vecFromMatCol1.reserve(Input.rows);

	MatConstIterator_<double> it = Input.begin<double>();
	for( ; it < Input.end<double>(); it++)
	{
		vecFromMatCol0.push_back(*it);
		it++;
		vecFromMatCol1.push_back(*it);
	}

	std::nth_element(vecFromMatCol0.begin(), vecFromMatCol0.begin() + vecFromMatCol0.size() / 2, vecFromMatCol0.end());
	res.x = vecFromMatCol0[vecFromMatCol0.size() / 2];

	std::nth_element(vecFromMatCol1.begin(), vecFromMatCol1.begin() + vecFromMatCol1.size() / 2, vecFromMatCol1.end());
	res.y = vecFromMatCol1[vecFromMatCol1.size() / 2];

	return res;
}

//double medianMat(cv::Mat Input, int nVals)
//{
//
//	// COMPUTE HISTOGRAM OF SINGLE CHANNEL MATRIX
//	float range[] = { 0, nVals };
//	const float* histRange = { range };
//	bool uniform = true; bool accumulate = false;
//	cv::Mat hist;
//	calcHist(&Input, 1, 0, cv::Mat(), hist, 1, &nVals, &histRange, uniform, accumulate);
//
//	// COMPUTE CUMULATIVE DISTRIBUTION FUNCTION (CDF)
//	cv::Mat cdf;
//	hist.copyTo(cdf);
//	for (int i = 1; i <= nVals-1; i++){
//		cdf.at<float>(i) += cdf.at<float>(i - 1);
//	}
//	cdf /= Input.total();
//
//	// COMPUTE MEDIAN
//	double medianVal;
//	for (int i = 0; i <= nVals-1; i++){
//		if (cdf.at<float>(i) >= 0.5) { medianVal = i;  break; }
//	}
//	return medianVal/nVals;
//}
