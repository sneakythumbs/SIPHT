#include "ICS_Adaptation.hpp"
#include <algorithm>

/*
 * Conversion between cv::KeyPoint and Elliptic_KeyPoint
 */
void keyPointToElliptic(cv::KeyPoint & keypoint, Elliptic_KeyPoint & ellipticKeypoint)
{
  ellipticKeypoint.centre.x = keypoint.pt.x;
  ellipticKeypoint.centre.y = keypoint.pt.y;
	ellipticKeypoint.axes = cv::Size(keypoint.size*3, keypoint.size*3);

	ellipticKeypoint.phi = keypoint.angle;
	ellipticKeypoint.size = keypoint.size * 2 * 3;
	ellipticKeypoint.si = keypoint.size;
}

void ellipticToKeyPoint(Elliptic_KeyPoint & ellipticKeypoint, cv::KeyPoint & keypoint)
{
  keypoint.pt = ellipticKeypoint.centre;
  keypoint.angle = ellipticKeypoint.phi;
  keypoint.size = ellipticKeypoint.size;
}

/*
 * Calculates second moments matrix in point p
 */
void calcSecondMomentMatrix(const cv::Mat & dx2, const cv::Mat & dxy, const cv::Mat & dy2, cv::Point p, cv::Mat & M)
{
    int x = p.x;
    int y = p.y;

    M.create(2, 2, CV_32FC1);
    M.at<float> (0, 0) = dx2.at<float> (y, x);
    M.at<float> (0, 1) = M.at<float> (1, 0) = dxy.at<float> (y, x);
    M.at<float> (1, 1) = dy2.at<float> (y, x);
}

/*
 * Performs affine adaptation
 */
bool calcAffineAdaptation(const cv::Mat & fimage, Elliptic_KeyPoint & keypoint, int iterations)
{
    cv::Mat_<float> transf(2, 3)/*Trasformation matrix*/,
            size(2, 1)/*Image size after transformation*/, 
            c(2, 1)/*Transformed point*/, 
            p(2, 1) /*Image point*/;
            
    cv::Mat U = cv::Mat::eye(2, 2, CV_32F) * 1; /*Normalization matrix*/

    cv::Mat warpedImg, Mk, Lxm2smooth, Lym2smooth, Lxmysmooth, img_roi, harrisPatch = cv::Mat(3,3,CV_32FC1);;
    float Qinv = 1, q, si = keypoint.si, sd = 0.75 * si;
    bool divergence = false, convergence = false;
    int i = 0;

    //Coordinates in image
    int py = cvRound(keypoint.centre.y);
    int px = cvRound(keypoint.centre.x);

    //Roi coordinates
    int roix, roiy;

    //Coordinates in U-trasformation
    int cx = px;
    int cy = py;
    int cxPr = cx;
    int cyPr = cy;

    float radius = keypoint.size / 2 * 1.4;
    float half_width, half_height;

    cv::Rect roi;
    float ax1, ax2;
    double phi = 0;
    ax1 = ax2 = keypoint.size / 2;
//    cv::Mat drawImg;

    //Affine adaptation
    while (i <= iterations && !divergence && !convergence)
    {
    	//drawImg = fimage.clone();
        //cvtColor(fimage, drawImg, CV_GRAY2RGB);
        
        //Transformation matrix 
        transf.setTo(0);
        cv::Mat col0 = transf.col(0);
        U.col(0).copyTo(col0);
        cv::Mat col1 = transf.col(1);
        U.col(1).copyTo(col1);
        keypoint.transf =cv::Mat(transf);

        cv::Size_<float> boundingBox;
        
        double ac_b2 = cv::determinant(U);
        boundingBox.width = ceil(U.at<float> (1, 1)/ac_b2  * 3 * si*1.4 );
        boundingBox.height = ceil(U.at<float> (0, 0)/ac_b2 * 3 * si*1.4 );

        //Create window around interest point
        half_width = std::min((float) std::min(fimage.cols - px-1, px), boundingBox.width);
        half_height = std::min((float) std::min(fimage.rows - py-1, py), boundingBox.height);
        roix = std::max(px - (int) boundingBox.width, 0);
        roiy = std::max(py - (int) boundingBox.height, 0);
        roi = cv::Rect(roix, roiy, px - roix + half_width+1, py - roiy + half_height+1);
               
		//create ROI
        img_roi = fimage(roi);

        
        //cv::Point within the ROI
        p(0, 0) = px - roix;
        p(1, 0) = py - roiy;

        if (half_width <= 0 || half_height <= 0)
        {
          divergence = true;
          return divergence;
        }

        //Find coordinates of square's angles to find size of warped ellipse's bounding box
        float u00 = U.at<float> (0, 0);
        float u01 = U.at<float> (0, 1);
        float u10 = U.at<float> (1, 0);
        float u11 = U.at<float> (1, 1);

        float minx = u01 * img_roi.rows < 0 ? u01 * img_roi.rows : 0;
        float miny = u10 * img_roi.cols < 0 ? u10 * img_roi.cols : 0;
        float maxx = (u00 * img_roi.cols > u00 * img_roi.cols + u01 * img_roi.rows ? u00
                * img_roi.cols : u00 * img_roi.cols + u01 * img_roi.rows) - minx;
        float maxy = (u11 * img_roi.rows > u10 * img_roi.cols + u11 * img_roi.rows ? u11
                * img_roi.rows : u10 * img_roi.cols + u11 * img_roi.rows) - miny;

        //Shift
        transf.at<float> (0, 2) = -minx;
        transf.at<float> (1, 2) = -miny;
        
        /*float min_width = minx >= 0 ? u00 * img_roi.cols - u01 * img_roi.rows : u00 * img_roi.cols
                + u01 * img_roi.rows;
        float min_height = miny >= 0 ? u11 * img_roi.rows - u10 * img_roi.cols : u10 * img_roi.cols
                + u11 * img_roi.rows;*/

        if (maxx >=  2*radius+1 && maxy >=  2*radius+1)
        {
            //Size of normalized window must be 2*radius
            //Transformation
            cv::Mat warpedImgRoi;
            cv::warpAffine(img_roi, warpedImgRoi, transf, cv::Size(maxx, maxy),cv::INTER_AREA, cv::BORDER_REPLICATE);

            //cv::Point in U-Normalized coordinates
            c = U * p;
            cx = c(0, 0) - minx;
            cy = c(1, 0) - miny;

            
            if (warpedImgRoi.rows > 2 * radius+1 && warpedImgRoi.cols > 2 * radius+1)
            {
                //Cut around normalized patch
                roix = std::max((float)(cx - ceil(radius)), 0.0f);
                roiy = std::max((float)(cy - ceil(radius)), 0.0f);
                roi = cv::Rect(roix, roiy,
                        cx - roix + std::min((float)ceil(radius), (float) warpedImgRoi.cols - cx-1)+1,
                        cy - roiy + std::min((float)ceil(radius), (float) warpedImgRoi.rows - cy-1)+1);
                warpedImg = warpedImgRoi(roi);
                
                //Coordinates in cutted ROI
                cx = cx - roix;
                cy = cy - roiy;
            } else
                warpedImgRoi.copyTo(warpedImg);
                
            
            //Integration Scale selection
            si = selIntegrationScale(warpedImg, si, cv::Point(cx, cy));

            //Differentation scale selection
            sd = selDifferentiationScale(warpedImg, Lxm2smooth, Lxmysmooth, Lym2smooth, si,
                    cv::Point(cx, cy));

            if(Lxm2smooth.rows == 0 || Lxmysmooth.rows == 0 || Lym2smooth.rows == 0) // WRONG??
		        {
            	divergence = true;
            	continue;
		        }
            //Spatial Localization
            cxPr = cx; //Previous iteration point in normalized window
            cyPr = cy;

            float cornMax = 0;
            for (int j = 0; j < 3; j++)
            {
                for (int t = 0; t < 3; t++)
                {
                    float dx2 = Lxm2smooth.at<float> (cyPr - 1 + j, cxPr - 1 + t);
                    float dy2 = Lym2smooth.at<float> (cyPr - 1 + j, cxPr - 1 + t);
                    float dxy = Lxmysmooth.at<float> (cyPr - 1 + j, cxPr - 1 + t);
                    float det = dx2 * dy2 - dxy * dxy;
                    float tr = dx2 + dy2;
                    float cornerness = det - (0.04 * pow(tr, 2));
                    if (cornerness > cornMax)
                    {
                        cornMax = cornerness;
                        cx = cxPr - 1 + t;
                        cy = cyPr - 1 + j;
                    }
                }
            }
            
            //Transform point in image coordinates
            p(0, 0) = px;
            p(1, 0) = py;
            //Displacement vector
            c(0, 0) = cx - cxPr;
            c(1, 0) = cy - cyPr;
            //New interest point location in image
            p = p + U.inv() * c;
            px = p(0, 0);
            py = p(1, 0);
            
            q = calcSecondMomentSqrt(Lxm2smooth, Lxmysmooth, Lym2smooth, cv::Point(cx, cy), Mk);
/********** INSTEAD OF THE SECOND MOMENT MATRIX, Mk, WE CALL ISOTROPY AND ICS SCRIPT **********/

            float ratio = 1 - q;

            //if ratio == 1 means q == 0 and one axes equals to 0
            if (!isnan(ratio) && ratio != 1)
            {
                //Update U matrix
                U = U * Mk;

                cv::Mat uVal, uV;
                cv::eigen(U, uVal, uV);

                Qinv = normMaxEval(U, uVal, uV);
                
                
                //Keypoint doesn't converge
                if (Qinv >= 6)
					        divergence = true;

                                   
                //Keypoint converges
                else if (ratio <= 0.05)
                {
                    convergence = true;

                    //Set transformation matrix
                    transf.setTo(0);
                    cv::Mat col0 = transf.col(0);
                    U.col(0).copyTo(col0);
                    cv::Mat col1 = transf.col(1);
                    U.col(1).copyTo(col1);
                    keypoint.transf =cv::Mat(transf);
/*                    
                    for (int j = 0; j < 5; j++)
                    {
                      for (int t = 0; t < 5; t++)
                      {
                        float dx2 = Lxm2smooth.at<float> (cy - 1 + j, cx - 1 + t);
                        float dy2 = Lym2smooth.at<float> (cy - 1 + j, cx - 1 + t);
                        float dxy = Lxmysmooth.at<float> (cy - 1 + j, cx - 1 + t);
                        float det = dx2 * dy2 - dxy * dxy;
                        float tr = dx2 + dy2;
                        float cornerness = det - (0.04 * pow(tr, 2));
                        harrisPatch.at<float>(j, t) = cornerness;
                      }
                    }
                    cv::Mat hessian, gradient;
                    deriv_2D( harrisPatch, gradient, 2, 2 );
                    hessian_2D( harrisPatch, hessian, 2, 2 );
*/                    
                    ax1 = 1. / std::abs(uVal.at<float> (0, 0)) * 3 * si;
                    ax2 = 1. / std::abs(uVal.at<float> (1, 0)) * 3 * si;
                    phi = atan(uV.at<float> (1, 0) / uV.at<float> (0, 0)) * (180) / CV_PI;
                    keypoint.axes = cv::Size_<float> (ax1, ax2);
                    keypoint.phi = phi;
                    keypoint.centre = cv::Point2f(px, py);
                    keypoint.si = si;
                    keypoint.size = 2 * 3 * si;
                    
                } else
                    radius = 3 * si * 1.4;

            } else divergence = true;
            
        } else divergence = true;
            
        ++i;
    }

    return divergence;
}

/*
 * Selects the integration scale that maximize LoG in point c
 */
float selIntegrationScale(const cv::Mat & image, float si, cv::Point c)
{
    cv::Mat Lap, L;
    int cx = c.x;
    int cy = c.y;
    float maxLap = 0;
    float maxsx = si;
    int gsize;
    float sigma, sigma_prev = 0;

    image.copyTo(L);
    /* Search best integration scale between previous and successive layer
     */
    for (float u = 0.7; u <= 1.41; u += 0.1)
    {
        float sik = u * si;
        sigma = sqrt(powf(sik, 2) - powf(sigma_prev, 2));

        gsize = ceil(sigma * 3) * 2 + 1;

        cv::GaussianBlur(L, L, cv::Size(gsize, gsize), sigma);
        sigma_prev = sik;

        cv::Laplacian(L, Lap, CV_32F, 3);

        float lapVal = sik * sik * std::abs(Lap.at<float> (cy, cx));

        if (u == 0.7)
            maxLap = lapVal;

        if (lapVal >= maxLap)
        {
            maxLap = lapVal;
            maxsx = sik;

        }

    }
    return maxsx;
}

/*
 * Calculates second moments matrix square root
 */
float calcSecondMomentSqrt(const cv::Mat & dx2, const cv::Mat & dxy, const cv::Mat & dy2, cv::Point p, cv::Mat & Mk)
{
    cv::Mat M, V, eigVal, Vinv, D;

    calcSecondMomentMatrix(dx2, dxy, dy2, p, M);

    /* *
     * M = V * D * V.inv()
     * V has eigenvectors as columns
     * D is a diagonalcv::Matrix with eigenvalues as elements
     * V.inv() is the inverse of V
     * */

    cv::eigen(M, eigVal, V);
//    V = V.t(); // WRONG??
    Vinv = V.inv();

    float eval1 = eigVal.at<float> (0, 0) = sqrt(eigVal.at<float> (0, 0));
    float eval2 = eigVal.at<float> (1, 0) = sqrt(eigVal.at<float> (1, 0));

    D = cv::Mat::diag(eigVal);

    //square root of M
    Mk = V * D * Vinv;
    //return q isotropic measure
    return std::min(eval1, eval2) / std::max(eval1, eval2);
}

float normMaxEval(cv::Mat & U, cv::Mat & uVal, cv::Mat & uVec)
{
    /* *
     * Decomposition:
     * U = V * D * V.inv()
     * V has eigenvectors as columns
     * D is a diagonalcv::Matrix with eigenvalues as elements
     * V.inv() is the inverse of V
     * */
//    uVec = uVec.t(); // WRONG??
    cv::Mat uVinv = uVec.inv();

    //Normalize min eigenvalue to 1 to expand patch in the direction of min eigenvalue of U.inv()
    double uval1 = uVal.at<float> (0, 0);
    double uval2 = uVal.at<float> (1, 0);

    if (std::abs(uval1) < std::abs(uval2))
    {
        uVal.at<float> (0, 0) = 1;
        uVal.at<float> (1, 0) = uval2 / uval1;

    } else
    {
        uVal.at<float> (1, 0) = 1;
        uVal.at<float> (0, 0) = uval1 / uval2;

    }

    cv::Mat D = cv::Mat::diag(uVal);
    //U normalized
    U = uVec * D * uVinv;

    return std::max(std::abs(uVal.at<float> (0, 0)), std::abs(uVal.at<float> (1, 0))) / std::min(
            std::abs(uVal.at<float> (0, 0)), std::abs(uVal.at<float> (1, 0))); //define the direction of warping
}

/*
 * Selects diffrentiation scale
 */
float selDifferentiationScale(const cv::Mat & img, cv::Mat & Lxm2smooth, cv::Mat & Lxmysmooth,
        cv::Mat & Lym2smooth, float si, cv::Point c)
{
	bool showImage =false;
    float s = 0.5;
    float sdk = s * si;
    float sigma_prev = 0, sigma;

    cv::Mat L, dx2, dxy, dy2;

    double qMax = 0;

    //Gaussian kernel size
    int gsize;
    cv::Size ksize;

    img.copyTo(L);

    if(showImage)
    {
    	 cv::imshow ("win", img * 255);
    	 cv::waitKey() ;
    }
    while (s <= 0.751)
    {
        cv::Mat M;
        float sd = s * si;

        //Smooth previous smoothed image L
        sigma = sqrt(powf(sd, 2) - powf(sigma_prev, 2));

        gsize = ceil(sigma * 3) * 2 + 1;

        cv::GaussianBlur(L, L, cv::Size(gsize, gsize), sigma);

        sigma_prev = sd;

        //X and Y derivatives
        cv::Mat Lx, Ly;
        cv::Sobel(L, Lx, L.depth(), 1, 0, 1);
        Lx = Lx * sd;
        cv::Sobel(L, Ly, L.depth(), 0, 1, 1);
        Ly = Ly * sd;

        //Size of gaussian kernel
        gsize = ceil(si * 3) * 2 + 1;
        ksize = cv::Size(gsize, gsize);

        cv::Mat Lxm2 = Lx.mul(Lx);
        cv::GaussianBlur(Lxm2, dx2, ksize, si);

        cv::Mat Lym2 = Ly.mul(Ly);
        cv::GaussianBlur(Lym2, dy2, ksize, si);

        cv::Mat Lxmy = Lx.mul(Ly);
        cv::GaussianBlur(Lxmy, dxy, ksize, si);

        calcSecondMomentMatrix(dx2, dxy, dy2, cv::Point(c.x, c.y), M);

        //calc eigenvalues
        cv::Mat eval;
        cv::eigen(M, eval);
        double eval1 = std::abs(eval.at<float> (0, 0));
        double eval2 = std::abs(eval.at<float> (1, 0));
        double q = std::min(eval1, eval2) / std::max(eval1, eval2);

        if (q >= qMax)
        {
            qMax = q;
            sdk = sd;
            dx2.copyTo(Lxm2smooth);
            dxy.copyTo(Lxmysmooth);
            dy2.copyTo(Lym2smooth);

        }

        s += 0.05;
    }

    return sdk;
}
