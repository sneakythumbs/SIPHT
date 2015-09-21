#include "Affine_Adaptation.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdio.h>

namespace pk
{

  Affine_Adaptation::Affine_Adaptation(int type)
  {
    this->method = type;
  }
  
  Affine_Adaptation::Affine_Adaptation(int type, std::string scriptLocation, std::string outputLocation)
  {
    this-> method = type;
    icsScriptLocation = scriptLocation;
    tempLocation = outputLocation;
  }
/*
 * Conversion between cv::KeyPoint and Elliptic_KeyPoint
 */
  void Affine_Adaptation::keyPointToElliptic(cv::KeyPoint & keypoint, Elliptic_KeyPoint & ellipticKeypoint)
  {
    ellipticKeypoint.centre.x = keypoint.pt.x;
    ellipticKeypoint.centre.y = keypoint.pt.y;
    ellipticKeypoint.axes = cv::Size(keypoint.size*3, keypoint.size*3);

    ellipticKeypoint.phi = keypoint.angle;
    ellipticKeypoint.size = keypoint.size * 2 * 3;
    ellipticKeypoint.si = keypoint.size;
  }

  void Affine_Adaptation::ellipticToKeyPoint(Elliptic_KeyPoint & ellipticKeypoint, cv::KeyPoint & keypoint)
  {
    keypoint.pt = ellipticKeypoint.centre;
    keypoint.angle = ellipticKeypoint.phi;
    keypoint.size = ellipticKeypoint.size;
  }

/*
 * Calculates second moments matrix in point p
 */
  void Affine_Adaptation::calcSecondMomentMatrix(const cv::Mat & dx2, const cv::Mat & dxy, const cv::Mat & dy2, cv::Point p, cv::Mat & M)
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
  bool Affine_Adaptation::calcAffineAdaptation(const cv::Mat & fimage, Elliptic_KeyPoint & keypoint, int iterations)
  {
    if (0 == method) return false;
    
    cv::Mat_<float> transf(2, 3)/*Trasformation matrix*/,
            size(2, 1)/*Image size after transformation*/, 
            c(2, 1)/*Transformed point*/, 
            p(2, 1) /*Image point*/,
            diff(2,1) /*update to image point*/;
            
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
    while (i <= 10 && !divergence && !convergence)
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
            diff = U.inv() * c;
//            p = p + U.inv() * c;
            p = p + diff;
            px = p(0, 0);
            py = p(1, 0);
            
            float ratio = 0;
            if (1 == method)
            {
              q = calcSecondMomentSqrt(Lxm2smooth, Lxmysmooth, Lym2smooth, cv::Point(cx, cy), Mk);
            }
            else if (2 == method)
            {
              q = calculateUnmixingMatrix( warpedImg, Mk, si, sd);
            }
            
            ratio = 1 - q;
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
//                else if (ratio <= 0.05)
                else if (cv::norm(diff) < 1.5)
                {
                    convergence = true;

                    //Set transformation matrix
                    transf.setTo(0);
                    cv::Mat col0 = transf.col(0);
                    U.col(0).copyTo(col0);
                    cv::Mat col1 = transf.col(1);
                    U.col(1).copyTo(col1);
                    keypoint.transf =cv::Mat(transf);
                  
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
            
            //////HERE
        } else divergence = true;
            
        ++i;
    }

    return divergence; //TODO At the moment the majority of points are not converging before running out of iterations (this still returns a false, and the original point is considered valid)
  }

/*
 * Selects the integration scale that maximize LoG in point c
 */
  float Affine_Adaptation::selIntegrationScale(const cv::Mat & image, float si, cv::Point c)
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
  float Affine_Adaptation::calcSecondMomentSqrt(const cv::Mat & dx2, const cv::Mat & dxy, const cv::Mat & dy2, cv::Point p, cv::Mat & Mk)
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

  float Affine_Adaptation::normMaxEval(cv::Mat & U, cv::Mat & uVal, cv::Mat & uVec)
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
  float Affine_Adaptation::selDifferentiationScale(const cv::Mat & img, cv::Mat & Lxm2smooth, cv::Mat & Lxmysmooth,
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

/*
  Computes the partial derivatives in x, and y of a pixel

  @param img matrix of pixels
  @param row pixel's image row
  @param col pixel's image column

  @return Returns the vector of partial derivatives for pixel I
    { dI/dx, dI/dy }^T as a CvMat&
*/
  void Affine_Adaptation::deriv_2D(const cv::Mat& img, cv::Mat& dI, int row, int col)
  {
    double dx, dy;
    
    dx = ( img.at<float>(row,col+1) -
           img.at<float>(row,col-1) ) / 2.0;
    dy = ( img.at<float>(row+1,col) -
           img.at<float>(row-1,col) ) / 2.0;
    
    dI = cv::Mat( 2, 1, CV_64FC1 );
    dI.at<double>(0,0) = dx;
    dI.at<double>(1,0) = dy;
  
    return;
  }

/*
  Computes the 2D Hessian matrix for a pixel in an image.
  
  @param img matrix of pixels
  @param row pixel's image row
  @param col pixel's image column
  
  @return Returns the Hessian matrix (below) for pixel I as a CvMat&
  
  / Ixx  Ixy \ <BR>
  \ Ixy  Iyy /
*/
  void Affine_Adaptation::hessian_2D( const cv::Mat& img, cv::Mat& H, int row, int col)
  {
    double val, dxx, dyy, dxy;
    
    val = img.at<float>(row,col);
    dxx = ( img.at<float>(row,col+1) +
            img.at<float>(row,col-1) - 2*val );
    dyy = ( img.at<float>(row+1,col) +
            img.at<float>(row-1,col) - 2*val );
    dxy = ( img.at<float>(row+1,col+1) -
            img.at<float>(row+1,col-1) -
            img.at<float>(row-1,col+1) + 
            img.at<float>(row-1,col-1) ) / 4.0;
              
    H = cv::Mat(2,2,CV_64FC1);
    H.at<double>(0,0) = dxx;
    H.at<double>(0,1) = dxy;
    H.at<double>(1,0) = dxy;
    H.at<double>(1,1) = dyy;
      
    return;
  
  }
  
  float Affine_Adaptation::executeICSscript(cv::Mat &ics)
  {

    std::string programPath = icsScriptLocation;
    programPath.append("/ics_single.R");
    programPath.append(" ");
    programPath.append(tempLocation);
//    programPath.append(" 2> ./log");
//    programPath.append(" 2> /dev/null");
//    programPath.append("  >NUL 2>NUL");

    system( programPath.c_str() );
        
    std::string resPath = tempLocation;

    resPath.append("/tmpMat.txt");
  
    std::ifstream in;
    while (in.is_open() != true )
      in.open(resPath.c_str());
    double value;
    for(int r=0; r<2; r++)
      for(int c=0; c<2; c++)
      {
        in >> value;
        ics.at<double>(r,c) = value;
      }
    
    in.close();
    resPath.insert(0, "rm ");
    system( resPath.c_str() );

    ics = ics.t();
    normalizeTransformationMatrix(ics, ics);

    //normalize ics matrices to det =1
    double detICS =  ics.at<double>(0,0)*ics.at<double>(1,1) - ics.at<double>(0,1)*ics.at<double>(1,0);
    ics /= sqrt(fabs(detICS));
    
    // return condition number of the normalised ics matrix
    cv::Mat lambda;
    cv::eigen(ics, lambda);;
    float eigVal1 = std::abs(lambda.at<float> (0, 0));
    float eigVal2 = std::abs(lambda.at<float> (1, 0));
    float q = std::min(eigVal1, eigVal2) / std::max(eigVal1, eigVal2);
    return q;
    
  }

  void Affine_Adaptation::normalizeTransformationMatrix(cv::Mat &src, cv::Mat &dst)
  {
    dst = src.clone();
    svdInv(dst);
//  Mat trMatCopy1 = trMat;
    double tr00 = dst.at<double>(0,0),
           tr01 = dst.at<double>(0,1),
           tr10 = dst.at<double>(1,0),
           tr11 = dst.at<double>(1,1);

    double l1 = sqrt(tr00*tr00 + tr10*tr10),
           l2 = sqrt(tr01*tr01 + tr11*tr11);

    double det = sqrt(fabs((dst.at<double>(0,0) * dst.at<double>(1,1) - dst.at<double>(0,1) * dst.at<double>(1,0))));
    cv::Mat normalizedTr = dst/det;
////  normalizedTr = normalizedTr.inv();
  // Normalize first column vector to 0 degree
    double angle = -atan2(dst.at<double>(1,0), dst.at<double>(0,0));
    cv::Mat rotationMatrix = (cv::Mat_<double>(2,2) << cos(angle), -sin(angle), sin(angle), cos(angle));
    normalizedTr = rotationMatrix * normalizedTr;

    cv::Mat flip = ( cv::Mat_<double>(2,2) << 1, 0, 0, 1);
    if(normalizedTr.at<double>(0,0) * normalizedTr.at<double>(1,1) < 0)
    {
      flip.at<double>(1,1) *= -1;
      flip.at<double>(0,1) *= -1;
    }  

    normalizedTr = flip * normalizedTr;

    det = sqrt(fabs((normalizedTr.at<double>(0,0) * normalizedTr.at<double>(1,1) - normalizedTr.at<double>(0,1) * normalizedTr.at<double>(1,0))));
    dst.at<double>(0,0) = normalizedTr.at<double>(0,0) / det;
    dst.at<double>(1,0) = normalizedTr.at<double>(1,0) / det;
    dst.at<double>(0,1) = normalizedTr.at<double>(0,1) / det;
    dst.at<double>(1,1) = normalizedTr.at<double>(1,1) / det;

    
  }


  void Affine_Adaptation::svdInv(cv::Mat &inp)
  {
    cv::Mat w, u, vt;
    cv::SVDecomp(inp, w, u , vt);

    cv::Mat W = (cv::Mat_<double>(2,2) << 1/sqrt(w.at<double>(0,0)), 0, 0, 1/sqrt(w.at<double>(0,1)));
    // SHould This be square root?
    inp = u* W * vt;
  }

  void Affine_Adaptation::calculateImageGradients( cv::Mat& img, double integrationScale, std::vector<cv::Point2d>& gradients )
  {
    cv::Point2d  grad;

      int r_i = img.rows / 2;
      int c_i = img.cols / 2;

      for( int i_i = -r_i; i_i < r_i; ++i_i )
        for( int j_i = -c_i; j_i < c_i; ++j_i )
        {
          double dx_d, dy_d;

            dx_d = img.at<float>( r_i + i_i, c_i + j_i) - img.at<float>( r_i + i_i, c_i + j_i + 1);
            dy_d = img.at<float>( r_i + i_i, c_i + j_i) - img.at<float>( r_i + i_i + 1, c_i + j_i);

            double X_d = (j_i - c_i);
            double Y_d = (i_i - r_i);
            double weight_d  = exp( -(X_d*X_d + Y_d*Y_d) / (2*integrationScale * integrationScale));

            grad.x = dx_d;// * weight_d;
            grad.y = dy_d;// * weight_d;

            double magnitude = sqrt( grad.x* grad.x + grad.y*grad.y);
            double phase = atan2(grad.y, grad.x);
            if(magnitude > 0.01)
            {
//              grad.x *= weight_d;
//              grad.y *= weight_d;

              gradients.push_back(grad);
            }
          
      }
  }





  void Affine_Adaptation::writeOutGradients(std::vector<cv::Point2d>& gradients)
  {

    std::string resPath = tempLocation;
    resPath.append("/gradients.txt");
  
    std::ofstream out;
    while (out.is_open() != true )
      out.open(resPath.c_str(), std::ios::out | std::ios::trunc);
    for (auto grad : gradients)
      out << grad.x << " " << grad.y << std::endl;
    out.close();
  }
  
  float Affine_Adaptation::calculateUnmixingMatrix( cv::Mat& warpedImagePatch, cv::Mat& ics, double integrationScale, double derivativeScale)
  {
    std::vector<cv::Point2d> gradients;
    cv::Mat smoothedPatch;
    ics = cv::Mat( 2, 2, CV_64FC1 );

    int gsize = ceil(derivativeScale * 3) * 2 + 1;
    cv::GaussianBlur(warpedImagePatch, smoothedPatch, cv::Size(gsize, gsize), derivativeScale);
    calculateImageGradients( smoothedPatch, integrationScale, gradients );
    if (gradients.size() < 2)
      return 0;
    
//    writeOutGradients(gradients);
//    float q = executeICSscript(ics);
    ICS icsCalculator;
    icsCalculator.calculateNormalizationMat(gradients, ics);
    normalizeTransformationMatrix(ics, ics);   
    double detICS = ics.at<double>(0,0)*ics.at<double>(1,1) - ics.at<double>(0,1)*ics.at<double>(1,0);
    ics /= sqrt(fabs(detICS));

    ics.convertTo(ics, CV_32FC1);
    int q = 1;    
    return q;
  }

} /* End namespace pk */  
