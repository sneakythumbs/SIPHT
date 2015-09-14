#include "Comparator.hpp"
#include "Hessian_Laplace.hpp"
#include "Harris_Laplace.hpp"
#include "Laplace.hpp"

namespace pk
{
    
  Comparator::Comparator(char* img, char* output, char* method, double xSc, double xSh, double ySh, double ySc)
  {
    filename = std::string(img);
    imageName = splitFilename(filename);
    outputPath = std::string(output);
    methodName = std::string(method);
    img1 = cv::imread( img, CV_LOAD_IMAGE_GRAYSCALE );
    double xScale = xSc;
    double xShear = tan ( xSh );
    double yShear = tan ( ySh );
    double yScale = 1/xScale;
    if ( ySc) yScale = ySc;
     
    matrix = "[" + std::to_string(xSc).substr(0,4) + "|" 
                 + std::to_string(xSh).substr(0,4) + "|" 
                 + std::to_string(ySh).substr(0,4) + "|" 
                 + std::to_string(yScale).substr(0,4) + "]";
    transform = (cv::Mat_<double>(2,3) << xScale, xShear, 0, yShear, yScale, 0);
      
    cv::Size dims;
    getSize(img1, transform, dims);
    warpAffine(img1, img2, transform, dims, CV_INTER_CUBIC, IPL_BORDER_CONSTANT);
    
    file.open(methodName + "-" + imageName + matrix, std::ios::out );
   
  }
  
  void Comparator::changeFile()
  {
    file.close();
    file.open(methodName + "-" + imageName + matrix, std::ios::out );
  }
  
  void Comparator::changeMethod(const char* method)
  {
    methodName = std::string(method);
    changeFile();
  }
    
  void Comparator::compare(std::vector<cv::KeyPoint>& img1Points, std::vector<cv::KeyPoint>& img2Points)
  {
    removeDuplicates(img1Points, 0.5);
    removeDuplicates(img2Points, 0.5);
    int orig = 0, warp = 0;

    for (auto&& i : img1Points)  ++orig;
    for (auto&& j : img2Points)  ++warp;
    
    std::vector<int> img1Mask(orig, 0);
    std::vector<int> img2Mask(warp, 0);    
    
    double xScale = transform.at<double>(0,0);
    double xShear = transform.at<double>(0,1);
    double yShear = transform.at<double>(1,0);
    double yScale = transform.at<double>(1,1);
    
    file << "# " << methodName << "\t" << imageName << "\t[" << xScale << "," 
         << xShear << ";" << yShear << "," << yScale << "]\n"
         << "# Transformation Norm: " << cv::norm(transform) << "\n"
         << "# unwarped points: " << orig << "\twarped points: " << warp
         << "\n# fraction found\t" << "fraction correct\t" << "second fraction\n";
    
    int count = 0;         
  { // Unrolling first loop iteration       
//    int count = 0;
    double tol = 0.01;
    for (int i = 0; i < img1Points.size(); ++i)
    {
      
      if (img1Mask[i]) continue;
         
      cv::Mat original = (cv::Mat_<double>(3,1) << img1Points[i].pt.x, img1Points[i].pt.y, 1);
      original = transform * original;                     
      for (int j = 0; j < img2Points.size(); ++j)
      {
        if (img2Mask[j]) continue;
        
        float min = std::min(img1Points[i].size, img2Points[j].size);
        if (fabs(img1Points[i].size - img2Points[j].size) > 0.2*min) continue;
        cv::Mat warped = (cv::Mat_<double>(2,1) << img2Points[j].pt.x, img2Points[j].pt.y);
        if (cv::norm(warped - original) < tol)
        {
          ++count;
          //std::cout << img1Points[i].size << "\t" << img2Points[j].size << std::endl;
          img1Mask[i] = j+1;
          img2Mask[j] = i+1;
          break;
        }
      }
    }
    std::cout << methodName << "\t" << imageName << std::endl;
    std::cout << "tolerance:\t" << tol << std::endl; 
    std::cout << "original method found:\t" << orig << std::endl;
    std::cout << "modified method found:\t" << warp << std::endl;
    std::cout << "number of matching points:\t" << count << std::endl;
    file  << tol << "\t" << double(warp)/orig << "\t" << double(count)/orig << "\t" << double(count)/warp << "\n";     
  }
    omp_set_num_threads(4);
//    # pragma omp parallel for ordered schedule(static,1)  
    for (int i = 1; i <= 30; ++i)
    {
      double tol = i * 1e-1;
//      int count = 0;

      for (int i = 0; i < img1Points.size(); ++i)
      {
        if (img1Mask[i]) continue;
           
        cv::Mat original = (cv::Mat_<double>(3,1) << img1Points[i].pt.x, img1Points[i].pt.y, 1);
        original = transform * original;                     
        for (int j = 0; j < img2Points.size(); ++j)
        {
          if (img2Mask[j]) continue;
          
          float min = std::min(img1Points[i].size, img2Points[j].size);
          if (fabs(img1Points[i].size - img2Points[j].size) > 0.2*min) continue;
          cv::Mat warped = (cv::Mat_<double>(2,1) << img2Points[j].pt.x, img2Points[j].pt.y);
          if (cv::norm(warped - original) < tol)
          {
            ++count;
            img1Mask[i] = j+1;
            img2Mask[j] = i+1;
            break;
          }
        }
      }
      std::cout << methodName << "\t" << imageName << std::endl;
      std::cout << "tolerance:\t" << tol << std::endl; 
      std::cout << "original method found:\t" << orig << std::endl;
      std::cout << "modified method found:\t" << warp << std::endl;
      std::cout << "number of matching points:\t" << count << std::endl;
//      # pragma omp ordered
      file  << tol << "\t" << double(warp)/orig << "\t" << double(count)/orig << "\t" << double(count)/warp << "\n";
    }
  }
        
    
  void Comparator::getSize(cv::Mat& img, cv::Mat& inTrans, cv::Size& size)
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
    std::cout << img.rows << " " << img.cols << std::endl;
    std::cout << rows << " " << cols << std::endl;
    size = cv::Size(cols,rows);
  }
  
  void Comparator::removeDuplicates(std::vector<cv::KeyPoint>& keypoints, double tol)
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
  
  std::string Comparator::splitFilename(const std::string& path)
  {
    return path.substr(path.find_last_of("/\\") + 1);
  }
  
} /* end namespace pk */

