#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>
#include "pk.hpp"
#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <omp.h>

namespace pk
{

  class Comparator
  {
  
    public:
    cv::Mat img1;
    cv::Mat img2;
    cv::Mat transform;
    std::string imageName;
    std::string methodName;
    std::string filename;
    std::string outputPath;
    std::string matrix;
    std::ofstream file;
    
    Comparator(char* img, char* output, char* method, double xSc, double xSh, double ySh, double ySc = 0);
    void changeFile();
    void changeMethod(const char* method);
    void compare(std::vector<cv::KeyPoint>& img1Points, std::vector<cv::KeyPoint>& img2Points);
    
    protected:
    void getSize(cv::Mat& img, cv::Mat& inTrans, cv::Size& size);
    void removeDuplicates(std::vector<cv::KeyPoint>& keypoints, double tol);
    
    private:
    std::string splitFilename(const std::string& path);
  
  };
} /* end namespace pk */
