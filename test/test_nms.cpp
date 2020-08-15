//
// Created by bruno on 8/14/20.
//
#include "../pyramid.h"
#include <iostream>

int main()
{
    cv::Mat_<float> src(6, 8);
    src << 8., 0., 0., -2., -2., -2., -2., -2.,
           0., 1., 0., -2., -2., -2., -2., -2.,
           0., 0., 1., -2., -2., -1., -2., -2.,
           0., 0., 6., -2., -2., -2., -2., -2.,
           0., 0., 1., -2., -2., -2., -2., -2.,
           0., 0., 0., 0., 0., 0., 0., 0.;

    cv::Mat result = createNMSMat(src, cv::Mat(), 5);
    std::cout << "No mask: " << std::endl;
    std::cout << result;
    std::cout << std::endl;

    cv::Mat mask = cv::Mat::ones(src.size(), CV_8U);
    mask.at<uchar>(2, 5) = 0;
    result = createNMSMat(src, mask, 5);
    std::cout << "With mask: " << std::endl;
    std::cout << result;
    std::cout << std::endl;
    return 0;
}