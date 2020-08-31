#include "test_util.h"
#include <opencv2/opencv.hpp>


void draw_features(cv::Mat& src, const std::vector<Feature>& features)
{
    for (int i = 0; i < features.size(); i ++) {
        cv::circle(src, cv::Point(features[i].x, features[i].y), 4, cv::Scalar(0, 0, 255), 2);
        //src.at<uchar>() = 122;
    }
}

std::vector<cv::Mat> draw_pyramid(cv::Mat& src, const Pyramid& pyramid)
{
    std::vector<cv::Mat> dst_vec;
    for (int i = 0; i < pyramid.size(); i ++) {
        cv::Mat src_draw = src.clone();
        auto pattern = pyramid[i];
        for (int i = 0; i < pattern.m_features.size(); i ++) {
            int base_x = pattern.base_x;
            int base_y = pattern.base_y;

            cv::circle(src_draw, cv::Point(pattern.m_features[i].x + base_x, pattern.m_features[i].y + base_y),
                       4, cv::Scalar(0, 0, 255), 2);
        }
        cv::rectangle(src_draw, cv::Point(pattern.base_x, pattern.base_y),
                  cv::Point(pattern.base_x + pattern.width, pattern.base_y + pattern.height), cv::Scalar(0, 255, 0));

        cv::Mat tmp;
        cv::resize(src, tmp, src.size()/2);
        src = tmp;
        
        dst_vec.push_back(src_draw);
    }
    return dst_vec;
}