//
// Created by bruno on 8/15/20.
//
#include "../feature.h"
#include "../pyramid.h"
#include <opencv2/opencv.hpp>

void test_feature(const cv::Mat& src)
{
    std::vector<Feature> features;
    createFeatures(src, cv::Mat(), features, 30.f, 60.f, 8, 120);

    //cv::Mat featureMat = cv::Mat::zeros(src.size(), CV_8U);
    cv::Mat src_draw = src.clone();
    for (int i = 0; i < features.size(); i ++) {
        cv::circle(src_draw, cv::Point(features[i].x, features[i].y), 4, 122, 2);
        //src.at<uchar>() = 122;
    }
    cv::imshow("Feature location", src_draw);
    cv::waitKey(0);
}


void test_pyramid(const cv::Mat& src)
{
    PyrDetectorParams params;
    PyramidDetector detector(params);
    auto pyramid = detector.detect(src, cv::Mat());

    cv::Mat src_pyr = src.clone();
    for (int i = 0; i < pyramid.size(); i ++) {
        cv::Mat src_draw = src_pyr.clone();
        auto pattern = pyramid[i];
        for (int i = 0; i < pattern.m_features.size(); i ++) {
            cv::circle(src_draw, cv::Point(pattern.m_features[i].x, pattern.m_features[i].y), 4, 122, 2);
        }
        cv::rectangle(src_draw, cv::Point(pattern.base_x, pattern.base_y),
                      cv::Point(pattern.base_x + pattern.width, pattern.base_y + pattern.height), cv::Scalar(122));

        cv::Mat tmp;
        cv::resize(src_pyr, tmp,src_pyr.size()/2);
        src_pyr = tmp;

        cv::imshow("py" + std::to_string(i), src_draw);
        cv::waitKey(0);
    }
}

int main()
{
    cv::Mat src = cv::imread("img/train.png", 0);

    cv::Rect roi(130, 110, 270, 270);
    src = src(roi).clone();

    test_feature(src);
    test_pyramid(src);
}