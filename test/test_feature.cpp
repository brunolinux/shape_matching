//
// Created by bruno on 8/15/20.
//

#include "test_util.h"
#include <opencv2/opencv.hpp>

void test_feature(const cv::Mat& src)
{
    std::vector<Feature> features;
    createFeatures(src, cv::Mat(), features, 30.f, 60.f, 8, 120);

    //cv::Mat featureMat = cv::Mat::zeros(src.size(), CV_8U);
    cv::Mat src_draw;
    if (src.channels() == 1)
        cv::cvtColor(src, src_draw, CV_GRAY2BGR);
    else
        src_draw = src.clone();

    draw_features(src_draw, features);

    cv::imshow("Feature location", src_draw);
    cv::waitKey(0);
}


void test_pyramid(const cv::Mat& src)
{
    PyrDetectorParams params;
    PyramidDetector detector(params);
    auto pyramid = detector.detect(src, cv::Mat());


    cv::Mat src_pyr;
    if (src.channels() == 1)
        cv::cvtColor(src, src_pyr, CV_GRAY2BGR);
    else
        src_pyr = src.clone();

    auto src_draw_vec = draw_pyramid(src_pyr, pyramid);

    for (int i = 0; i < src_draw_vec.size(); i ++) {
        cv::imshow("py" + std::to_string(i), src_draw_vec[i]);
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