//
// Created by bruno on 8/28/20.
//
#include "matching.h"
#include <opencv2/opencv.hpp>

using namespace cv;

void test_params_save(const std::string& config_file, const std::string& image_file, const std::string& pyr_file)
{
    MatchingParams params(0, 360, 1, 1, 1, 1, {4, 8});
    Matching matching(params, 100, 30.f, 60.f, 8);
    matching.writeMatchingParams(config_file);


    Mat img = imread(image_file);
    assert(!img.empty() && "check your img path");

    Rect roi(130, 110, 270, 270);
    img = img(roi).clone();
    Mat mask = Mat(img.size(), CV_8UC1, {255});

    // padding to avoid rotating out
    int padding = 100;
    cv::Mat padded_img = cv::Mat(img.rows + 2*padding, img.cols + 2*padding, img.type(), cv::Scalar::all(0));
    img.copyTo(padded_img(Rect(padding, padding, img.cols, img.rows)));

    cv::Mat padded_mask = cv::Mat(mask.rows + 2*padding, mask.cols + 2*padding, mask.type(), cv::Scalar::all(0));
    mask.copyTo(padded_mask(Rect(padding, padding, img.cols, img.rows)));

    matching.addClassPyramid(padded_img, padded_mask, "test");

    matching.writeClassPyramid(pyr_file, "test");
}

void test_params_read(const std::string& config_file, const std::string& pyr_file)
{
    Matching new_matching = Matching::readMatchingParams(config_file);

    new_matching.readClassPyramid(pyr_file, "test");

    std::cout << "OK" << std::endl;
}

int main()
{
    std::string config_file = "../../test_file/matching.yaml";
    std::string pyr_file = "../../test_file/train.yaml";

    test_params_save(config_file, "../../img/train.png", pyr_file);
    test_params_read(config_file, pyr_file);


}