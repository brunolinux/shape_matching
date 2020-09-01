#include <iostream>
#include <opencv2/opencv.hpp>
#include "main_util.h"

using namespace cv;

void test()
{
    std::vector<int> T_vec{4, 8};
    int py_level = T_vec.size();
    MatchingParams params(270, 360, 0.5, 1, 1, 1, T_vec);
    Matching matching(params, 2000);

    {
        Mat img = imread("img/circle_train.bmp");
        assert(!img.empty() && "check your img path");

        Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
        cv::circle(mask, cv::Point(1839, 1311), 1260, cv::Scalar(255), -1);
        cv::circle(mask, cv::Point(1839, 1311), 1190, cv::Scalar(0), -1);


        // padding to avoid rotating out
        int padding = 100;
        cv::Mat padded_img, padded_mask;
        cv::copyMakeBorder(img, padded_img, padding, padding, padding, padding, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        cv::copyMakeBorder(mask, padded_mask, padding, padding, padding, padding, cv::BORDER_CONSTANT, cv::Scalar::all(0));

        matching.addClassPyramid(padded_img, padded_mask, "test");
    }

    for (int i = 0; i < 17; i++) {
        Mat test_img = imread("img/circle_rotate_" + std::to_string(i) + ".bmp");
        assert(!test_img.empty() && "check your img path");

        cv::Mat padded_test_img = matching.createPaddedImage(test_img);

        std::vector<MatchingResult> matches = matching.matchClass(padded_test_img, "test", 90);

        // construct scene
        Scene_edge scene;
        // buffer
        std::vector<::Vec2f> pcd_buffer, normal_buffer;
        scene.init_Scene_edge_cpu(padded_test_img, pcd_buffer, normal_buffer);

        if (matches.size() > 0) {
            cv::Matx33f icp_mat = icpMatching(matching, scene, matches[0]);
            std::cout << "\n\n";
            std::cout << icp_mat << std::endl;
            double icp_diff_angle = -std::asin(icp_mat(1, 0))/CV_PI*180;
            std::cout << "angle: " << matches[0].angle + icp_diff_angle << "\n";

        }
    }
}

int main() {
    test();
    return 0;
}
