#include <iostream>
#include <opencv2/opencv.hpp>
#include "main_util.h"

using namespace cv;

void test()
{
    MatchingParams params(0, 360, 1, 1, 1, 1, {4, 8});
    Matching matching(params);

    cv::Mat padded_img, padded_mask;
    {
        Mat img = imread("img/train.png");
        assert(!img.empty() && "check your img path");

        Rect roi(130, 110, 270, 270);
        img = img(roi).clone();
        Mat mask = Mat(img.size(), CV_8UC1, {255});

        // padding to avoid rotating out
        int padding = 100;

        cv::copyMakeBorder(img, padded_img, padding, padding, padding, padding, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        cv::copyMakeBorder(mask, padded_mask, padding, padding, padding, padding, cv::BORDER_CONSTANT, cv::Scalar::all(0));

        matching.addClassPyramid(padded_img, padded_mask, "test");
    }

    Mat test_img = imread("img/test1.png");
    Mat padded_test_img = matching.createPaddedImage(test_img);

    // without nms
    {
        std::vector<MatchingResult> matches = matching.matchClass(padded_test_img, "test", 90);

        for (int i = 0; i < matches.size(); i ++) {
            std::cout << "(" << matches[i].x << ", " << matches[i].y << ")\tangle:" << matches[i].angle
                      << "\tscore:" << matches[i].similarity << std::endl;
        }
    }


    std::cout << "After NMS removal" << std::endl;
    // nms remove
    std::vector<MatchingResult> new_matches = matching.matchClassWithNMS(padded_test_img, "test", 90, 0.5);

    for (int i = 0; i < new_matches.size(); i ++) {
        std::cout << "(" << new_matches[i].x << ", " << new_matches[i].y << ")\tangle:" << new_matches[i].angle
                  << "\tscore:" << new_matches[i].similarity << std::endl;
    }

    CV_Assert(new_matches.size() == 1);
    cv::Matx33f trans_mat = matching.getMatchingMatrix(new_matches[0]);

    cv::Mat padded_img_trans;
    cv::warpPerspective(padded_img, padded_img_trans, trans_mat, padded_img.size()*2);
    cv::imshow("test", padded_test_img);
    cv::imshow("without icp matching", padded_img_trans);
    cv::waitKey(0);

    // construct scene
    Scene_edge scene;
    // buffer
    std::vector<::Vec2f> pcd_buffer, normal_buffer;
    scene.init_Scene_edge_cpu(padded_test_img, pcd_buffer, normal_buffer);

    cv::Matx33f icp_mat = icpMatching(matching, scene, new_matches[0]);

    float angle = new_matches[0].angle - asin(icp_mat(1, 0))/CV_PI*180;
    std::cout << "angle: " << angle << std::endl;

    trans_mat = icp_mat * trans_mat;
    cv::warpPerspective(padded_img, padded_img_trans, trans_mat, padded_img.size()*2);
    cv::imshow("icp matching", padded_img_trans);
    cv::waitKey(0);
}

int main() {
    test();
    return 0;
}
