#include <iostream>
#include <opencv2/opencv.hpp>
#include "matching.h"
#include "cuda_icp/scene/edge_scene/edge_scene.h"
#include "cuda_icp/icp.h"

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
        cv::Mat padded_img = cv::Mat(img.rows + 2*padding, img.cols + 2*padding, img.type(), cv::Scalar::all(0));
        img.copyTo(padded_img(Rect(padding, padding, img.cols, img.rows)));

        cv::Mat padded_mask = cv::Mat(mask.rows + 2*padding, mask.cols + 2*padding, mask.type(), cv::Scalar::all(0));
        mask.copyTo(padded_mask(Rect(padding, padding, img.cols, img.rows)));

        matching.addClassPyramid(padded_img, padded_mask, "test");
    }

    for (int i = 0; i < 17; i++) {
        Mat test_img = imread("img/circle_rotate_" + std::to_string(i) + ".bmp");
        assert(!test_img.empty() && "check your img path");

        int padding = 250;
        cv::Mat padded_test_img = cv::Mat(test_img.rows + 2*padding,
                                          test_img.cols + 2*padding, test_img.type(), cv::Scalar::all(0));
        test_img.copyTo(padded_test_img(Rect(padding, padding, test_img.cols, test_img.rows)));
        int stride = pow(4, py_level-2) * 16;
        int n = padded_test_img.rows/stride;
        int m = padded_test_img.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);
        Mat img = padded_test_img(roi).clone();
        assert(img.isContinuous());

        std::vector<MatchingResult> matches = matching.matchClass(img, "test", 90);

        // construct scene
        Scene_edge scene;
        // buffer
        std::vector<::Vec2f> pcd_buffer, normal_buffer;
        scene.init_Scene_edge_cpu(img, pcd_buffer, normal_buffer);

        if (matches.size() > 0) {
            const MatchingResult& match = matches[0];
            Pyramid matchedPyr = matching.getClassPyramid(match);
            const auto& patternLevel0 = matchedPyr[0];
            std::vector<::Vec2f> model_pcd(patternLevel0.m_features.size());
            for(int i=0; i<patternLevel0.m_features.size(); i++){
                auto& feat = patternLevel0.m_features[i];
                model_pcd[i] = {
                        float(feat.x + match.x),
                        float(feat.y + match.y)
                };
            }
            cuda_icp::RegistrationResult result = cuda_icp::ICP2D_Point2Plane_cpu(model_pcd, scene);

            std::cout << "\n\n\n\n";
            std::cout << result.transformation_ << std::endl;
            double icp_diff_angle = -std::asin(result.transformation_[1][0])/CV_PI*180;
            std::cout << icp_diff_angle << std::endl;
        }
    }
}

int main() {
    test();
    return 0;
}
