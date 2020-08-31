#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/matching.h"
#include "src/cuda_icp/scene/edge_scene/edge_scene.h"
#include "src/cuda_icp/icp.h"
#include "src/nms.h"

using namespace cv;

void test()
{
    MatchingParams params(0, 360, 1, 1, 1, 1, {4, 8});
    Matching matching(params);

    cv::Mat padded_img;
    {
        Mat img = imread("img/train.png");
        assert(!img.empty() && "check your img path");

        Rect roi(130, 110, 270, 270);
        img = img(roi).clone();
        Mat mask = Mat(img.size(), CV_8UC1, {255});

        // padding to avoid rotating out
        int padding = 100;
        padded_img = cv::Mat(img.rows + 2*padding, img.cols + 2*padding, img.type(), cv::Scalar::all(0));
        img.copyTo(padded_img(Rect(padding, padding, img.cols, img.rows)));

        cv::Mat padded_mask = cv::Mat(mask.rows + 2*padding, mask.cols + 2*padding, mask.type(), cv::Scalar::all(0));
        mask.copyTo(padded_mask(Rect(padding, padding, img.cols, img.rows)));

        matching.addClassPyramid(padded_img, padded_mask, "test");
    }

    Mat test_img = imread("img/test1.png");
    assert(!test_img.empty() && "check your img path");

    int padding = 250;
    cv::Mat padded_test_img = cv::Mat(test_img.rows + 2*padding,
                                 test_img.cols + 2*padding, test_img.type(), cv::Scalar::all(0));
    test_img.copyTo(padded_test_img(Rect(padding, padding, test_img.cols, test_img.rows)));
    int stride = 32;
    int n = padded_test_img.rows/stride;
    int m = padded_test_img.cols/stride;
    Rect roi(0, 0, stride*m , stride*n);
    Mat img = padded_test_img(roi).clone();
    assert(img.isContinuous());


    std::vector<MatchingResult> matches = matching.matchClass(img, "test", 90);

    for (int i = 0; i < matches.size(); i ++) {
        std::cout << "(" << matches[i].x << ", " << matches[i].y << ")\tangle:" << matches[i].angle
                  << "\tscore:" << matches[i].similarity << std::endl;
    }

    std::cout << "After NMS removal" << std::endl;
    // nms remove
    std::vector<MatchingResult> new_matches = matching.matchClassWithNMS(img, "test", 90, 0.5);

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
    scene.init_Scene_edge_cpu(img, pcd_buffer, normal_buffer);

    Pyramid matchedPyr = matching.getClassPyramid(new_matches[0]);
    const auto& patternLevel0 = matchedPyr[0];
    std::vector<::Vec2f> model_pcd(patternLevel0.m_features.size());
    for(int i = 0; i < patternLevel0.m_features.size(); i++){
        auto& feat = patternLevel0.m_features[i];
        model_pcd[i] = {
                float(feat.x + new_matches[0].x),
                float(feat.y + new_matches[0].y)
        };
    }
    cuda_icp::RegistrationResult result = cuda_icp::ICP2D_Point2Plane_cpu(model_pcd, scene);
    std::cout << result.transformation_ << std::endl;
    auto t = result.transformation_;
    cv::Matx33f icp_mat;
    icp_mat << t[0][0], t[0][1], t[0][2],
               t[1][0], t[1][1], t[1][2],
               t[2][0], t[2][1], t[2][2];

    float angle = new_matches[0].angle - asin(t[1][0])/CV_PI*180;
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
