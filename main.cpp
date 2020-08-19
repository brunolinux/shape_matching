#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/matching.h"
#include "src/cuda_icp/scene/edge_scene/edge_scene.h"
#include "src/cuda_icp/icp.h"

using namespace cv;

void test()
{
    MatchingParams params(0, 360, 1, 1, 1, 1, {4, 8});
    Matching matching(params);

    {
        Mat img = imread("img/train.png");
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
        std::cout << matches[i].x << ", " << matches[i].y << "\t" << matches[i].angle
                  << "\t" << matches[i].similarity << std::endl;
    }


/*    // construct scene
    Scene_edge scene;
    // buffer
    std::vector<::Vec2f> pcd_buffer, normal_buffer;
    scene.init_Scene_edge_cpu(img, pcd_buffer, normal_buffer);

    cuda_icp::RegistrationResult result = cuda_icp::ICP2D_Point2Plane_cpu(model_pcd, scene);*/
}

int main() {
    test();
    return 0;
}
