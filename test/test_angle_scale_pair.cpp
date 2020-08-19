#include "matching.h"
#include "test_util.h"
#include <iostream>
#include <opencv2/opencv.hpp>


void printASPairs(const std::vector<float>& angle_vec, const std::vector<float>& scale_vec)
{
	for (auto s : scale_vec) {
		for (auto a : angle_vec) {
			std::cout << "angle: " << a << "\t scale: " << s << std::endl;
		}
	}
}

void test_pair()
{
	MatchingParams params;
	std::vector<float> angle_vec;
	std::vector<float> scale_vec;
	
	{
		std::cout << "[1]\n";
		params.angle_start = 0;
		params.angle_end = 0;
		params.angle_step = 1;

		params.scale_start = 1;
		params.scale_end = 1;
		params.scale_step = 0;

		params.createAngleScaleVec(angle_vec, scale_vec);
		printASPairs(angle_vec, scale_vec);
	}

	
	{
		std::cout << "[2]\n";
		params.angle_start = 0;
		params.angle_end = 10;
		params.angle_step = 1;

		params.scale_start = 1;
		params.scale_end = 1;
		params.scale_step = 0;

		params.createAngleScaleVec(angle_vec, scale_vec);
		printASPairs(angle_vec, scale_vec);
	}



	{
		std::cout << "[3]\n";
		params.angle_start = 0;
		params.angle_end = 0;
		params.angle_step = 1;

		params.scale_start = 1;
		params.scale_end = 0;
		params.scale_step = 0.1;

		params.createAngleScaleVec(angle_vec, scale_vec);
		printASPairs(angle_vec, scale_vec);
	}

	{
		std::cout << "[4]\n";
		params.angle_start = 0;
		params.angle_end = 4;
		params.angle_step = 1;

		params.scale_start = 1;
		params.scale_end = 0.5;
		params.scale_step = 0.1;

		params.createAngleScaleVec(angle_vec, scale_vec);
		printASPairs(angle_vec, scale_vec);
	}
}

void test_rotatePyr(const cv::Mat& src, float angle)
{
	PyrDetectorParams params;
	params.num_feature_toplevel = 63;
	PyramidDetector detector(params);


	// origin pyramid
	auto pyramid = detector.detect(src, cv::Mat());
	cv::Mat src_draw = src.clone();
	auto src_draw_vec0 = draw_pyramid(src_draw, pyramid);
	for (int i = 0; i < src_draw_vec0.size(); i++) {
		std::string name = "old_py" + std::to_string(i);
		cv::imshow(name, src_draw_vec0[i]);
		cv::imwrite("img/intermediate/" + name + ".png", src_draw_vec0[i]);
	}
	cv::waitKey(0);


	// rotated pyramid
	auto new_pyramid = pyramid.rotatePyramid(angle, params.angle_bin_number);

	// debug angle
    for (int i = 0; i < pyramid.at(0).m_features.size(); i++) {
        std::cout << "old: " << pyramid.at(0).m_features[i].angle;
        std::cout << "\t new: " << new_pyramid.at(0).m_features[i].angle;
        std::cout << std::endl;
    }

	cv::Mat src_rot;
	cv::Point2f center = pyramid.getPatternCenter(0);
	cv::Mat rot_mat = cv::getRotationMatrix2D(center, 360 - angle, 1.);
	cv::warpAffine(src, src_rot, rot_mat, src.size());
	
	auto src_draw_vec = draw_pyramid(src_rot, new_pyramid);

	for (int i = 0; i < src_draw_vec.size(); i++) {
		std::string name = "py" + std::to_string(i);
		cv::imshow(name, src_draw_vec[i]);
		cv::imwrite("img/intermediate/" + name + ".png", src_draw_vec[i]);
	}
	cv::waitKey(0);
}


int main()
{
	test_pair();


	/*
		cv::Mat src = cv::imread("img/circle_train.bmp", 0);
		cv::Rect roi(500, 15, 1000, 2000);
	*/
	cv::Mat src = cv::imread("img/train.png", 0);
	cv::Rect roi(130, 110, 270, 270);

	src = src(roi).clone();
	// angle 是顺时针角度
	test_rotatePyr(src, 18.f);

}
