#include "../similarity.h"
#include "../pyramid.h"
#include "../response_map.h"

#include <opencv2/opencv.hpp>

int main()
{
	std::vector<int> T_vec{ 4, 8 };

	cv::Mat src = cv::imread("img/train.png", 0);

	cv::Rect roi(130, 110, 270, 270);
	src = src(roi).clone();

	PyrDetectorParams params;
	PyramidDetector detector(params);
	auto pyramid = detector.detect(src, cv::Mat());

	cv::Mat test = cv::imread("img/test.png", 0);
	cv::Rect roi_test(0, 0, 1600, 1120);
	test = test(roi_test).clone();


	LinearMemoryPyramid linearMemPyr = createLinearMemoryPyramid(test, cv::Mat(), T_vec,
		params.weak_threshold, params.angle_bin_number);

	
	cv::Mat similarities;
	//similarity(linearMemPyr[0], pyramid[0], similarities, test.size(), T_vec[0]);
}