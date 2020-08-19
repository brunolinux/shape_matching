#ifndef _TEST_UTIL_H
#define _TEST_UTIL_H

#include "feature.h"
#include "pyramid.h"

void draw_features(cv::Mat& src, const std::vector<Feature>& features);

std::vector<cv::Mat> draw_pyramid(cv::Mat& src, const Pyramid& pyramid);

#endif
