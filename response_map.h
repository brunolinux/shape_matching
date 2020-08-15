//
// Created by bruno on 8/15/20.
//

#ifndef SHAPE_MATCHING_RESPONSE_MAP_H
#define SHAPE_MATCHING_RESPONSE_MAP_H

#include <opencv2/core.hpp>

using LinearMemories = std::vector<cv::Mat> ;
using LinearMemoryPyramid = std::vector<LinearMemories>;


void spread(const cv::Mat &src, cv::Mat &dst, int T, int angle_bin_number=8);

void computeResponseMaps(const cv::Mat &src, std::vector<cv::Mat> &response_maps, int angle_bin_number = 8);

void linearize(const cv::Mat &response_map, cv::Mat &linearized, int T);
#endif //SHAPE_MATCHING_RESPONSE_MAP_H
