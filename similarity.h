#ifndef SHAPE_MATCHING_SIMILARITY_H
#define SHAPE_MATCHING_SIMILARITY_H

#include <opencv2/core.hpp>

#include "pyramid.h"

void similarity(const std::vector<cv::Mat>& linear_memories,
    const Pattern& pattern,
    cv::Mat& dst, cv::Size img_size, int T);

void similarity_64(const std::vector<cv::Mat>& linear_memories,
    const Pattern& pattern,
    cv::Mat& dst, cv::Size img_size, int T);

#endif