//
// Created by bruno on 8/31/20.
//

#ifndef SHAPE_MATCHING_NMS_H
#define SHAPE_MATCHING_NMS_H

#include <opencv2/core.hpp>


void NMSBoxes(const std::vector<cv::Rect>& bboxes, const std::vector<float>& scores,
              const float score_threshold, const float nms_threshold,
              std::vector<int>& indices, const float eta=1, const int top_k=0);

#endif //SHAPE_MATCHING_NMS_H
