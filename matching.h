//
// Created by bruno on 8/15/20.
//

#ifndef SHAPE_MATCHING_MATCHING_H
#define SHAPE_MATCHING_MATCHING_H

#include <map>

#include "pyramid.h"

struct AS_pair {
    AS_pair(float _angle, float _scale)
    :angle(_angle), scale(_scale) {}

    float angle;
    float scale;
};

struct MatchingParams
{
    MatchingParams();
    MatchingParams(float _angle_start, float _angle_end, float _angle_step,
                  float _scale_start, float _scale_end, float _scale_step,
                  std::vector<int>  _T_vec);

    std::vector<AS_pair> createASPairs();

    float angle_start;
    float angle_end;
    float angle_step;
    float scale_start;
    float scale_end;
    float scale_step;

    std::vector<int> T_vec;
};

class Matching {
public:
    Matching(const MatchingParams &params = MatchingParams());

    void addClassPyramid(const cv::Mat& src, const cv::Mat& mask, const std::string& class_id);

    void matchClass(const cv::Mat& src, const std::string& class_id);
    //void loadPyramidDetector();
private:
    std::map<std::string, std::vector<Pyramid>> m_classPyramids;

    MatchingParams m_params;
    PyramidDetector *m_detector;
};

#endif //SHAPE_MATCHING_MATCHING_H
