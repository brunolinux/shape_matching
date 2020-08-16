//
// Created by bruno on 8/15/20.
//

#ifndef SHAPE_MATCHING_MATCHING_H
#define SHAPE_MATCHING_MATCHING_H

#include <map>

#include "pyramid.h"

struct MatchingParams
{
    MatchingParams();
    MatchingParams(float _angle_start, float _angle_end, float _angle_step,
                  float _scale_start, float _scale_end, float _scale_step,
                  std::vector<int>  _T_vec);

    void createAngleScaleVec(std::vector<float>& angle_vec, std::vector<float>& scale_vec);

    float angle_start;
    float angle_end;
    float angle_step;
    float scale_start;
    float scale_end;
    float scale_step;

    std::vector<int> T_vec;
};

struct MatchingResult
{
    MatchingResult(int _x, int _y, float _sim, float _scale, float _angle, const std::string& _id)
        :x(_x), y(_y), similarity(_sim), scale(_scale), angle(_angle), class_id(_id)
    {}

    int x;
    int y;
    float similarity;
    float scale;
    float angle;
    std::string class_id;
};


class Matching {
public:
    Matching(const MatchingParams &params = MatchingParams());

    void addClassPyramid(const cv::Mat& src, const cv::Mat& mask, const std::string& class_id);

    void Matching::matchClass(const cv::Mat& src, const std::string& class_id, 
                              float threshold,
                              const cv::Rect& roi = cv::Rect(0,0,0,0), 
                              const cv::Mat& mask = cv::Mat());
    //void loadPyramidDetector();
private:
    // class id (string) --> template (images) vector (consider scale vector)
    std::map<std::string, std::vector<std::vector<Pyramid>>> m_classPyramids;

    MatchingParams m_params;
    std::vector<float> m_angleVec;
    std::vector<float> m_scaleVec;

    PyrDetectorParams m_detectorParams;
    PyramidDetector *m_detector = nullptr;
};

#endif //SHAPE_MATCHING_MATCHING_H
