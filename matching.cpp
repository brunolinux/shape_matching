//
// Created by bruno on 8/15/20.
//

#include "matching.h"

#include <utility>

MatchingParams::MatchingParams()
{
    angle_start = 0.f;
    angle_end = 360.f;
    angle_step = 1.f;

    scale_start = 1.f;
    scale_end = 1.f;
    scale_step = 0.f;

    T_vec = {4, 8};
}

MatchingParams::MatchingParams(float _angle_start, float _angle_end, float _angle_step,
                               float _scale_start, float _scale_end, float _scale_step,
                               std::vector<int>  _T_vec)
        :angle_start(_angle_start), angle_end(_angle_end), angle_step(_angle_step),
         scale_start(_scale_start), scale_end(_scale_end), scale_step(_scale_step),
         T_vec(std::move(_T_vec))
{
    CV_Assert(angle_end > angle_start && angle_step >= 0);
    CV_Assert(scale_start > scale_end && scale_step >= 0);
}

std::vector<AS_pair> MatchingParams::createASPairs()
{
#define EPSILON  (0.0001f)
    std::vector<AS_pair> ASPair_vec;

    bool use_angle_step = true;
    bool use_scale_step = true;
    if ((angle_end < angle_start + angle_step) ||
        (angle_end < angle_start + EPSILON)) {
        use_angle_step = false;
    }
    if ((scale_start < scale_end + scale_step) ||
        (scale_start < scale_end + EPSILON)){
        use_scale_step = false;
    }

    if(!use_angle_step && !use_scale_step){
        ASPair_vec.emplace_back(angle_start, scale_start);
    }else if(!use_angle_step){
        float angle = angle_start;
        float scale = scale_start;
        while(scale > scale_end) {
            ASPair_vec.emplace_back(angle, scale);
            scale -= scale_step;
        }
    }else if(!use_scale_step){

        float scale = scale_start;
        float angle = angle_start;
        while(angle < angle_end) {
            ASPair_vec.emplace_back(angle, scale);
            angle += angle_step;
        }
    } else {

        float angle = angle_start;
        while(angle < angle_end) {
            float scale = scale_start;
            while(scale > scale_end) {
                ASPair_vec.emplace_back(angle, scale);
                scale -= scale_step;
            }
            angle += angle_step;
        }
    }
    return ASPair_vec;
}





Matching::Matching(const MatchingParams &params)
:m_params(params)
{
    PyrDetectorParams detectorParams;
    detectorParams.pyramid_level = m_params.T_vec.size();
    detectorParams.angle_bin_number = 8;

    m_detector = new PyramidDetector(detectorParams);
}

void Matching::addClassPyramid(const cv::Mat& src, const cv::Mat& mask, const std::string& class_id)
{
    Pyramid pyr = m_detector->detect(src, mask);
    m_classPyramids[class_id].push_back(pyr);
}

void Matching::matchClass(const cv::Mat& src, const std::string& class_id, const cv::Rect& roi)
{
    // create linear memory

}