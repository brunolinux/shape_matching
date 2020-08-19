//
// Created by bruno on 8/15/20.
//

#ifndef SHAPE_MATCHING_MATCHING_H
#define SHAPE_MATCHING_MATCHING_H

#include <map>

#include "pyramid.h"
#include "response_map.h"

struct MatchingParams
{
    MatchingParams();
    MatchingParams(float _angle_start, float _angle_end, float _angle_step,
                  float _scale_start, float _scale_end, float _scale_step,
                  std::vector<int>  _T_vec);

    void createAngleScaleVec(std::vector<float>& angle_vec, std::vector<float>& scale_vec) const;

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
    MatchingResult(int _x, int _y, float _sim, int _scale_id, float _scale,
                   int _angle_id, float _angle,
                   int _template_id, const std::string& _id)
        :x(_x), y(_y), similarity(_sim),
         scale_id(_scale_id), scale(_scale),
         angle_id(_angle_id), angle(_angle),
         template_id(_template_id), class_id(_id)
    {}

    /// Sort matches with high similarity to the front
    bool operator<(const MatchingResult &rhs) const
    {
        // Secondarily sort on template_id for the sake of duplicate removal
        if (similarity != rhs.similarity)
            return similarity > rhs.similarity;
        else
            return abs(scale - 1) < abs(rhs.scale - 1);
    }

    int x;
    int y;
    float similarity;
    int scale_id;
    float scale;
    int angle_id;
    float angle;
    int template_id;
    std::string class_id;
};

using MatchingResultVec = std::vector<MatchingResult>;

class Matching {
public:
    explicit Matching(const MatchingParams &params = MatchingParams(), int number_feature_toplevel = 63);

    void addClassPyramid(const cv::Mat& src, const cv::Mat& mask, const std::string& class_id);

    Pyramid getClassPyramid(const std::string& class_id, int template_id,
                            int scale_id, int angle_id);

    MatchingResultVec matchClass(const cv::Mat& src, const std::string& class_id,
                                           float threshold,
                                           const cv::Rect& roi = cv::Rect(0, 0, 0, 0),
                                           const cv::Mat& mask = cv::Mat());

private:
    MatchingResultVec coarseMatching(const LinearMemories& lm,
                                     const Pattern& pattern,
                                     const cv::Size& img_size, int lowest_T,
                                     float threshold,
                                     int scale_id, int angle_id,
                                     int template_id, const std::string& class_id);

    // class id (string) --> template (images) vector (consider scale vector)
    std::map<std::string, std::vector<std::vector<Pyramid>>> m_classPyramids;

    MatchingParams m_params;
    std::vector<float> m_angleVec;
    std::vector<float> m_scaleVec;

    PyrDetectorParams m_detectorParams;
    PyramidDetector *m_detector = nullptr;
};

#endif //SHAPE_MATCHING_MATCHING_H
