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

    float angle_start;          ///< angle 是逆时针角度 (counter-clockwise)
    float angle_end;
    float angle_step;
    float scale_start;
    float scale_end;
    float scale_step;

    std::vector<int> T_vec; /// vector of T

    void write(cv::FileStorage& fs) const;
    void read(const cv::FileNode& fs);
};

struct MatchingResult
{
    MatchingResult(int _x, int _y, const cv::Rect& range, float _sim,
                   int _scale_id, float _scale,
                   int _angle_id, float _angle,
                   int _template_id, const std::string& _id)
        : x(_x), y(_y), origin_temp_rect(range), similarity(_sim),
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
    cv::Rect origin_temp_rect;
    float similarity;
    int scale_id;
    float scale;
    int angle_id;
    float angle;                ///< angle 是逆时针角度 (counter-clockwise)
    int template_id;
    std::string class_id;
};
using MatchingResultVec = std::vector<MatchingResult>;


struct MatchingCandidate {
    MatchingCandidate(int _x, int _y, float _score)
            :x(_x), y(_y), score(_score)
    {}

    int x;
    int y;
    float score;
};
using MatchingCandidateVec = std::vector<MatchingCandidate>;


class Matching {
public:
    explicit Matching(const MatchingParams &params = MatchingParams(),
                      int det_num_feature_top = 63,
                      float det_weak_thres = 30.f,
                      float det_strong_thres = 60.f,
                      int det_angle_bin_num = 8);

    Matching(const MatchingParams &params, const PyrDetectorParams& detector_params);

    void addClassPyramid(const cv::Mat& src, const cv::Mat& mask, const std::string& class_id);


    MatchingResultVec matchClass(const cv::Mat& src, const std::string& class_id,
                                 float threshold,
                                 const cv::Mat& mask = cv::Mat()) const;

    MatchingResultVec matchClassWithNMS(const cv::Mat& src, const std::string& class_id,
                                        float score_threshold, float nms_threshold,
                                        const cv::Mat& mask = cv::Mat(),
                                        const float eta=1, const int top_k=0) const;

    Pyramid getClassPyramid(const MatchingResult& match) const;
    cv::Matx33f getMatchingMatrix(const MatchingResult& match);

    void writeMatchingParams(const std::string& file_name) const;
    static Matching readMatchingParams(const std::string& file_name);

    void writeClassPyramid(const std::string& file_name, const std::string& class_id) const;
    void readClassPyramid(const std::string& file_name, const std::string& class_id);

    cv::Mat createPaddedImage(const cv::Mat& src) const;
private:
    MatchingCandidateVec coarseMatching(const LinearMemories& lm,
                                     const Pattern& pattern,
                                     const cv::Size& img_size, int lowest_T,
                                     float threshold) const;

    // class id (string) --> pyramid (images) vector (consider scale vector)
    std::map<std::string, std::vector<std::vector<Pyramid>>> m_classPyramids;

    MatchingParams m_params;
    std::vector<float> m_angleVec;
    std::vector<float> m_scaleVec;

    PyrDetectorParams m_detectorParams;
    PyramidDetector *m_detector = nullptr;
};

#endif //SHAPE_MATCHING_MATCHING_H
