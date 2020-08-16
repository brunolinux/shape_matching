//
// Created by bruno on 8/15/20.
//

#include "matching.h"
#include "similarity.h"
#include "response_map.h"
#include <opencv2/opencv.hpp>

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

void MatchingParams::createAngleScaleVec(std::vector<float>& angle_vec, 
                                         std::vector<float>& scale_vec)
{
    angle_vec.clear();
    scale_vec.clear();

#define EPSILON  (0.0001f)
    
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

    if (use_angle_step) {
        float angle = angle_start;
        while (angle < angle_end) {
            angle_vec.push_back(angle);
            angle += angle_step;
        }
    }
    else {
        angle_vec.push_back(angle_start);
    }

    if (use_scale_step) {
        float scale = scale_start;
        while(scale > scale_end) {
            scale_vec.push_back(scale);
            scale -= scale_step;
        }
    }
    else {
        scale_vec.push_back(scale_start);
    }
}


Matching::Matching(const MatchingParams &params)
:m_params(params)
{
    m_params.createAngleScaleVec(m_angleVec, m_scaleVec);

    m_detectorParams.pyramid_level = m_params.T_vec.size();
    m_detectorParams.angle_bin_number = 8;  // default 8

    m_detector = new PyramidDetector(m_detectorParams);
}

void Matching::addClassPyramid(const cv::Mat& src, const cv::Mat& mask, const std::string& class_id)
{
    CV_Assert(!src.empty());
    CV_Assert(mask.empty() || src.size() == mask.size());

    std::vector<Pyramid> pyr_vec;
    cv::Mat src_resized, mask_resized = cv::Mat();
    // 只保存 scale_vec, 不考虑 angle_vec
    // 因为缩放会显著改变特征点位置 (因为对比度变化)，但旋转不会
    for (auto scale : m_scaleVec) {
        cv::Size size(src.cols * scale, src.rows * scale);
        cv::resize(src, src_resized, size);
        if (!mask.empty()) {
            cv::resize(mask, mask_resized, size);
        }

        Pyramid pyr = m_detector->detect(src_resized, mask_resized);
        pyr_vec.push_back(pyr);
    }
    m_classPyramids[class_id].push_back(pyr_vec);
}

static std::vector<cv::Size> getPyrSize(cv::Size& size, int pyr_level)
{
    std::vector<cv::Size> size_vec;
    for (int i = 0; i < pyr_level; i++) {
        size_vec.push_back(size);
        size /= 2;
    }
    return std::move(size_vec);
}


static std::vector<MatchingResult> coarseMatching(const LinearMemories& lm,
                                                  const Pattern& pattern,
                                                  cv::Size img_size, int lowest_T,
                                                  float threshold,
                                                  float scale, float angle, const std::string& class_id)
{
    cv::Mat similarities;
    std::vector<MatchingResult> candidates;

    if (pattern.m_features.size() < 64) {
        similarity_64(lm, pattern, similarities, img_size, lowest_T);
        similarities.convertTo(similarities, CV_16U);
    }
    else if (pattern.m_features.size() < 8192) {
        similarity(lm, pattern, similarities, img_size, lowest_T);
    }
    else {
        CV_Error(cv::Error::StsBadArg, "feature size too large");
    }

    // Find initial matches
    for (int r = 0; r < similarities.rows; ++r)
    {
        ushort* row = similarities.ptr<ushort>(r);
        for (int c = 0; c < similarities.cols; ++c)
        {
            int raw_score = row[c];
            float score = (raw_score * 100.f) / (4 * pattern.m_features.size());

            if (score > threshold)
            {
                // @todo, 这是是不是有问题
                //int offset = lowest_T / 2 + (lowest_T % 2 - 1);
                //int x = c * lowest_T + offset;
                //int y = r * lowest_T + offset;
                int x = c * lowest_T;
                int y = r * lowest_T;

                candidates.push_back(MatchingResult(x, y, score, scale, angle, class_id));
            }
        }
    }

    return std::move(candidates);
}



void Matching::matchClass(const cv::Mat& src, const std::string& class_id,
    float threshold,
    const cv::Rect& roi, const cv::Mat& mask)
{
    // create linear memory
    LinearMemoryPyramid linearMemPyr = createLinearMemoryPyramid(src, mask, m_params.T_vec,
        m_detectorParams.weak_threshold,
        m_detectorParams.angle_bin_number);
    //
    int pyramid_level = m_detectorParams.pyramid_level;
    std::vector<cv::Size> imgSizePyr = getPyrSize(src.size(), pyramid_level);

    for (const auto& pyr_origin_vec : m_classPyramids[class_id]) {  // index: per image (template)
        for (int i = 0; i < pyr_origin_vec.size(); i++) {           // index: per scale
            for (float angle : m_angleVec) {                        // index: per angle

                Pyramid pyr = pyr_origin_vec[i].rotatePyramid(angle, m_detectorParams.angle_bin_number);

                // 金字塔最高层 (面积最小)
                std::vector<MatchingResult> candidates = coarseMatching(linearMemPyr.back(),
                    pyr.at(pyr.size() - 1), imgSizePyr.back(), m_params.T_vec.back(),
                    threshold, m_scaleVec[i], angle, class_id);
            }
        }
    }
}


