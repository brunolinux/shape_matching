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
    CV_Assert(angle_end >= angle_start && angle_step >= 0);
    CV_Assert(scale_start >= scale_end && scale_step >= 0);
}

void MatchingParams::createAngleScaleVec(std::vector<float>& angle_vec, 
                                         std::vector<float>& scale_vec) const
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


Matching::Matching(const MatchingParams &params, int number_feature_toplevel)
:m_params(params)
{
    m_params.createAngleScaleVec(m_angleVec, m_scaleVec);

    m_detectorParams.num_feature_toplevel = number_feature_toplevel;
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

Pyramid Matching::getClassPyramid(const std::string& class_id, int template_id,
                                  int scale_id, int angle_id)
{
    const Pyramid &pyr = m_classPyramids[class_id][template_id][scale_id];
    return std::move(pyr.rotatePyramid(m_angleVec[angle_id], m_detectorParams.angle_bin_number));
}


static std::vector<cv::Size> getPyrSize(const cv::Size& size, int pyr_level)
{
    cv::Size size_pyr = size;
    std::vector<cv::Size> size_vec;
    for (int i = 0; i < pyr_level; i++) {
        size_vec.push_back(size_pyr);
        size_pyr /= 2;
    }
    return std::move(size_vec);
}


MatchingResultVec Matching::coarseMatching(const LinearMemories& lm,
                                           const Pattern& pattern,
                                           const cv::Size& img_size, int lowest_T,
                                           float threshold,
                                           int scale_id, int angle_id,
                                           int template_id, const std::string& class_id)
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

            //std::cout << score << std::endl;
            if (score > threshold)
            {
                // @todo, 这是是不是有问题
                //int offset = lowest_T / 2 + (lowest_T % 2 - 1);
                //int x = c * lowest_T + offset;
                //int y = r * lowest_T + offset;
                int x = c * lowest_T;
                int y = r * lowest_T;

                candidates.push_back(MatchingResult(x, y, score, scale_id, m_scaleVec[scale_id],
                                                    angle_id, m_angleVec[angle_id], template_id, class_id));
            }
        }
    }

    {
        double minVal;
        double maxVal;
        cv::Point minLoc;
        cv::Point maxLoc;
        minMaxLoc(similarities, &minVal, &maxVal, &minLoc, &maxLoc );

        std::cout << (maxVal * 100.f) / (4 * pattern.m_features.size()) << std::endl;
/*        std::cout << "num feature: " << pattern.m_features.size() << "\t\t";
        std::cout << "angle id:" << angle_id << "\t" << maxVal << "\n";
        cv::Mat simShow;
        similarities.convertTo(simShow, CV_8U, 255./90);
        cv::imwrite("img/intermediate/" + std::to_string(angle_id) + ".png", simShow);

        cv::Mat bk = cv::Mat::zeros(pattern.height, pattern.width, CV_8UC3);
        for (int i = 0; i < pattern.m_features.size(); i ++) {
            cv::circle(bk, cv::Point(pattern.m_features[i].x, pattern.m_features[i].y), 4, cv::Scalar(0, 0, 255), 2);
        }
        cv::imwrite("img/intermediate/feature" + std::to_string(angle_id) + ".png", bk);*/
    }

    return std::move(candidates);
}



MatchingResultVec Matching::matchClass(const cv::Mat& src, const std::string& class_id,
                                     float threshold,
                                     const cv::Rect& roi, const cv::Mat& mask)
{
    std::vector<MatchingResult> matching_vec;
    // create linear memory
    LinearMemoryPyramid linearMemPyr = createLinearMemoryPyramid(src, mask, m_params.T_vec,
        m_detectorParams.weak_threshold,
        m_detectorParams.angle_bin_number);
    //
    int pyramid_level = m_detectorParams.pyramid_level;
    std::vector<cv::Size> imgSizePyr = getPyrSize(src.size(), pyramid_level);

    for (int template_id = 0; template_id < m_classPyramids[class_id].size(); template_id ++) {  // index: per image (template)
        const auto & pyr_origin_vec = m_classPyramids[class_id][template_id];
        for (int scale_id = 0; scale_id < pyr_origin_vec.size(); scale_id++) {                  // index: per scale
            for (int angle_id = 0; angle_id < m_angleVec.size(); angle_id++) {                  // index: per angle

                float angle = m_angleVec[angle_id];
                Pyramid pyr = pyr_origin_vec[scale_id].rotatePyramid(angle, m_detectorParams.angle_bin_number);

                // 金字塔最高层 (面积最小)
                std::vector<MatchingResult> candidates = coarseMatching(linearMemPyr.back(),
                                                                        pyr.at(pyr.size() - 1), imgSizePyr.back(), m_params.T_vec.back(),
                                                                        threshold, scale_id, angle_id, template_id, class_id);

                // Locally refine each match by marching up the pyramid
                std::vector<MatchingResult> new_candidates;
                for (int l = pyramid_level - 2; l >= 0; --l)
                {
                    //const std::vector<LinearMemories> &lms = linearMemPyr.at(l);
                    int T = m_params.T_vec[l];
                    //int start = static_cast<int>(l);
                    cv::Size size = imgSizePyr[l];
                    int border = 8 * T;
                    //int offset = T / 2 + (T % 2 - 1);
                    int max_x = size.width - pyr.at(l).width - border;
                    int max_y = size.height - pyr.at(l).height - border;

                    cv::Mat similarities2;
                    for (int m = 0; m < (int)candidates.size(); ++m)
                    {
                        MatchingResult &match2 = candidates[m];
                        // 金字塔原理
                        int x = match2.x * 2 + 1; /// @todo Support other pyramid distance
                        int y = match2.y * 2 + 1;

                        // Require 8 (reduced) row/cols to the up/left
                        x = std::max(x, border);
                        y = std::max(y, border);

                        // Require 8 (reduced) row/cols to the down/left, plus the template size
                        x = std::min(x, max_x);
                        y = std::min(y, max_y);

                        // Compute local similarity maps for each ColorGradient
                        int numFeatures = 0;

                        {
                            const Pattern &pattern = pyr.at(l);
                            numFeatures += static_cast<int>(pattern.m_features.size());

                            if (pattern.m_features.size() < 64) {
                                similarityLocal_64(linearMemPyr.at(l), pattern, similarities2, size, T, cv::Point(x, y));
                                similarities2.convertTo(similarities2, CV_16U);
                            } else if (pattern.m_features.size() < 8192) {
                                similarityLocal(linearMemPyr.at(l), pattern, similarities2, size, T, cv::Point(x, y));
                            } else {
                                CV_Error(cv::Error::StsBadArg, "feature size too large");
                            }
                        }

                        // Find best local adjustment
                        float best_score = 0;
                        int best_r = -1, best_c = -1;
                        for (int r = 0; r < similarities2.rows; ++r)
                        {
                            auto *row = similarities2.ptr<ushort>(r);
                            for (int c = 0; c < similarities2.cols; ++c)
                            {
                                int score_int = row[c];
                                float score = (score_int * 100.f) / (4 * numFeatures);

                                if (score > best_score)
                                {
                                    best_score = score;
                                    best_r = r;
                                    best_c = c;
                                }
                            }
                        }

                        if (best_score > threshold) {
                            int _x = (x / T - 8 + best_c) * T;
                            int _y = (y / T - 8 + best_r) * T;
                            MatchingResult ret(_x, _y, best_score, candidates[m].scale_id, candidates[m].scale,
                                               candidates[m].angle_id, candidates[m].angle,
                                               candidates[m].template_id, candidates[m].class_id);

                            bool save = true;
                            for (const auto& candidate : new_candidates) {
                                if (candidate.x == _x && candidate.y == _y) {
                                    save = false;
                                    break;
                                }
                            }
                            if (save)
                                new_candidates.push_back(ret);
                        }
                    }
                    // update candidate
                    candidates.clear();
                    candidates.insert(candidates.begin(), new_candidates.begin(), new_candidates.end());
                    new_candidates.clear();
                }

                matching_vec.insert(matching_vec.end(), candidates.begin(), candidates.end());
            }
        }
    }

    std::sort(matching_vec.begin(), matching_vec.end());
    return matching_vec;
}


