//
// Created by bruno on 8/14/20.
//

#include <opencv2/imgproc.hpp>
#include <iostream>

#include "pyramid.h"
#include "feature.h"

PyrDetectorParams::PyrDetectorParams()
{
    weak_threshold = 30.f;
    strong_threshold = 60.f;

    num_feature_toplevel = 63;
    pyramid_level = 2;
    angle_bin_number = 8;
}

Pyramid::Pyramid()
{
    clear();
}

Pyramid Pyramid::rotatePyramid(float _angle, int angle_bin_number) const
{
    float _sin_angle = sin(_angle * CV_PI / 180);
    float _cos_angle = cos(_angle * CV_PI / 180);

    Pyramid new_pyr;

    
    for (int l = 0; l < m_pattern.size(); l++) {
        cv::Point2f _center = getPatternCenter(l);
        cv::Point2f _base(m_pattern[l].base_x, m_pattern[l].base_y);

        Pattern new_p;
        for (const auto& f: m_pattern[l].m_features) {
            new_p.m_features.push_back(f.rotateFeature(_center, _base, _angle, _sin_angle, _cos_angle, angle_bin_number));
        }

        new_p.pyramid_level = l;
        new_pyr.push_back(new_p);
    }
    new_pyr.cropLocationRange();

    return std::move(new_pyr);
}


void Pyramid::cropLocationRange()
{
    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int max_y = std::numeric_limits<int>::min();

    // First pass: find min/max feature x,y over all pyramid levels and modalities
    for (auto & templ : m_pattern)
    {
        for (int j = 0; j < (int)templ.m_features.size(); ++j)
        {
            int x = templ.m_features[j].x << templ.pyramid_level;
            int y = templ.m_features[j].y << templ.pyramid_level;
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
        }
    }

    /// @todo Why require even min_x, min_y?
    if (min_x % 2 == 1)
        --min_x;
    if (min_y % 2 == 1)
        --min_y;

    // Second pass: set width/height and shift all feature positions
    for (auto & templ : m_pattern)
    {
        templ.width = (max_x - min_x) >> templ.pyramid_level;
        templ.height = (max_y - min_y) >> templ.pyramid_level;
        templ.base_x = min_x >> templ.pyramid_level;
        templ.base_y = min_y  >> templ.pyramid_level;

        // 相对基准点坐标
        for (int j = 0; j < (int)templ.m_features.size(); ++j)
        {
            templ.m_features[j].x -= templ.base_x;
            templ.m_features[j].y -= templ.base_y;
        }
    }
}


Pyramid PyramidDetector::detect(const cv::Mat &src, const cv::Mat &mask) const
{
    cv::Mat src_pyr = src.clone();
    cv::Mat mask_pyr = mask.empty() ? cv::Mat() : mask.clone();
    int num_feature = m_params.num_feature_toplevel;

    Pyramid pyramid;
    for (int l = 0; l < m_params.pyramid_level; l ++) {
        std::vector<Feature> features;
        createFeatures(src_pyr, mask_pyr, features,
                       m_params.weak_threshold, m_params.strong_threshold,
                       m_params.angle_bin_number, num_feature);

        if (features.empty()) {     // no feature found
            pyramid.clear();
            return pyramid;
        }
        Pattern pat;
        pat.pyramid_level = l;
        pat.m_features = std::move(features);
        pyramid.push_back(pat);

        if (l < m_params.pyramid_level - 1) {
            cv::Size size(src_pyr.cols/2, src_pyr.rows/2);
            cv::pyrDown(src_pyr, src_pyr, size);

            if (!mask.empty()) {
                cv::Mat next_mask;
                cv::resize(mask_pyr, next_mask, size, 0.0, 0.0, cv::INTER_NEAREST);
                mask_pyr = next_mask;
            }

            // Some parameters need to be adjusted
            num_feature /= 2; /// @todo Why not 4?
        }
    }
    pyramid.cropLocationRange();

    return pyramid;
}

