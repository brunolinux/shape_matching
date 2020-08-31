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

        cv::Matx33f rotation_matrix = createRotationMatrix(_center, _sin_angle, _cos_angle);

        Pattern new_p;
        for (const auto& f: m_pattern[l].m_features) {
            new_p.m_features.push_back(f.rotateFeature(_base, _angle, rotation_matrix, angle_bin_number));
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

cv::Matx33f createRotationMatrix(const cv::Point2f& _center, float _sin_angle, float _cos_angle)
{
    cv::Matx33f t0, t1;
    t0 << 1, 0, -_center.x,
            0, 1, -_center.y,
            0, 0, 1;
    t1 << 1, 0, _center.x,
            0, 1, _center.y,
            0, 0, 1;

    // 注意: x 轴水平朝右，y 轴竖直朝下
    // angle 是逆时针 (和 OpenCV 一致)
    /*
    [   cos_a, sin_a,
       -sin_a, cos_a ]
    */
    cv::Matx33f rot;
    rot <<  _cos_angle, _sin_angle, 0,
           -_sin_angle, _cos_angle, 0,
            0,           0,         1;

    return t1 * rot * t0;
}

/////////////////////////////////////////////////////
// write/read
////////////////////////////////////////////////////
void Pyramid::write(cv::FileStorage &fs) const
{
    fs << "[";
    for (int i = 0; i < m_pattern.size(); i ++) {
        fs << m_pattern[i];
    }
    fs << "]";
}


void Pyramid::read(const cv::FileNode& fs)
{
    m_pattern.clear();

    CV_Assert(fs.type() == cv::FileNode::SEQ);
    cv::FileNodeIterator it = fs.begin(), it_end = fs.end();
    for (; it != it_end; ++it) {
        Pattern pattern;
        (*it) >> pattern;
        m_pattern.push_back(pattern);
    }
}


void write(cv::FileStorage& fs, const std::string&, const Pyramid& pyr)
{
    pyr.write(fs);
}

void read(const cv::FileNode& node, Pyramid& pyr,
          const Pyramid& default_value)
{
    if(node.empty())
        pyr = default_value;
    else
        pyr.read(node);
}

void write(cv::FileStorage& fs, const std::string&, const Pattern& pattern)
{
    fs << "{"
       << "pyr_level" << pattern.pyramid_level
       << "base_x" << pattern.base_x
       << "base_y" << pattern.base_y
       << "width" << pattern.width
       << "height" << pattern.height;

    fs << "features" << "[";
    for (int i = 0; i < pattern.m_features.size(); i ++) {
        fs << pattern.m_features[i];
    }
    fs << "]" << "}";
}


void read(const cv::FileNode& node, Pattern& pattern, const Pattern& default_value)
{
    pattern.pyramid_level = (int)node["pyr_level"];
    pattern.base_x = (int)node["base_x"];
    pattern.base_y = (int)node["base_y"];
    pattern.width = (int)node["width"];
    pattern.height = (int)node["height"];

    cv::FileNode features_node = node["features"];
    CV_Assert(features_node.type() == cv::FileNode::SEQ);
    cv::FileNodeIterator it = features_node.begin(), it_end = features_node.end();
    std::vector<Feature> feature_vec;
    for (; it != it_end; ++it) {
        Feature feature;
        (*it) >> feature;
        feature_vec.push_back(feature);
    }
    pattern.m_features = std::move(feature_vec);
}