//
// Created by bruno on 8/14/20.
//

#ifndef SHAPE_MATCHING_PYRAMID_H
#define SHAPE_MATCHING_PYRAMID_H

#include "feature.h"

struct Pattern {
    Pattern()
    {base_x = 0, base_y = 0, width = 0, height = 0, pyramid_level = 0, m_features.clear();};

    int base_x;
    int base_y;
    int width;
    int height;
    int pyramid_level;
    std::vector<Feature> m_features;
};


struct PyrDetectorParams {
    PyrDetectorParams();

    float weak_threshold;
    float strong_threshold;
    int num_feature_toplevel;
    int pyramid_level;
    int angle_bin_number;
};

class Pyramid {
public:
    Pyramid();

    void cropLocationRange();

    Pyramid rotatePyramid(float _angle, int angle_bin_number) const;

    void clear() { m_pattern.clear(); }

    size_t size() const {return m_pattern.size();}

    const Pattern& operator[](size_t index) const {
        return m_pattern.at(index);
    }

    const Pattern& at(size_t index) const {
        return m_pattern.at(index);
    }

    void push_back(const Pattern& pattern) {
        m_pattern.push_back(pattern);
    }

    cv::Point2f getPatternCenter(size_t index) const {
        return cv::Point2f(m_pattern[index].base_x + m_pattern[index].width / 2,
                           m_pattern[index].base_y + m_pattern[index].height / 2);
    }
private:
    std::vector<Pattern> m_pattern;
};


class PyramidDetector {
public:

    explicit PyramidDetector(const PyrDetectorParams &parameters = PyrDetectorParams())
        :m_params(parameters)
    {};

    Pyramid detect(const cv::Mat& src, const cv::Mat& mask) const;
private:
    PyrDetectorParams m_params;
};


#endif //SHAPE_MATCHING_PYRAMID_H
