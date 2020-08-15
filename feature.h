//
// Created by bruno on 8/14/20.
//

#ifndef SHAPE_MATCHING_FEATURE_H
#define SHAPE_MATCHING_FEATURE_H

#include <vector>
#include <opencv2/core.hpp>



struct Feature {
    Feature(int _x, int _y, float _angle, uchar _angle_quantized)
    :x(_x), y(_y), angle(_angle), angle_quantized(_angle_quantized)
    {}

    int dist(const Feature &rhs) const
    {
        return (x-rhs.x)*(x-rhs.x) + (y-rhs.y)*(y-rhs.y);
    }

    int x;
    int y;
    float angle;
    uchar angle_quantized;  ///< range: 0 ~ bins_number-1 ( bins_number = 8, 16, 32)
};


struct FeatureCandidate
{
    FeatureCandidate(int x, int y, float angle, uchar angle_quantized, float score);

    /// Sort candidates with high score to the front
    bool operator<(const FeatureCandidate &rhs) const
    {
        return m_score > rhs.m_score;
    }

    Feature m_feature;
    float m_score;
};

using CandidatesVec = std::vector<FeatureCandidate>;

void createFeatures(const cv::Mat& src, const cv::Mat& mask,
                    std::vector<Feature>& features,
                    float weak_threshold, float strong_threshold,
                    int angle_bin_number, int num_features);


void hysteresisGradient(const cv::Mat &magnitude, const cv::Mat &angle,
                        cv::Mat_<uchar> &quantized_angle,
                        float threshold, int angle_bin_number);

void quantizedOrientations(const cv::Mat &src, cv::Mat &magnitude,
                           cv::Mat &angle, cv::Mat_<uchar>& quantized_angle,
                           float threshold, int angle_bin_number);

bool selectScatteredFeatures(const CandidatesVec &candidates,
                             std::vector<Feature> &features,
                             size_t num_features, int distance);

cv::Mat createNMSMat(const cv::Mat& src, const cv::Mat& mask, int ksize);
#endif //SHAPE_MATCHING_FEATURE_H
