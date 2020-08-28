
#include <iostream>
#include "feature.h"

void write(cv::FileStorage& fs, const std::string&, const Feature& feature)
{
    fs << "{" << "x" << feature.x << "y" << feature.y
       << "angle" << feature.angle
       << "angle_quantized" << feature.angle_quantized << "}";
}

void read(const cv::FileNode& node, Feature& feature, const Feature& default_value)
{
    feature.x = (int)node["x"];
    feature.y = (int)node["y"];
    feature.angle = (float)node["angle"];
    feature.angle_quantized = (int)node["angle_quantized"];
}


// 注意: angle 是顺时针角度
Feature Feature::rotateFeature(const cv::Point2f& _center, const cv::Point2f& _base,
                               float _angle, float _sin_angle, float _cos_angle, int angle_bin_number) const
{
    // feature 坐标是相对坐标 (在 pyramid 类的 cropLocationRange 实现)
    cv::Point2f p = cv::Point2f(x, y) + _base - _center;

    //  注意: x 轴水平朝右，y 轴竖直朝下
    /*
    [  cos_a, -sin_a,
       sin_a, cos_a ]
    */
    cv::Point2f rot_p;
    rot_p.x = _cos_angle * p.x - _sin_angle * p.y + _center.x;
    rot_p.y = _sin_angle * p.x + _cos_angle * p.y + _center.y;

    int new_x = round(rot_p.x);
    int new_y = round(rot_p.y);

    float new_angle = angle + _angle;
    while (new_angle > 360) new_angle -= 360;
    while (new_angle < 0) new_angle += 360;

    int new_quantized_angle = round(new_angle * 2 * angle_bin_number / 360);
    new_quantized_angle &= (angle_bin_number - 1);

    Feature new_f(new_x, new_y, new_angle, new_quantized_angle);
    return std::move(new_f);
}


FeatureCandidate::FeatureCandidate(int x, int y, float angle, uchar angle_quantized, float score)
:m_feature(x, y, angle, angle_quantized), m_score(score)
{}

void createFeatures(const cv::Mat& src, const cv::Mat& mask,
                    std::vector<Feature>& features,
                    float weak_threshold, float strong_threshold,
                    int angle_bin_number, int num_features)
{
    cv::Mat magnitude, angle;
    cv::Mat_<uchar> quantized_angle;
    quantizedOrientations(src, magnitude, angle, quantized_angle,
                          weak_threshold, angle_bin_number);

    // Want features on the border to distinguish from background
//    cv::Mat local_mask;
//    if (!mask.empty())
//    {
//        cv::erode(mask, local_mask, cv::Mat(), cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
//    }

    float threshold_sq = pow(strong_threshold, 2);

#define NMS_KSize (5)
    cv::Mat localNMSMat = createNMSMat(magnitude, mask, NMS_KSize);

    CandidatesVec candidates;
    for (int r = 0; r < magnitude.rows; r ++) {
        const float * mag_ptr = magnitude.ptr<float>(r);
        const uchar * quant_angle_ptr = quantized_angle.ptr<uchar>(r);
        const uchar * nms_ptr = localNMSMat.ptr<uchar>(r);
        const float * angle_ptr = angle.ptr<float>(r);
        for (int c = 0; c < localNMSMat.cols; c++) {
            // quant_angle_ptr[c] > 0 说明该点有值，然后存储的实际量化角度索引要 - 1
            if ( nms_ptr[c] && (quant_angle_ptr[c] > 0) && (mag_ptr[c] > threshold_sq)) {
                FeatureCandidate f(c, r, angle_ptr[c], quant_angle_ptr[c] - 1, mag_ptr[c]);
                candidates.push_back(f);
            }
        }
    }

    features.clear();
    if (candidates.size() < num_features) {
        if(candidates.size() <= 4) {
            std::cerr << "too few features, abort" << std::endl;
            return;
        }
        std::cout << "have no enough features, exaustive mode" << std::endl;
    }

    // NOTE: Stable sort to agree with old code, which used std::list::sort()
    std::stable_sort(candidates.begin(), candidates.end());

    // Use heuristic based on surplus of candidates in narrow outline for initial distance threshold
    int distance = candidates.size() / num_features + 1;

    // selectScatteredFeatures always return true
    selectScatteredFeatures(candidates, features, num_features, distance);
}


bool selectScatteredFeatures(const CandidatesVec &candidates,
                             std::vector<Feature> &features,
                             size_t num_features, int distance)
{
    features.reserve(num_features);

    int distance_sq = distance * distance;

    bool first_select = true;
    int i = 0;
    while(true) {
        FeatureCandidate c = candidates[i];

        // Add if sufficient distance away from any previously chosen feature
        bool keep = true;
        for (int j = 0; (j < (int)features.size()) && keep; ++j) {
            Feature f = features[j];
            keep = (c.m_feature.dist(f) >= distance_sq);
        }
        if (keep)
            features.push_back(c.m_feature);

        if (++i == (int)candidates.size()) {
            bool num_ok = (features.size() >= num_features);

            if(first_select) {
                if(num_ok) {
                    features.clear(); // we don't want too many first time
                    i = 0;
                    distance += 1;
                    distance_sq = distance * distance;
                    continue;
                } else {
                    first_select = false;
                }
            }

            // Start back at beginning, and relax required distance
            i = 0;
            distance -= 1;
            distance_sq = distance * distance;
            if (num_ok || distance < 3){
                break;
            }
        }
    }
    return true;
}