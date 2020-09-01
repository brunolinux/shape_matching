#ifndef MAIN_UTIL_H
#define MAIN_UTIL_H

#include "src/matching.h"
#include "src/cuda_icp/scene/edge_scene/edge_scene.h"
#include "src/cuda_icp/icp.h"

cv::Matx33f icpMatching(const Matching &matching, const Scene_edge &scene,
                        const MatchingResult &match)
{
    Pyramid matchedPyr = matching.getClassPyramid(match);
    const auto& patternLevel0 = matchedPyr[0];
    std::vector<::Vec2f> model_pcd(patternLevel0.m_features.size());
    for(int i = 0; i < patternLevel0.m_features.size(); i++){
        auto& feat = patternLevel0.m_features[i];
        model_pcd[i] = {
                float(feat.x + match.x),
                float(feat.y + match.y)
        };
    }
    cuda_icp::RegistrationResult result = cuda_icp::ICP2D_Point2Plane_cpu(model_pcd, scene);
    auto t = result.transformation_;
    cv::Matx33f icp_mat;
    icp_mat << t[0][0], t[0][1], t[0][2],
            t[1][0], t[1][1], t[1][2],
            t[2][0], t[2][1], t[2][2];

    return icp_mat;
}

#endif