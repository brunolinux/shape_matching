//
// Created by bruno on 8/15/20.
//

#include "matching.h"
#include "similarity.h"
#include "response_map.h"
#include "nms.h"
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

void MatchingParams::write(cv::FileStorage& fs) const {
    fs << "{";
    fs << "angle" << "{" << "start" << angle_start
       << "end" << angle_end << "step" << angle_step << "}"
       << "scale" << "{" << "start" << scale_start
       << "end" << scale_end << "step" << scale_step << "}";

    fs << "T_vec" << "[";
    for (auto t : T_vec) {
        fs << t;
    }
    fs << "]";

    fs << "}";
}

void MatchingParams::read(const cv::FileNode& fs)
{
    const cv::FileNode angle_fs = fs["angle"];
    angle_start = (float)angle_fs["start"];
    angle_end = (float)angle_fs["end"];
    angle_step = (float)angle_fs["step"];
    const cv::FileNode scale_fs = fs["scale"];
    scale_start = (float)scale_fs["start"];
    scale_end = (float)scale_fs["end"];
    scale_step = (float)scale_fs["step"];

    T_vec.clear();
    const cv::FileNode t_vec_fs = fs["T_vec"];
    CV_Assert(t_vec_fs.type() == cv::FileNode::SEQ);

    cv::FileNodeIterator it = t_vec_fs.begin(), it_end = t_vec_fs.end();
    for (; it != it_end; ++it)
        T_vec.push_back((int)*it);
}



Matching::Matching(const MatchingParams &params, int det_num_feature_top,
                   float det_weak_thres, float det_strong_thres, int det_angle_bin_num)
:m_params(params)
{
    m_params.createAngleScaleVec(m_angleVec, m_scaleVec);

    m_detectorParams.weak_threshold = det_weak_thres;
    m_detectorParams.strong_threshold = det_strong_thres;
    m_detectorParams.num_feature_toplevel = det_num_feature_top;
    m_detectorParams.pyramid_level = m_params.T_vec.size();
    m_detectorParams.angle_bin_number = det_angle_bin_num;  // default 8

    m_detector = new PyramidDetector(m_detectorParams);
}

Matching::Matching(const MatchingParams &params, const PyrDetectorParams& detector_params)
:m_params(params), m_detectorParams(detector_params)
{
    CV_Assert(m_params.T_vec.size() == detector_params.pyramid_level);

    m_params.createAngleScaleVec(m_angleVec, m_scaleVec);
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

Pyramid Matching::getClassPyramid(const MatchingResult& match) const
{
    const Pyramid &pyr = m_classPyramids.at(match.class_id)[match.template_id][match.scale_id];
    return std::move(pyr.rotatePyramid(m_angleVec[match.angle_id], m_detectorParams.angle_bin_number));
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



MatchingCandidateVec Matching::coarseMatching(const LinearMemories& lm,
                                              const Pattern& pattern,
                                              const cv::Size& img_size, int lowest_T,
                                              float threshold) const
{
    cv::Mat similarities;
    MatchingCandidateVec candidates;

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

                candidates.push_back(MatchingCandidate(x, y, score));
            }
        }
    }

    // 测试 部分
//    {
//        static int ind = 0;
//        double minVal;
//        double maxVal;
//        cv::Point minLoc;
//        cv::Point maxLoc;
//        minMaxLoc(similarities, &minVal, &maxVal, &minLoc, &maxLoc );
//
//        std::cout << "id: " << ind << "\tmax response: " << (maxVal * 100.f) / (4 * pattern.m_features.size()) << std::endl;
//        std::cout << "num feature: " << pattern.m_features.size() << std::endl;
//
//        cv::Mat simShow;
//        similarities.convertTo(simShow, CV_8U, 255./90);
//        cv::imwrite("img/tmp/" + std::to_string(ind) + ".png", simShow);
//
//        cv::Mat bk = cv::Mat::zeros(pattern.height, pattern.width, CV_8UC3);
//        for (int i = 0; i < pattern.m_features.size(); i ++) {
//            cv::circle(bk, cv::Point(pattern.m_features[i].x, pattern.m_features[i].y), 4, cv::Scalar(0, 0, 255), 2);
//        }
//        cv::imwrite("img/tmp/feature_" + std::to_string(ind) + ".png", bk);
//        ind ++;
//    }

    return std::move(candidates);
}



MatchingResultVec Matching::matchClass(const cv::Mat& src, const std::string& class_id,
                                     float threshold, const cv::Mat& mask) const
{
    MatchingResultVec matching_vec;
    // create linear memory
    LinearMemoryPyramid linearMemPyr = createLinearMemoryPyramid(src, mask, m_params.T_vec,
        m_detectorParams.weak_threshold, m_detectorParams.angle_bin_number);
    //
    int pyramid_level = m_detectorParams.pyramid_level;
    std::vector<cv::Size> imgSizePyr = getPyrSize(src.size(), pyramid_level);

    for (int template_id = 0; template_id < m_classPyramids.at(class_id).size(); template_id ++) {  // index: per image (template)
        const auto & pyr_origin_vec = m_classPyramids.at(class_id)[template_id];
        for (int scale_id = 0; scale_id < pyr_origin_vec.size(); scale_id++) {                  // index: per scale
            for (int angle_id = 0; angle_id < m_angleVec.size(); angle_id++) {                  // index: per angle (degree)

                float angle = m_angleVec[angle_id];
                Pyramid pyr = pyr_origin_vec[scale_id].rotatePyramid(angle, m_detectorParams.angle_bin_number);

                // 金字塔最高层 (面积最小)
                MatchingCandidateVec candidates = coarseMatching(linearMemPyr.back(),
                                                                pyr.at(pyr.size() - 1),
                                                                imgSizePyr.back(),
                                                                m_params.T_vec.back(),
                                                                threshold);

                // Locally refine each match by marching up the pyramid
                MatchingCandidateVec new_candidates;

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
                        MatchingCandidate &match2 = candidates[m];
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


                            bool save = true;
                            for (const auto& candidate : new_candidates) {
                                if (candidate.x == _x && candidate.y == _y) {
                                    save = false;
                                    break;
                                }
                            }
                            if (save)
                                new_candidates.push_back(MatchingCandidate(_x, _y, best_score));
                        }
                    }
                    // update candidate
                    candidates.clear();
                    std::swap(candidates, new_candidates);
                }

                for (const auto& candidate : candidates) {
                    matching_vec.push_back(MatchingResult(candidate.x, candidate.y,
                                                          pyr.getPyrLevel0Range(),
                                                          candidate.score,
                                                          scale_id, m_scaleVec[scale_id],
                                                          angle_id, m_angleVec[angle_id],
                                                          template_id, class_id));
                }
            }
        }
    }

    std::sort(matching_vec.begin(), matching_vec.end());
    return matching_vec;
}

MatchingResultVec
Matching::matchClassWithNMS(const cv::Mat &src, const std::string &class_id,
                            float score_threshold, float nms_threshold,
                            const cv::Mat &mask, const float eta, const int top_k) const
{
    auto matching_vec = matchClass(src, class_id, score_threshold, mask);

    std::vector<int> idxs;
    {
        std::vector<cv::Rect> bboxes;
        std::vector<float> scores;
        for (const auto& match : matching_vec) {
            bboxes.push_back(cv::Rect(match.x, match.y,
                                      match.origin_temp_rect.width, match.origin_temp_rect.height));
            scores.push_back(match.similarity);
        }
        NMSBoxes(bboxes, scores, 0, nms_threshold, idxs, eta, top_k);
    }

    MatchingResultVec new_matching_vec;
    for (const auto id : idxs) {
        new_matching_vec.push_back(std::move(matching_vec[id]));
    }
    return new_matching_vec;
}

cv::Matx33f Matching::getMatchingMatrix(const MatchingResult &match)
{
    cv::Matx33f scale_mat;
    scale_mat << match.scale, 0, 0,
                 0, match.scale, 0,
                 0, 0, 1;

    const Pyramid &pyr = m_classPyramids.at(match.class_id)[match.template_id][match.scale_id];

    cv::Point2f center = pyr.getPatternCenter(0);
    float _sin_angle = sin(match.angle * CV_PI / 180);
    float _cos_angle = cos(match.angle * CV_PI / 180);
    auto rotation_mat = createRotationMatrix(center, _sin_angle, _cos_angle);

    int delta_x = match.x - match.origin_temp_rect.x;
    int delta_y = match.y - match.origin_temp_rect.y;
    cv::Matx33f trans_mat;
    trans_mat << 1, 0, delta_x,
                 0, 1, delta_y,
                 0, 0, 1;
    return trans_mat * rotation_mat * scale_mat;
}

cv::Mat Matching::createPaddedImage(const cv::Mat &src) const
{
    CV_Assert(!src.empty());

    const auto& T_vec = m_params.T_vec;
    int padding = (int)pow(2, T_vec.size()-1) * T_vec[T_vec.size()-1];

    cv::Mat padded_src;
    cv::copyMakeBorder(src, padded_src, padding, padding, padding, padding, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    int n = padded_src.rows/padding;
    int m = padded_src.cols/padding;
    cv::Rect roi(0, 0, padding*m , padding*n);

    cv::Mat ret =  padded_src(roi).clone();
    CV_Assert(ret.isContinuous());
    return std::move(ret);
}

/////////////////////////////////////////////////////
// write/read
////////////////////////////////////////////////////
static void write(cv::FileStorage& fs, const std::string&, const MatchingParams& x)
{
    x.write(fs);
}


static void read(const cv::FileNode& node, MatchingParams& x,
                 const MatchingParams& default_value = MatchingParams())
{
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}

static void write(cv::FileStorage& fs, const std::string&, const PyrDetectorParams& x)
{
    x.write(fs);
}

static void read(const cv::FileNode& node, PyrDetectorParams& x,
                 const PyrDetectorParams& default_value = PyrDetectorParams())
{
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}


void Matching::writeMatchingParams(const std::string &file_name) const
{
    cv::FileStorage params_fs(file_name, cv::FileStorage::WRITE);

    params_fs << "matching_parameter" << m_params;
    params_fs << "detector_parameter" << m_detectorParams;
}

Matching Matching::readMatchingParams(const std::string &file_name)
{
    cv::FileStorage params_fs(file_name, cv::FileStorage::READ);
    if (!params_fs.isOpened()) {
        std::cerr << "config file: " << file_name << " is not existed!";
        exit(-1);
    }

    MatchingParams matching_param;
    PyrDetectorParams detector_params;
    params_fs["matching_parameter"] >> matching_param;
    params_fs["detector_parameter"] >> detector_params;
    Matching matching(matching_param, detector_params);
    return std::move(matching);
}

void Matching::writeClassPyramid(const std::string &file_name, const std::string& class_id) const
{
    if (m_classPyramids.count(class_id) == 0) {
        std::cerr << "the class: " << class_id << " does not exist";
        exit(-1);
    }

    cv::FileStorage class_fs(file_name, cv::FileStorage::WRITE);
    const auto pyr_vec =  m_classPyramids.at(class_id);

    class_fs << "class_id" << class_id;
    class_fs << "class_vec" << "[";
    for (int i = 0; i < pyr_vec.size(); i ++) {
        class_fs << "{"
                 << "template_id" << i
                 << "scale_pyramid_vec" << "[";
        for (int j = 0; j < pyr_vec[i].size(); j ++) {
            class_fs << "{"
                     << "scale" << m_scaleVec[j]
                     << "pyramid" << pyr_vec[i][j]
                     << "}";
        }
        class_fs << "]" << "}";
    }
    class_fs << "]";
}

void Matching::readClassPyramid(const std::string &file_name, const std::string& class_id)
{
    if (m_classPyramids.count(class_id) != 0) {
        m_classPyramids[class_id].clear();
    }

    cv::FileStorage class_fs(file_name, cv::FileStorage::READ);
    CV_Assert((std::string)class_fs["class_id"] == class_id);

    const cv::FileNode vec_fs = class_fs["class_vec"];
    CV_Assert(vec_fs.type() == cv::FileNode::SEQ);

    cv::FileNodeIterator temp_it = vec_fs.begin(), temp_it_end = vec_fs.end();
    int temp_ind = 0;
    std::vector<std::vector<Pyramid>> pyr_vec_vec;
    for (; temp_it != temp_it_end; ++temp_it) {
        CV_Assert((int)((*temp_it)["template_id"]) == temp_ind);
        temp_ind ++;
        cv::FileNode temp_node = (*temp_it)["scale_pyramid_vec"];

        CV_Assert(temp_node.type() == cv::FileNode::SEQ);
        cv::FileNodeIterator it = temp_node.begin(), it_end = temp_node.end();

        std::vector<Pyramid> pyr_vec;
        int ind = 0;
        for (; it != it_end; ++it) {
            CV_Assert((float)(*it)["scale"] == m_scaleVec[ind]);
            ind ++;

            Pyramid pyr;
            (*it)["pyramid"] >> pyr;
            pyr_vec.push_back(pyr);
        }
        pyr_vec_vec.push_back(pyr_vec);
    }
    m_classPyramids[class_id] = std::move(pyr_vec_vec);
}










