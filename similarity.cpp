#include "similarity.h"
#include "pyramid.h"
#include "response_map.h"

#include <opencv2/core.hpp>

#include <mipp.h>


void similarity(const std::vector<cv::Mat>& linear_memories, 
                const Pattern & pattern,
                cv::Mat& dst, cv::Size img_size, int T)
{
    // we only have one modality, so 8192*2, due to mipp, back to 8192
    CV_Assert(pattern.m_features.size() < 8192);

    // Decimate input image size by factor of T
    // 基准点是 pattern 的 base point (base_x, base_y)
    int W = img_size.width / T;
    int H = img_size.height / T;

    // Feature dimensions, decimated by factor T and rounded up
    int wf = (pattern.width - 1) / T + 1;
    int hf = (pattern.height - 1) / T + 1;

    // Span is the range over which we can shift the template around the input image
    int span_x = W - wf + 1;
    int span_y = H - hf + 1;

    // T 间隔遍历结果
    dst = cv::Mat::zeros(H, W, CV_16U);
    short* dst_ptr = dst.ptr<short>();

#ifdef USE_MIPP
    mipp::Reg<uint8_t> zero_v(uint8_t(0));
#endif
    for (int i = 0; i < (int)pattern.m_features.size(); ++i)
    {

        Feature f = pattern.m_features[i];

        if (f.x < 0 || f.x >= img_size.width || f.y < 0 || f.y >= img_size.height)
            continue;

        const uchar* lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        int j = 0;
        int end = 0;
#ifdef USE_MIPP
        for (int r = 0; r < span_y; r++) {
            j = r * W;
            end = r * W + span_x;
            // *2 to avoid int8 read out of range
            for (; j <= end - mipp::N<int16_t>() * 2; j += mipp::N<int16_t>()) {
                mipp::Reg<uint8_t> src8_v((uint8_t*)lm_ptr + j);

                // uchar to short, once for N bytes
                mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

                mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + j);

                mipp::Reg<int16_t> res_v = src16_v + dst_v;
                res_v.store((int16_t*)dst_ptr + j);
            }
            // 遍历基准点，累加该特征点的响应值
            for (; j < end; j++)
                dst_ptr[j] += short(lm_ptr[j]);
        }
#else 
        for (int r = 0; r < span_y; r++) {
            for (j = r * W; j < r * W + span_x; j++) {
                dst_ptr[j] += short(lm_ptr[j]);
            }
        }
#endif
    }
}

void similarity_64(const std::vector<cv::Mat>& linear_memories, 
                   const Pattern& pattern,
                   cv::Mat& dst, cv::Size img_size, int T)
{
    // 63 features or less is a special case because the max similarity per-feature is 4.
    // 255/4 = 63, so up to that many we can add up similarities in 8 bits without worrying
    // about overflow. Therefore here we use _mm_add_epi8 as the workhorse, whereas a more
    // general function would use _mm_add_epi16.
    CV_Assert(pattern.m_features.size() < 64);
    /// @todo Handle more than 255/MAX_RESPONSE features!!

    // Decimate input image size by factor of T
    int W = img_size.width / T;
    int H = img_size.height / T;

    // Feature dimensions, decimated by factor T and rounded up
    int wf = (pattern.width - 1) / T + 1;
    int hf = (pattern.height - 1) / T + 1;

    // Span is the range over which we can shift the template around the input image
    int span_x = W - wf;
    int span_y = H - hf;


    /// @todo In old code, dst is buffer of size m_U. Could make it something like
    /// (span_x)x(span_y) instead?
    dst = cv::Mat::zeros(H, W, CV_8U);
    uchar* dst_ptr = dst.ptr<uchar>();

    // Compute the similarity measure for this template by accumulating the contribution of
    // each feature
    for (int i = 0; i < (int)pattern.m_features.size(); ++i)
    {
        // Add the linear memory at the appropriate offset computed from the location of
        // the feature in the template
        Feature f = pattern.m_features[i];
        // Discard feature if out of bounds
        /// @todo Shouldn't actually see x or y < 0 here?
        if (f.x < 0 || f.x >= img_size.width || f.y < 0 || f.y >= img_size.height)
            continue;
        const uchar* lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        // Now we do an aligned/unaligned add of dst_ptr and lm_ptr with template_positions elements
        int j = 0;
        int end = 0;
#ifdef USE_MIPP
        for (int r = 0; r < span_y; r++) {
            j = r * W;
            end = r * W + span_x;
            for (; j <= end - mipp::N<uint8_t>(); j += mipp::N<uint8_t>()) {
                mipp::Reg<uint8_t> src_v((uint8_t*)lm_ptr + j);
                mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr + j);

                mipp::Reg<uint8_t> res_v = src_v + dst_v;
                res_v.store((uint8_t*)dst_ptr + j);
            }

            for (; j < end; j++)
                dst_ptr[j] += lm_ptr[j];
        }
#else 
        for (int r = 0; r < span_y; r++) {
            for (j = r * W; j < r * W + span_x; j++) {
                dst_ptr[j] += short(lm_ptr[j]);
            }
        }
#endif


    }
}