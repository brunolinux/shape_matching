//
// Created by bruno on 8/15/20.
//

#include <opencv2/imgproc.hpp>
#include <iostream>

#ifdef USE_MIPP
#include "mipp/mipp.h"
#endif

#include "response_map.h"
#include "feature.h"


LinearMemoryPyramid createLinearMemoryPyramid(const cv::Mat& src, const cv::Mat& mask,
                                              const std::vector<int>& T_at_level,
                                              float threshold,
                                              int angle_bin_number)
{
    int pyramid_levels = T_at_level.size();

    LinearMemoryPyramid lm_pyramid(pyramid_levels, LinearMemories(angle_bin_number));

    cv::Mat src_pyr = src.clone();
    cv::Mat mask_pyr = mask.clone();
    // For each pyramid level, precompute linear memories
    for (int l = 0; l < pyramid_levels; ++l)
    {
        int T = T_at_level[l];

        cv::Mat_<uchar> quantized, quantized_masked;
        cv::Mat spread_quantized;
        cv::Mat _magnitude, _angle;
        quantizedOrientations(src_pyr, _magnitude, _angle, quantized, threshold, angle_bin_number);

        {
            if (!mask_pyr.empty()) {
                quantized.copyTo(quantized_masked, mask_pyr);
            } else {
                quantized_masked = quantized;
            }
        }

        std::vector<cv::Mat> response_maps;
        spread(quantized_masked, spread_quantized, T, angle_bin_number);
        computeResponseMaps(spread_quantized, response_maps);

        LinearMemories &memories = lm_pyramid[l];
        for (int j = 0; j < 8; ++j)
            linearize(response_maps[j], memories[j], T);

        if (l < pyramid_levels - 1) {
            cv::Size size(src_pyr.cols / 2, src_pyr.rows / 2);
            cv::pyrDown(src_pyr, src_pyr, size);

            if (!mask.empty()) {
                cv::Mat next_mask;
                cv::resize(mask_pyr, next_mask, size, 0.0, 0.0, cv::INTER_NEAREST);
                mask_pyr = next_mask;
            }
        }
    }
    return lm_pyramid;
}


template<class T>
void orUnaligned(const T *src, const int src_stride,
                 T *dst, const int dst_stride,
                 const int width, const int height)
{
#ifdef USE_MIPP
    for (int r = 0; r < height; ++r)
    {
        int c = 0;
        // not aligned, which will happen because we move 1 bytes a time for spreading
        while (reinterpret_cast<unsigned long long>(src + c) % 16 != 0) {
            dst[c] |= src[c];
            c++;
        }
        // avoid out of bound when can't divid
        // note: can't use c<width !!!
        for (; c <= width-mipp::N<T>(); c+=mipp::N<T>()){
            mipp::Reg<T> src_v((T*)src + c);
            mipp::Reg<T> dst_v((T*)dst + c);

            mipp::Reg<T> res_v = mipp::orb(src_v, dst_v);
            res_v.store((T*)dst + c);
        }
        for(; c<width; c++)
            dst[c] |= src[c];
        // Advance to next row
        src += src_stride;
        dst += dst_stride;
    }
#else
    for (int r = 0; r < height; r++)
    {
        for(int c = 0; c < width; c++) {
            dst[c] |= src[c];
        }
        // Advance to next row
        src += src_stride;
        dst += dst_stride;
    }
#endif
}

template<class T>
static void shiftedMapWithTopLeftPadding(const cv::Mat &src, cv::Mat &dst, int padding)
{
    CV_Assert(src.type() == CV_8U);

    dst = cv::Mat_<T>(cv::Size(src.cols + padding, src.rows + padding));
    for (int r = 0; r < src.rows; r ++) {
        const auto *src_ptr = src.ptr<uchar>(r);
        T *dst_ptr = dst.ptr<T>(r + padding);
        for (int c = 0; c < src.cols; c ++) {
            dst_ptr[c + padding] = src_ptr[c] ? (1 << (src_ptr[c]-1)) : 0;
        }
    }
}

void spread(const cv::Mat &src, cv::Mat &dst, int T, int angle_bin_number)
{
    // Allocate and zero-initialize spread (OR'ed) image
    cv::Mat shifted_src;

    int T_half = T / 2;

    if (angle_bin_number == 8) {
        // 左上角添加 padding
        shiftedMapWithTopLeftPadding<uchar>(src, shifted_src, T_half);

        dst = cv::Mat_<uchar>::zeros(shifted_src.size());

        const int src_step = static_cast<const int>(shifted_src.step1());
        const int dst_step = static_cast<const int>(dst.step1());
        uchar *dst_ptr = dst.ptr<uchar>();
        for (int r = 0; r < T; ++r)
        {
            for (int c = 0; c < T; ++c)
            {
                orUnaligned(&shifted_src.at<uchar>(r, c), src_step, dst_ptr, dst_step,
                            shifted_src.cols - c, shifted_src.rows - r);
            }
        }
        // 右下角减除 padding
        dst = dst(cv::Rect(0, 0, src.cols, src.rows));
    } else if (angle_bin_number == 16) {
        // 左上角添加 padding
        shiftedMapWithTopLeftPadding<ushort>(src, shifted_src, T_half);

        dst = cv::Mat_<ushort>::zeros(shifted_src.size());

        const int src_step = static_cast<const int>(shifted_src.step1());
        const int dst_step = static_cast<const int>(dst.step1());
        ushort *dst_ptr = dst.ptr<ushort>();
        for (int r = 0; r < T; ++r)
        {
            for (int c = 0; c < T; ++c)
            {
                orUnaligned(&shifted_src.at<ushort>(r, c), src_step, dst_ptr, dst_step,
                            shifted_src.cols - c, shifted_src.rows - r);
            }
        }
        // 右下角减除 padding
        dst = dst(cv::Rect(0, 0, src.cols, src.rows));
    } else if (angle_bin_number == 32) {
        // 左上角添加 padding
        shiftedMapWithTopLeftPadding<int>(src, shifted_src, T_half);

        dst = cv::Mat_<int>::zeros(shifted_src.size());

        const int src_step = static_cast<const int>(shifted_src.step1());
        const int dst_step = static_cast<const int>(dst.step1());
        int *dst_ptr = dst.ptr<int>();
        for (int r = 0; r < T; ++r)
        {
            for (int c = 0; c < T; ++c)
            {
                orUnaligned(&shifted_src.at<int>(r, c), src_step, dst_ptr, dst_step,
                            shifted_src.cols - c, shifted_src.rows - r);
            }
        }
        // 右下角减除 padding
        dst = dst(cv::Rect(0, 0, src.cols, src.rows));
    } else {
        std::cerr << "Unspported bins number" << std::endl;
        exit(-1);
    }
}

static const unsigned char LUT3 = 3;
CV_DECL_ALIGNED(16)
// 任何一个 8 bit 数都可以拆分为 高 4bit 和 低 4bit
static const unsigned char SIMILARITY_LUT[256] = {
0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4,                 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3,
0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4,           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, LUT3, LUT3, 4, 4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4,           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4,                 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3,
0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3,     0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                             0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                             0, 0, LUT3, LUT3, 4, 4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4,
0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3,     0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4
};


void computeResponseMaps(const cv::Mat &src, std::vector<cv::Mat> &response_maps, int angle_bin_number)
{
    CV_Assert((src.rows * src.cols) % 16 == 0);
    CV_Assert(angle_bin_number == 8);       // 目前只实现了 8 bit

    // Allocate response maps
    response_maps.resize(angle_bin_number);
    for (int i = 0; i < angle_bin_number; ++i)
        response_maps[i].create(src.size(), CV_8U);

    cv::Mat lsb4(src.size(), CV_8U);
    cv::Mat msb4(src.size(), CV_8U);

    for (int r = 0; r < src.rows; ++r)
    {
        const uchar *src_r = src.ptr(r);
        uchar *lsb4_r = lsb4.ptr(r);
        uchar *msb4_r = msb4.ptr(r);

        for (int c = 0; c < src.cols; ++c)
        {
            // Least significant 4 bits of spread image pixel
            lsb4_r[c] = src_r[c] & 15;
            // Most significant 4 bits, right-shifted to be in [0, 16)
            msb4_r[c] = (src_r[c] & 240) >> 4;
        }
    }

    uchar *lsb4_data = lsb4.ptr<uchar>();
    uchar *msb4_data = msb4.ptr<uchar>();

#ifdef USE_MIPP

    bool no_max = true;
    bool no_shuff = true;

#ifdef has_max_int8_t
    no_max = false;
#endif

#ifdef has_shuff_int8_t
    no_shuff = false;
#endif
    // LUT is designed for 128 bits SIMD, so quite triky for others

    // For each of the 8 quantized orientations...
    for (int ori = 0; ori < 8; ++ori) {
        uchar *map_data = response_maps[ori].ptr<uchar>();
        const uchar *lut_low = SIMILARITY_LUT + 32 * ori;

        if (mipp::N<uint8_t>() == 1) { // no SIMD
            for (int i = 0; i < src.rows * src.cols; ++i)
                map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
        } else if (mipp::N<uint8_t>() == 16) { // 128 SIMD, no add base
            const uchar *lut_low = SIMILARITY_LUT + 32 * ori;
            mipp::Reg<uint8_t> lut_low_v((uint8_t *) lut_low);
            mipp::Reg<uint8_t> lut_high_v((uint8_t *) lut_low + 16);

            for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>()) {
                mipp::Reg<uint8_t> low_mask((uint8_t *) lsb4_data + i);
                mipp::Reg<uint8_t> high_mask((uint8_t *) msb4_data + i);

                mipp::Reg<uint8_t> low_res = mipp::shuff(lut_low_v, low_mask);
                mipp::Reg<uint8_t> high_res = mipp::shuff(lut_high_v, high_mask);

                mipp::Reg<uint8_t> result = mipp::max(low_res, high_res);
                result.store((uint8_t *) map_data + i);
            }
        } else if (mipp::N<uint8_t>() == 16 || mipp::N<uint8_t>() == 32
                   || mipp::N<uint8_t>() == 64) { //128 256 512 SIMD
            CV_Assert((src.rows * src.cols) % mipp::N<uint8_t>() == 0);

            uint8_t lut_temp[mipp::N<uint8_t>()] = {0};

            for (int slice = 0; slice < mipp::N<uint8_t>() / 16; slice++) {
                std::copy_n(lut_low, 16, lut_temp + slice * 16);
            }
            mipp::Reg<uint8_t> lut_low_v(lut_temp);

            uint8_t base_add_array[mipp::N<uint8_t>()] = {0};
            for (uint8_t slice = 0; slice < mipp::N<uint8_t>(); slice += 16) {
                std::copy_n(lut_low + 16, 16, lut_temp + slice);
                std::fill_n(base_add_array + slice, 16, slice);
            }
            mipp::Reg<uint8_t> base_add(base_add_array);
            mipp::Reg<uint8_t> lut_high_v(lut_temp);

            for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>()) {
                mipp::Reg<uint8_t> mask_low_v((uint8_t *) lsb4_data + i);
                mipp::Reg<uint8_t> mask_high_v((uint8_t *) msb4_data + i);

                mask_low_v += base_add;
                mask_high_v += base_add;

                mipp::Reg<uint8_t> shuff_low_result = mipp::shuff(lut_low_v, mask_low_v);
                mipp::Reg<uint8_t> shuff_high_result = mipp::shuff(lut_high_v, mask_high_v);

                mipp::Reg<uint8_t> result = mipp::max(shuff_low_result, shuff_high_result);
                result.store((uint8_t *) map_data + i);
            }
        } else {
            for (int i = 0; i < src.rows * src.cols; ++i)
                map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
        }
    }
#else
    for (int ori = 0; ori < angle_bin_number; ++ori) {
        uchar *map_data = response_maps[ori].ptr<uchar>();
        const uchar *lut_low = SIMILARITY_LUT + 32 * ori;

        for (int i = 0; i < src.rows * src.cols; ++i)
            map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
    }
#endif
}

//
void linearize(const cv::Mat &response_map, cv::Mat &linearized, int T)
{
    CV_Assert(response_map.rows % T == 0);
    CV_Assert(response_map.cols % T == 0);

    // linearized has T^2 rows, where each row is a linear memory
    int mem_width = response_map.cols / T;
    int mem_height = response_map.rows / T;
    linearized.create(T * T, mem_width * mem_height, CV_8U);

    // Outer two for loops iterate over top-left T^2 starting pixels
    int index = 0;
    for (int r_start = 0; r_start < T; ++r_start)
    {
        for (int c_start = 0; c_start < T; ++c_start)
        {
            uchar *memory = linearized.ptr(index);
            ++index;

            // Inner two loops copy every T-th pixel into the linear memory
            for (int r = r_start; r < response_map.rows; r += T)
            {
                const uchar *response_data = response_map.ptr(r);
                for (int c = c_start; c < response_map.cols; c += T)
                    *memory++ = response_data[c];
            }
        }
    }
}


const uchar* accessLinearMemory(const std::vector<cv::Mat>& linear_memories,
                                const Feature& f, int T, int Width_TTBlock)
{
    // Retrieve the TxT grid of linear memories associated with the feature label
    const cv::Mat& memory_grid = linear_memories[f.angle_quantized];
    CV_DbgAssert(memory_grid.rows == T * T);
    CV_DbgAssert(f.x >= 0);
    CV_DbgAssert(f.y >= 0);
    // The LM we want is at (x%T, y%T) in the TxT grid (stored as the rows of memory_grid)
    int grid_x = f.x % T;
    int grid_y = f.y % T;
    int grid_index = grid_y * T + grid_x;
    CV_DbgAssert(grid_index >= 0);
    CV_DbgAssert(grid_index < memory_grid.rows);
    const unsigned char* memory = memory_grid.ptr(grid_index);
    // Within the LM, the feature is at (x/T, y/T). W is the "width" of the LM, the
    // input image width decimated by T.
    int lm_x = f.x / T;
    int lm_y = f.y / T;
    int lm_index = lm_y * Width_TTBlock + lm_x;
    CV_DbgAssert(lm_index >= 0);
    CV_DbgAssert(lm_index < memory_grid.cols);
    return memory + lm_index;
}