//
// Created by bruno on 8/14/20.
//

#include <opencv2/imgproc.hpp>

#include "feature.h"

void hysteresisGradient(const cv::Mat &magnitude, const cv::Mat &angle,
                        cv::Mat_<uchar> &quantized_angle,
                        float threshold, int angle_bin_number)
{
    int N = angle_bin_number;
    // Quantize 360 degree range of orientations into 2*N buckets
    // for stability of horizontal and vertical features.
    cv::Mat_<uchar> quantized_unfiltered;
    angle.convertTo(quantized_unfiltered, CV_8U, 2 * N / 360.0);

    // Mask 2*N buckets into N quantized orientations (neglect direction)
    for (int r = 0; r < angle.rows; ++r)
    {
        uchar *quant_ptr = quantized_unfiltered.ptr<uchar>(r);
        for (int c = 0; c < angle.cols; ++c)
        {
            quant_ptr[c] &= N-1;
        }
    }

#define PATCH_HALFSIZE          (4)
#define PATCH_SIZE              ((2 * PATCH_HALFSIZE) + 1)
#define NEIGHBOR_THRESHOLD      ((PATCH_SIZE * PATCH_SIZE) / 2)


    quantized_angle = cv::Mat_<uchar>::zeros(angle.size());
    // Filter the raw quantized image. Only accept pixels where the magnitude is above some
    // threshold, and there is local agreement on the quantization.
    for (int r = PATCH_HALFSIZE; r < angle.rows - PATCH_HALFSIZE; ++r)
    {
        const float *mag_ptr = magnitude.ptr<float>(r);
        uchar *quantized_angle_ptr = quantized_angle.ptr<uchar>(r);
        for (int c = PATCH_HALFSIZE; c < angle.cols - PATCH_HALFSIZE; ++c)
        {
            if (mag_ptr[c] > threshold)
            {
                // Compute histogram of quantized bins in 3x3 patch around pixel
                std::vector<int> histo(N, 0);
                {
                    uchar *patch_ptr = quantized_unfiltered.ptr<uchar>(r - PATCH_HALFSIZE) + c - PATCH_HALFSIZE;
                    for (int k_r = 0; k_r < PATCH_SIZE; k_r ++) {
                        for (int k_c = 0; k_c < PATCH_SIZE; k_c ++) {
                            histo[*(patch_ptr + k_c)] ++;
                        }
                        patch_ptr += quantized_unfiltered.step1();
                    }
                }

                // Find bin with the most votes from the patch
                int max_votes = 0;
                int index = -1;
                for (int i = 0; i < N; ++i)
                {
                    if (max_votes < histo[i])
                    {
                        index = i;
                        max_votes = histo[i];
                    }
                }

                // Only accept the quantization if majority of pixels in the patch agree
                // 为了避免 0 的歧义，0 表示该点不是显著梯度点，0 同时也表示量化角度索引为 0
                // 因此将索引 + 1
                if (max_votes >= NEIGHBOR_THRESHOLD)
                    quantized_angle_ptr[c] = (1 + index); // (1 << index);
            }
        }
    }
}


void quantizedOrientations(const cv::Mat &src, cv::Mat &magnitude,
                           cv::Mat &angle, cv::Mat_<uchar>& quantized_angle,
                           float threshold, int angle_bin_number)
{
    cv::Mat smoothed;
    // Compute horizontal and vertical image derivatives on all color channels separately
#define GAUSSIAN_KSIZE  (7)
#define SOBEL_KSIZE     (3)

    // For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
    GaussianBlur(src, smoothed, cv::Size(GAUSSIAN_KSIZE, GAUSSIAN_KSIZE), 0, 0, cv::BORDER_REPLICATE);
    if(src.channels() == 1){
        cv::Mat sobel_dx, sobel_dy, sobel_ag;
        Sobel(smoothed, sobel_dx, CV_32F, 1, 0, SOBEL_KSIZE, 1.0, 0.0, cv::BORDER_REPLICATE);
        Sobel(smoothed, sobel_dy, CV_32F, 0, 1, SOBEL_KSIZE, 1.0, 0.0, cv::BORDER_REPLICATE);
        magnitude = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
        phase(sobel_dx, sobel_dy, angle, true);
        hysteresisGradient(magnitude, angle, quantized_angle, threshold * threshold, angle_bin_number);
    }else{
        magnitude.create(src.size(), CV_32F);

        // Allocate temporary buffers
        cv::Size size = src.size();
        cv::Mat sobel_3dx;              // per-channel horizontal derivative
        cv::Mat sobel_3dy;              // per-channel vertical derivative
        cv::Mat sobel_dx(size, CV_32F); // maximum horizontal derivative
        cv::Mat sobel_dy(size, CV_32F); // maximum vertical derivative
        cv::Mat sobel_ag;               // final gradient orientation (unquantized)

        Sobel(smoothed, sobel_3dx, CV_16S, 1, 0, SOBEL_KSIZE, 1.0, 0.0, cv::BORDER_REPLICATE);
        Sobel(smoothed, sobel_3dy, CV_16S, 0, 1, SOBEL_KSIZE, 1.0, 0.0, cv::BORDER_REPLICATE);

        short *ptrx = (short *)sobel_3dx.data;
        short *ptry = (short *)sobel_3dy.data;
        float *ptr0x = (float *)sobel_dx.data;
        float *ptr0y = (float *)sobel_dy.data;
        float *ptrmg = (float *)magnitude.data;

        const int length1 = static_cast<const int>(sobel_3dx.step1());
        const int length2 = static_cast<const int>(sobel_3dy.step1());
        const int length3 = static_cast<const int>(sobel_dx.step1());
        const int length4 = static_cast<const int>(sobel_dy.step1());
        const int length5 = static_cast<const int>(magnitude.step1());
        const int length0 = sobel_3dy.cols * 3;

        for (int r = 0; r < sobel_3dy.rows; ++r)
        {
            int ind = 0;

            for (int i = 0; i < length0; i += 3)
            {
                // Use the gradient orientation of the channel whose magnitude is largest
                int mag1 = ptrx[i + 0] * ptrx[i + 0] + ptry[i + 0] * ptry[i + 0];
                int mag2 = ptrx[i + 1] * ptrx[i + 1] + ptry[i + 1] * ptry[i + 1];
                int mag3 = ptrx[i + 2] * ptrx[i + 2] + ptry[i + 2] * ptry[i + 2];

                if (mag1 >= mag2 && mag1 >= mag3)
                {
                    ptr0x[ind] = ptrx[i];
                    ptr0y[ind] = ptry[i];
                    ptrmg[ind] = (float)mag1;
                }
                else if (mag2 >= mag1 && mag2 >= mag3)
                {
                    ptr0x[ind] = ptrx[i + 1];
                    ptr0y[ind] = ptry[i + 1];
                    ptrmg[ind] = (float)mag2;
                }
                else
                {
                    ptr0x[ind] = ptrx[i + 2];
                    ptr0y[ind] = ptry[i + 2];
                    ptrmg[ind] = (float)mag3;
                }
                ++ind;
            }
            ptrx += length1;
            ptry += length2;
            ptr0x += length3;
            ptr0y += length4;
            ptrmg += length5;
        }

        // Calculate the final gradient orientations
        phase(sobel_dx, sobel_dy, angle, true);
        hysteresisGradient(magnitude, angle, quantized_angle, threshold * threshold, angle_bin_number);
    }
}

cv::Mat createNMSMat(const cv::Mat& src, const cv::Mat& mask, int ksize)
{
    CV_Assert(mask.empty() || src.size() == mask.size());
    CV_Assert(src.type() == CV_32F);

    bool no_mask = mask.empty();
    int half_size = ksize / 2;

    cv::Mat localMaximalMat = cv::Mat::zeros(src.size(), CV_8U);

    // 方法一 (dilate)
    cv::Mat dilated_src;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksize, ksize));
    cv::dilate(src, dilated_src, kernel);

    for (int r = half_size; r < src.rows-half_size; ++r) {
        const uchar *mask_ptr = no_mask ? nullptr : mask.ptr<uchar>(r);
        const float *src_ptr = src.ptr<float>(r);
        const float *dilated_src_ptr = dilated_src.ptr<float>(r);
        uchar *lmax_ptr = localMaximalMat.ptr<uchar>(r);
        for (int c = half_size; c < src.cols-half_size; ++c) {
            if ((no_mask || mask_ptr[c]) && (src_ptr[c] == dilated_src_ptr[c])) {
                lmax_ptr[c] = 1;
            }
        }
    }

    // 方法二
    //localMaximalMat(cv::Rect(half_size, half_size, src.cols - 2*half_size, src.rows - 2*half_size)) = cv::Scalar(1);
//    for (int r = half_size; r < src.rows-half_size; ++r)
//    {
//        const uchar *mask_ptr = no_mask ? nullptr : mask.ptr<uchar>(r);
//        const float *src_ptr = src.ptr<float>(r);
//        uchar *lmax_ptr = localMaximalMat.ptr<uchar>(r);
//        for (int c = half_size; c < src.cols-half_size; ++c)
//        {
//            if (no_mask || mask_ptr[c])
//            {
//                float score = 0;
//                if(localMaximalMat.at<uchar>(r, c) > 0) {       // unchecked otherwise non-local maximal
//                    score = src_ptr[c];
//                    bool is_local_maximal = true;
//
//                    const float *patch_ptr = src.ptr<float>(r-half_size) + c;
//                    for(int r_offset = -half_size; r_offset <= half_size; r_offset++) {
//                        for(int c_offset = -half_size; c_offset <= half_size; c_offset++) {
//                            if(r_offset == 0 && c_offset == 0) continue;
//
//                            if(score < *(patch_ptr-c_offset)) {
//                                score = 0;
//                                is_local_maximal = false;
//                                break;
//                            }
//                        }
//                        if(!is_local_maximal) break;
//                        patch_ptr += src.step1();
//                    }
//
//                    if (is_local_maximal) {
//                        uchar *local_ptr = localMaximalMat.ptr<uchar>(r-half_size) + c;
//                        for(int r_offset = -half_size; r_offset <= half_size; r_offset++){
//                            for(int c_offset = -half_size; c_offset <= half_size; c_offset++){
//                                if(r_offset == 0 && c_offset == 0) continue;
//                                *(local_ptr + c_offset) = 0;
//                            }
//                            local_ptr += localMaximalMat.step1();
//                        }
//                    } else {
//                        lmax_ptr[c] = 0;
//                    }
//                }
//            } else {
//                lmax_ptr[c] = 0;
//            }
//        }
//    }

    return localMaximalMat;
}
