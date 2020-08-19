#include "response_map.h"
#include <iostream>

int main()
{
    cv::Mat_<uchar> src(8, 8);
    src <<  1, 0, 0, 0, 2, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 4, 0, 0, 0, 3,
            3, 0, 0, 0, 4, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 16, 0,
            0, 27, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 2, 0, 0, 0, 1;

    // test linearize
    {
        std::cout << "Input: " << std::endl;
        std::cout << src << std::endl << std::endl;

        cv::Mat dst;
        linearize(src, dst, 4);
        std::cout << dst << std::endl;

        linearize(src, dst, 8);
        std::cout << dst << std::endl;
    }

    cv::Mat dst;
    linearize(src, dst, 4);
    std::vector<cv::Mat> lm_vec;
    for (int i = 0; i < 8; i++) {
        lm_vec.push_back(dst.clone());
    }

    Feature f(0, 4, 0, 2);
    const uchar* f_ptr = accessLinearMemory(lm_vec, f, 4, src.cols/4);
    std::cout << static_cast<int>(*f_ptr) << std::endl;
    
    Feature f2(6, 5, 0, 0);
    f_ptr = accessLinearMemory(lm_vec, f2, 4, src.cols / 4);
    std::cout << static_cast<int>(*f_ptr) << std::endl;
}
