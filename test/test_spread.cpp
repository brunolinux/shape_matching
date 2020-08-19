//
// Created by bruno on 8/15/20.
//

#include "response_map.h"
#include <iostream>

int main()
{
    cv::Mat_<uchar> src(6, 8);
    src << 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 0;

    std::cout << "Input: " << std::endl;
    std::cout << src << std::endl << std::endl;

    cv::Mat result;
    {
        spread(src, result, 3, 8);
        std::cout << "Bin Number 8: " << std::endl;
        std::cout << result << std::endl;
        std::cout << "matice type:" << result.type() << std::endl;
    }

    {
        spread(src, result, 3, 16);
        std::cout << "Bin Number 16: " << std::endl;
        std::cout << result << std::endl;
        std::cout << "matice type:" << result.type() << std::endl;
    }

    {
        spread(src, result,3, 32);
        std::cout << "Bin Number 32: " << std::endl;
        std::cout << result << std::endl;
        std::cout << "matice type:" << result.type() << std::endl;
        std::cout << result.at<int>(1, 1) << std::endl;
    }

    {
        spread(src, result, 4, 8);
        std::cout << "Bin Number 8: " << std::endl;
        std::cout << result << std::endl;
        std::cout << "matice type:" << result.type() << std::endl;
    }
    return 0;
}
