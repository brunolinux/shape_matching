//
// Created by bruno on 8/15/20.
//

#include "response_map.h"
#include <iostream>

int main() {
    cv::Mat_<uchar> src(8, 8);
    src << 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 3, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0;

    std::cout << "Input: " << std::endl;
    std::cout << src << std::endl << std::endl;

    cv::Mat dst;
    spread(src, dst, 3, 8);
    std::cout << "Bin Number 8: " << std::endl;
    std::cout << dst << std::endl;

    std::vector<cv::Mat> response;
    computeResponseMaps(dst, response);

    for (int i = 0; i < response.size(); i ++) {
        std::cout << "[" << i << "]:\n";
        std::cout << response[i] << std::endl;
    }
}
