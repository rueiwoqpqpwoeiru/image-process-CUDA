
#include <iostream>
#include "Basic.h"
#include "Filter.h"
#include "Warp.h"

int main(int argc, char* argv[]) {
    argv[1] = "test_img.tif";
    argv[2] = ".";
    unit_test_Basic(argc, argv);
    unit_test_Filter(argc, argv);
    unit_test_Warp(argc, argv);
    return 0;
};
