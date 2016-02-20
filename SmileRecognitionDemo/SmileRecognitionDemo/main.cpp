/**
* Created by Yu-Ci Huang <j810326@gmail.com> on 2016/2/21.
* Copyright(c) 2015 imLab. All rights reserved.
*
*/

#include "SmileRecognizer.h"
#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"

int main() {
	// read test images
	cv::Mat testImage[2];
	testImage[0] = cv::imread("smile.jpg");
	testImage[1] = cv::imread("nonsmile.jpg");

	// print out result
	SmileRecognizer smileRecognizer;
	std::cout << smileRecognizer.Recognize(testImage[0]) << std::endl;
	std::cout << smileRecognizer.Recognize(testImage[1]) << std::endl;

	system("pause");
	return 0;
}