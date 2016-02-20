/**
* imLab Copyright 2015 All rights reserved
*
*/

#include "SmileRecognizer.h"
#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"

int main() {
	cv::Mat testImage[2];
	testImage[0] = cv::imread("smile.jpg");
	testImage[1] = cv::imread("nonsmile.jpg");

	SmileRecognizer smileRecognizer;
	std::cout << smileRecognizer.Recognize(testImage[0]) << std::endl;
	std::cout << smileRecognizer.Recognize(testImage[1]) << std::endl;

	system("pause");
	return 0;
}