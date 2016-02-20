/**
* imLab Copyright 2015 All rights reserved
*
*/

#include "SmileRecognizer.h"

#include <iostream>
#include <vector>
#include "opencv2/imgproc.hpp"

SmileRecognizer::SmileRecognizer()
{
	featureDescriptor = new cv::HOGDescriptor(cv::Size(HOG_WINDOW_WIDTH, HOG_WINDOW_HEIGHT),
												cv::Size(HOG_BLOCK_WIDTH, HOG_BLOCK_HEIGHT),
												cv::Size(HOG_BLOCKSTRIDE_WIDTH, HOG_BLOCKSTRIDE_HEIGHT),
												cv::Size(HOG_CELL_WIDTH, HOG_CELL_HEIGHT), HOG_NUMBER_OF_BINS);

	recognitionModel = svm_load_model(MODEL_FILENAME);

	if (!recognitionModel)
		std::cerr << "Cannot read model file";
}

SmileRecognizer::~SmileRecognizer()
{
	delete featureDescriptor;
	delete recognitionModel;
}

double SmileRecognizer::Recognize(const cv::Mat faceImage) {
	if (faceImage.empty()) {
		return -1;
	}

	cv::Mat recognitionImage;
	cv::cvtColor(faceImage, recognitionImage, CV_BGR2GRAY);
	cv::resize(recognitionImage, recognitionImage, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));

	std::vector<float> descriptorValue;
	featureDescriptor->compute(recognitionImage, descriptorValue);

	struct svm_node *svmNode = new svm_node[FEATURE_DIMENSION];
	for (int i = 0; i < FEATURE_DIMENSION; i++){
		svmNode[i].value = descriptorValue[i];
		svmNode[i].index = i;
	}

	return svm_predict(recognitionModel, svmNode);
}