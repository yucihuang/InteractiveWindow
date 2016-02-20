/**
* Created by Yu-Ci Huang <j810326@gmail.com> on 2016/2/21.
* Copyright(c) 2015 imLab. All rights reserved.
*
*/

#include "SmileRecognizer.h"

#include <iostream>
#include <vector>
#include "opencv2/imgproc.hpp"

SmileRecognizer::SmileRecognizer()
{
	// initialize HOG feature descirptor
	featureDescriptor = new cv::HOGDescriptor(cv::Size(HOG_WINDOW_WIDTH, HOG_WINDOW_HEIGHT),
												cv::Size(HOG_BLOCK_WIDTH, HOG_BLOCK_HEIGHT),
												cv::Size(HOG_BLOCKSTRIDE_WIDTH, HOG_BLOCKSTRIDE_HEIGHT),
												cv::Size(HOG_CELL_WIDTH, HOG_CELL_HEIGHT), HOG_NUMBER_OF_BINS);

	// initialize SVM recognition model
	recognitionModel = svm_load_model(MODEL_FILENAME);

	if (!recognitionModel)
		std::cerr << "Cannot Read Model File";
}

SmileRecognizer::~SmileRecognizer()
{
	delete featureDescriptor;
	delete recognitionModel;
}

double SmileRecognizer::Recognize(const cv::Mat faceImage) {
	// if image is empty return false
	if (faceImage.empty()) {
		return -1;
	}

	// preprocess input image
	cv::Mat recognitionImage;
	cv::cvtColor(faceImage, recognitionImage, CV_BGR2GRAY);
	cv::resize(recognitionImage, recognitionImage, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));

	// extract features
	std::vector<float> descriptorValue;
	featureDescriptor->compute(recognitionImage, descriptorValue);

	// copy the descriptor value to LIBSVM structure
	struct svm_node *svmNode = new svm_node[FEATURE_DIMENSION];
	for (int i = 0; i < FEATURE_DIMENSION; ++i) {
		svmNode[i].value = descriptorValue[i];
		svmNode[i].index = i;
	}

	// do recognition
	double reconitionValue = svm_predict(recognitionModel, svmNode);

	// release memory
	recognitionImage.release();
	descriptorValue.clear();
	delete svmNode;

	return reconitionValue;
}