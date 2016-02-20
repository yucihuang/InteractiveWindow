/**
* imLab Copyright 2015 All rights reserved
*
*/
#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "svm.h"

class SmileRecognizer {
public:
	SmileRecognizer();

	~SmileRecognizer();

	double Recognize(const cv::Mat faceImage);

private:
	const int IMAGE_WIDTH = 100;
	const int IMAGE_HEIGHT = 100;
	
	const int FEATURE_DIMENSION = 5184;

	const int HOG_WINDOW_WIDTH = 30;
	const int HOG_WINDOW_HEIGHT = 30;
	const int HOG_BLOCK_WIDTH = 30;
	const int HOG_BLOCK_HEIGHT = 30;
	const int HOG_BLOCKSTRIDE_WIDTH = 9;
	const int HOG_BLOCKSTRIDE_HEIGHT = 9;
	const int HOG_CELL_WIDTH = 10;
	const int HOG_CELL_HEIGHT = 10;
	const int HOG_NUMBER_OF_BINS = 9;

	const char *MODEL_FILENAME = "linear.model";

	cv::HOGDescriptor *featureDescriptor;
	struct svm_model *recognitionModel;
};