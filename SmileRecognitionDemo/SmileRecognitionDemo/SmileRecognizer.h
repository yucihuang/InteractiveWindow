/**
* Created by Yu-Ci Huang <j810326@gmail.com> on 2016/2/21.
* Copyright(c) 2015 imLab. All rights reserved.
*
*/

#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "svm.h"

class SmileRecognizer {
public:
	/**
	* Constructor that initialize featureDescriptor and recognitionModel.
	*
	*/
	SmileRecognizer();

	~SmileRecognizer();

	/**
	* Return recognition value of the smile expression of the input face image.
	* To ensure the result is correct, the faceImage must be cropped that it only contain a face.
	* The return value is depended on the model.
	* For classification model, it return 1.0 for smile and -1.0 for nonsmile.
	* For regression model, it return the confidence value in [0, 1], 1 is the strongest.
	*
	* @param faceImage	a OpenCV Mat object for the face image.
	* @return recognition value.
	*
	*/
	double Recognize(const cv::Mat faceImage);

private:
	// SmileRecognizer constants

	// Recognition image size
	const int IMAGE_WIDTH = 100;
	const int IMAGE_HEIGHT = 100;
	
	const int FEATURE_DIMENSION = 5184;

	// HOG parameter
	const int HOG_WINDOW_WIDTH = 30;
	const int HOG_WINDOW_HEIGHT = 30;
	const int HOG_BLOCK_WIDTH = 30;
	const int HOG_BLOCK_HEIGHT = 30;
	const int HOG_BLOCKSTRIDE_WIDTH = 9;
	const int HOG_BLOCKSTRIDE_HEIGHT = 9;
	const int HOG_CELL_WIDTH = 10;
	const int HOG_CELL_HEIGHT = 10;
	const int HOG_NUMBER_OF_BINS = 9;

	// LIBSVM model file
	const char *MODEL_FILENAME = "linear.model";

	cv::HOGDescriptor *featureDescriptor;
	struct svm_model *recognitionModel;
};