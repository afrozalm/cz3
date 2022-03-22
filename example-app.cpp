#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/features2d.hpp>
#include <typeinfo>

/**
 * @file Example-app.cpp
 *
 * @brief Loading and running R2D2 keypoint descriptor in C++
 *
 * @author Afroz Alam
 * Contact: afrozalm@berkeley.edu
 * @author Seth Zhao
 * Contact: sethzhao506@berkeley.edu
 *
 */

int main(int argc, const char* argv[]) {
 	if (argc != 2) {
    	std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    	return -1;
  	}
	// Deserialize the ScriptModule from a file using torch::jit::load().
    std::shared_ptr<torch::jit::script::Module> model_;
	try {
		model_ = torch::jit::load(argv[1]);
		model_->to(torch::kCUDA);
	}
	catch (const c10::Error& e) {
    	std::cerr << "error loading the model\n";
    	return -1;
  	}

	assert(model_ != nullptr);
	std::cout << "ok\n";
	cv::VideoCapture camera(1);
	if (!camera.isOpened()) {
		std::cerr << "ERROR: Could not open camera" << std::endl;
		return 1;
	}
	// cv::namedWindow("CAM", CV_WINDOW_AUTOSIZE);

	// this will contain the image from the webcam
	cv::Mat frame, kpt_frame;

	while (1) {
		// capture the next frame from the webcam
		std::vector<torch::jit::IValue> inputs;
		camera >> frame;

        torch::Tensor img = torch::from_blob(frame.data, { 1, frame.rows, frame.cols, 1 }, torch::kByte).to(torch::kCUDA);
        img = img.permute({ 0, 3, 1, 2 });
		img = img.toType(torch::kFloat);
		img = img.div(255);
        inputs.push_back(img);

        auto output = model_->forward(inputs).toTuple();

		// std::cout << "forward done\n";
        auto pts  = output->elements()[0].toTensor().squeeze().to(torch::kCPU);
        auto desc = output->elements()[1].toTensor().squeeze().to(torch::kCPU);

        cv::Mat pts_mat(cv::Size(3, pts.size(0)), CV_32FC1, pts.data<float>());
        // cv::Mat desc_mat(cv::Size(32, pts.size(0)), CV_8UC1, desc.data<unsigned char>());

        std::vector<cv::KeyPoint> kpts;
        // cv::Mat descriptors;

        for (int i = 0; i < pts_mat.rows; i++) {
			cv::KeyPoint kpt;
			kpt.pt.x = pts_mat.at<float>(i, 0);
			kpt.pt.y = pts_mat.at<float>(i, 1);
			kpt.size = 10; // TODO - what should this be? affects geometry check
			kpts.push_back(kpt);
		}

        cv::imshow("Camera", frame);
        cv::drawKeypoints(frame, kpts, kpt_frame, cv::Scalar(127, 200, 10), cv::DrawMatchesFlags::DEFAULT);

		cv::imshow("Keypoints", kpt_frame);

		// show the image on the window
		// wait (10ms) for a key to be pressed
		if (cv::waitKey(10) >= 0)
			break;
	}
	return 0;
}
