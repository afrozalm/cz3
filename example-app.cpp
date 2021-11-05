#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <typeinfo>

int main(int argc, const char* argv[]) {

	// Deserialize the ScriptModule from a file using torch::jit::load().
	std::string model_path = "C:/Users/OpenARK/Desktop/r2d2_test/traced_r2d2_WASF_N16.pt";
	//model_path = "C:/Users/OpenARK/Desktop/openark_dependencies/okvis-master/traced_superpoint_model_cuda.pt";
	std::unique_ptr<torch::jit::script::Module> model_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(model_path)); 
	//std::unique_ptr<torch::jit::script::Module> superpoint_ = std::make_unique<torch::jit::script::Module>(torch::jit::load("C:/Users/OpenARK/Desktop/openark_dependencies/okvis-master/traced_superpoint_model_cuda.pt"));
	model_->to(torch::kCUDA);
	assert(model_ != nullptr);

	std::cout << "ok\n";
	cv::VideoCapture camera(1);
	if (!camera.isOpened()) {
		std::cerr << "ERROR: Could not open camera" << std::endl;
		return 1;
	}
	cv::namedWindow("Webcam", CV_WINDOW_AUTOSIZE);

	// this will contain the image from the webcam
	cv::Mat frame, grayFrame, kpt_frame;


	while (1) {
		// capture the next frame from the webcam
		std::cout << "looping\n";
		std::vector<torch::jit::IValue> inputs;

		camera >> frame;
		cv::cvtColor(frame, grayFrame, CV_BGR2GRAY);

		// show the image on the window
		//cv::imshow("Webcam", grayFrame);

		// pass gray image through r2d2

		//torch::Tensor img_tensor = torch::from_blob(grayFrame.data, { 1, grayFrame.rows, grayFrame.cols, 1 }, torch::kByte).to(torch::kCUDA);
		torch::Tensor img_tensor = torch::from_blob(frame.data, { 1, frame.rows, frame.cols, 3 }, torch::kByte).to(torch::kCUDA);
		img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
		img_tensor = img_tensor.toType(torch::kFloat);
		img_tensor = img_tensor.div(255);
		inputs.push_back(img_tensor);

		// execute the model
		auto output = model_->forward(inputs);

		std::cout << "forward done\n";
		//::cout << typeid(output).name() << "\n";

		torch::Tensor t0 = torch::squeeze(((torch::Tensor) output.toTensorList()[0]).to(torch::kCUDA)); // descriptors 128, W, H
		torch::Tensor t1 = torch::squeeze(((torch::Tensor) output.toTensorList()[1]).to(torch::kCUDA)); // reliability W, H
		torch::Tensor t2 = torch::squeeze(((torch::Tensor) output.toTensorList()[2]).to(torch::kCUDA)); // repeatibility W, H
		std::cout << "unsq: " << t0.sizes() << ", " << t1.sizes() << "\n";
		torch::Tensor score = torch::multiply(t1, t2);
		torch::Tensor mask = score > 0.98;
		std::cout << "score sizes" << score.sizes() << " mask : " << mask.sizes() << "\n";

		int H = t1.sizes()[0], W = t1.sizes()[1];
		torch::Tensor idxs = torch::arange(W*H).to(torch::kCUDA);
		std::cout << idxs.view({ H,W }).sizes() << " :<- idxs shape\n";
		torch::Tensor kpts_ = torch::masked_select( idxs.view({H, W}), mask);
		std::cout << "kpts_ size: " << kpts_.sizes() << "\n";

		std::vector<cv::KeyPoint> kpts;
		for (int i = 0; i < kpts_.sizes()[0]; i++) {
			cv::KeyPoint kpt;
			kpt.pt.y = kpts_[i].item<int>()/W;
			kpt.pt.x = kpts_[i].item<int>()%W;
			kpt.size = 10; // TODO - what should this be? affects geometry check
			kpts.push_back(kpt);
		}

		cv::drawKeypoints(frame, kpts, kpt_frame, cv::Scalar(127, 200, 10), 4);

		cv::imshow("Keypoints", kpt_frame);

		// wait (10ms) for a key to be pressed
		if (cv::waitKey(100) >= 0)
			break;
	}
	return 0;
}