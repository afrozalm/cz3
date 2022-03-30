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

void nms(cv::Mat det, std::vector<cv::KeyPoint>& pts, int border, int dist_thresh,
         int img_width, int img_height){

    std::vector<cv::Point2f> pts_raw;

    for (int i = 0; i < det.rows; i++){

        int u = (int) det.at<float>(i, 0);
        int v = (int) det.at<float>(i, 1);
        // float conf = det.at<float>(i, 2);

        pts_raw.push_back(cv::Point2f(u, v));
    }

    cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
    cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

    grid.setTo(0);
    inds.setTo(0);

    for (int i = 0; i < pts_raw.size(); i++)
    {
        int uu = (int) pts_raw[i].x;
        int vv = (int) pts_raw[i].y;

        grid.at<char>(vv, uu) = 1;
        inds.at<unsigned short>(vv, uu) = i;
    }

    cv::copyMakeBorder(grid, grid, dist_thresh, dist_thresh, dist_thresh, dist_thresh, cv::BORDER_CONSTANT, 0);

    for (int i = 0; i < pts_raw.size(); i++)
    {
        int uu = (int) pts_raw[i].x + dist_thresh;
        int vv = (int) pts_raw[i].y + dist_thresh;

        if (grid.at<char>(vv, uu) != 1)
            continue;

        for(int k = -dist_thresh; k < (dist_thresh+1); k++)
            for(int j = -dist_thresh; j < (dist_thresh+1); j++)
            {
                if(j==0 && k==0) continue;

                grid.at<char>(vv + k, uu + j) = 0;

            }
        grid.at<char>(vv, uu) = 2;
    }


    for (int v = 0; v < (img_height + dist_thresh); v++){
        for (int u = 0; u < (img_width + dist_thresh); u++)
        {
            if (u -dist_thresh>= (img_width - border) || u-dist_thresh < border || v-dist_thresh >= (img_height - border) || v-dist_thresh < border)
            continue;

            if (grid.at<char>(v,u) == 2)
            {
                int select_ind = (int) inds.at<unsigned short>(v-dist_thresh, u-dist_thresh);
                pts.push_back(cv::KeyPoint(pts_raw[select_ind].x, pts_raw[select_ind].y, 1.0f));
            }
        }
    }
}


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
	cv::Mat frame, img_frame, kpt_frame;

    int img_width = 320;
    int img_height = 240;

	while (1) {
		// capture the next frame from the webcam
		std::vector<torch::jit::IValue> inputs;
		camera >> frame;

        cv::resize(frame, img_frame, cv::Size(img_width, img_height));
        torch::Tensor img = torch::from_blob(img_frame.data, { 1, img_frame.rows, img_frame.cols, 1 }, torch::kByte).to(torch::kCUDA);
        img = img.permute({ 0, 3, 1, 2 });
        img = img.toType(torch::kFloat32);
        img = img.div(255);
        inputs.push_back(img);

        auto output = model_->forward(inputs).toTuple();

		// std::cout << "forward done\n";
        auto pts  = output->elements()[0].toTensor().squeeze().to(torch::kCPU);
        auto desc = output->elements()[1].toTensor().squeeze().to(torch::kCPU);

        cv::Mat pts_mat(cv::Size(3, pts.size(0)), CV_32FC1, pts.data<float>());
        // cv::Mat desc_mat(cv::Size(32, pts.size(0)), CV_8UC1, desc.data<unsigned char>());

        std::vector<cv::KeyPoint> kpts;
        // nms(pts_mat, kpts, border, dist_thresh, img_width, img_height);

        // cv::Mat descriptors;

        for (int i = 0; i < pts_mat.rows; i++) {
		 	cv::KeyPoint kpt;
		 	kpt.pt.x = pts_mat.at<float>(i, 0);
		 	kpt.pt.y = pts_mat.at<float>(i, 1);
		 	kpt.size = 1.0f; // TODO - what should this be? affects geometry check
             if (kpt.pt.x > img_width || kpt.pt.y > img_height)
                std::cout << "x: " << kpt.pt.x << ", y: " << kpt.pt.y << "\n";
		 	kpts.push_back(kpt);
		 }

        //cv::imshow("Camera", frame);
        cv::drawKeypoints(frame, kpts, kpt_frame, cv::Scalar(127, 200, 10), cv::DrawMatchesFlags::DEFAULT);

		cv::imshow("Keypoints", kpt_frame);

		// show the image on the window
		// wait (10ms) for a key to be pressed
		if (cv::waitKey(10) >= 0)
			break;
	}
	return 0;
}
