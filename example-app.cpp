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

void NonMaximalSuppression( cv::Mat kpts_mat, \
                            cv::Mat desc_mat, \
                            std::vector<cv::KeyPoint>& _keypoints, \
                            cv::Mat& descriptors, \
                            int border, \
                            int dist_threshold, \
                            int img_width, \
                            int img_height)
{
    cv::Mat kpt_grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
    cv::Mat kpt_index = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

    kpt_grid.setTo(0);
    kpt_index.setTo(0);

    for (int i = 0; i < kpts_mat.rows; i++){

        int u = (int) kpts_mat.at<float>(i, 0);
        int v = (int) kpts_mat.at<float>(i, 1);

        // kpts_raw[i] = cv::Point2f(u, v);
        kpt_grid.at<char>(v, u) = 1;
        kpt_index.at<unsigned short>(v, u) = i;
    }

    cv::copyMakeBorder( kpt_grid, kpt_grid, dist_threshold, dist_threshold,
                        dist_threshold, dist_threshold, cv::BORDER_CONSTANT, 0 );

    for ( int i = 0; i < kpts_mat.rows; ++i )
    {
        int u = (int) kpts_mat.at<float>(i, 0) + dist_threshold;
        int v = (int) kpts_mat.at<float>(i, 1) + dist_threshold;


        if ( kpt_grid.at<char>(v, u) != 1 )
            continue;

        for ( int j = -dist_threshold; j <= dist_threshold; ++j )
            for ( int k = -dist_threshold; k <= dist_threshold; ++k )
            {
                if ( j == 0 && k == 0)
                    continue;

                kpt_grid.at<char>(v + j, u + k) = 0;
            }

        kpt_grid.at<char>(v, u) = 2;
    }

    std::vector<int> valid_idxs;
    for ( int u = dist_threshold; u < img_width - dist_threshold; ++u )
        for ( int v = dist_threshold; v < img_height - dist_threshold; ++v )
        {
            if (kpt_grid.at<char>(v, u) == 2) {
                int idx = (int) kpt_index.at<unsigned short>(v-dist_threshold, u-dist_threshold);
                int x = (int) kpts_mat.at<float>(idx, 0);
                int y = (int) kpts_mat.at<float>(idx, 1);
                _keypoints.push_back( cv::KeyPoint( x, y, 1.0f ) );
                valid_idxs.push_back(idx);
            }
        }

    descriptors.create(valid_idxs.size(), 32, CV_8U);

    for ( int i = 0; i < valid_idxs.size(); ++i )
        for ( int j = 0; j < 32; ++j )
        {
            descriptors.at<unsigned char>(i, j) = desc_mat.at<unsigned char>(valid_idxs[i], j);
        }
}


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
    int img_width = 640;
    int img_height = 480;

	cv::Mat frame, img_frame, kpt_frame, tmp_frame;

    int border = 16;
    int dist_thresh = 8;

	while (1) {
		// capture the next frame from the webcam
		std::vector<torch::jit::IValue> inputs;
		camera >> frame;

        cv::resize(frame, tmp_frame, cv::Size(img_width/3, img_height/3));
        cv::hconcat(tmp_frame, tmp_frame, img_frame);
        cv::hconcat(img_frame, tmp_frame, img_frame);

        cv::copyMakeBorder(img_frame, img_frame, 0, 2*img_height/3, 0, 0, cv::BORDER_CONSTANT, 0);

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
        cv::Mat desc_mat(cv::Size(32, pts.size(0)), CV_8UC1, desc.data<unsigned char>());

        std::vector<cv::KeyPoint> kpts;
        cv::Mat descriptors;

        // nms(pts_mat, kpts, border, dist_thresh, img_width, img_height);
        NonMaximalSuppression(pts_mat, desc_mat, kpts, descriptors, border, dist_thresh, img_width, img_height);


        cv::drawKeypoints(frame, kpts, kpt_frame, cv::Scalar(127, 200, 10), cv::DrawMatchesFlags::DEFAULT);

		cv::imshow("Keypoints", kpt_frame);
        cv::imshow("Input Image", img_frame);

		// show the image on the window
		// wait (10ms) for a key to be pressed
		if (cv::waitKey(10) >= 0)
			break;
	}
	return 0;
}
