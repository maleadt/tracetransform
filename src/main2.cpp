
int main() {
	cv::Mat sinogram(
		6,
		4,
		CV_64FC1
	);

	sinogram.at<double>(0, 0) = 2;
	sinogram.at<double>(0, 1) = 1;
	sinogram.at<double>(0, 2) = 1;
	sinogram.at<double>(0, 3) = 8;

	sinogram.at<double>(1, 0) = 5;
	sinogram.at<double>(1, 1) = 2;
	sinogram.at<double>(1, 2) = 2;
	sinogram.at<double>(1, 3) = 6;

	sinogram.at<double>(2, 0) = 7;
	sinogram.at<double>(2, 1) = 4;
	sinogram.at<double>(2, 2) = 3;
	sinogram.at<double>(2, 3) = 4;

	sinogram.at<double>(3, 0) = 2;
	sinogram.at<double>(3, 1) = 9;
	sinogram.at<double>(3, 2) = 5;
	sinogram.at<double>(3, 3) = 3;

	sinogram.at<double>(4, 0) = 3;
	sinogram.at<double>(4, 1) = 6;
	sinogram.at<double>(4, 2) = 10;
	sinogram.at<double>(4, 3) = 2;

	sinogram.at<double>(5, 0) = 1;
	sinogram.at<double>(5, 1) = 2;
	sinogram.at<double>(5, 2) = 2;
	sinogram.at<double>(5, 3) = 1;

	for (int i = 0; i < sinogram.rows; i++) {
		for (int j = 0; j < sinogram.cols; j++) {
			std::cerr << sinogram.at<double>(i, j) << "\t";
		}
		std::cerr << "\n";
	}
	std::cerr << "---------------------------------------------------" << std::endl;

	cv::Mat circus = getCircusFunction(
		sinogram,
		new PFunctionalHermite<double>(2)
	);

	for (int i = 0; i < circus.rows; i++) {
		for (int j = 0; j < circus.cols; j++) {
			std::cerr << circus.at<double>(i, j) << "\t";
		}
		std::cerr << "\n";
	}
	std::cerr << std::flush;
}

