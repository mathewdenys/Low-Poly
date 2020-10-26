#include <iostream>
#include <array>  // for std::array
#include <vector> // for std::vector
#include <ctime>  // for std::time()

#include <opencv2/core/mat.hpp>  // for basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgcodecs.hpp> // for reading and writing
#include <opencv2/imgproc.hpp>   // for GaussianBlur

// Calculate the cumulative sums of a std::vector of integers
// e.g. {1,2,3,-2} -> {1,3,6,4}
std::vector<int> calculateCumulativeSum(std::vector<int> input)
{
	std::vector<int> cumulativeSum;
	cumulativeSum.resize(input.size());
	cumulativeSum[0] = input[0];
	for(int i=1; i<input.size(); i++)
		cumulativeSum[i] = input[i] + cumulativeSum[i-1];
	return cumulativeSum;
}

// Generate a random number between min and max (inclusive)
// Assumes std::srand() has already been called
// Assumes max - min <= RAND_MAX
// From: https://www.learncpp.com/cpp-tutorial/59-random-number-generation/
int getRandomNumber(int min, int max)
{
	static constexpr double fraction { 1.0 / (RAND_MAX + 1.0) }; // for normalising random numbers to [0,1); static so it is only calculated once
	return min + static_cast<int>((max - min + 1) * (std::rand() * fraction)); // evenly distribute the random number across our range
}

// Find the index of the ceiling of r in arr
// e.g. findCeil({1,4,7,10},2) -> 4
// Returns -1 if r > max(arr)
// Assumes arr.size() actually corresponds to the number of values assigned to arr
// From (with modification): https://www.geeksforgeeks.org/random-number-generator-in-arbitrary-probability-distribution-fashion/
int findCeil(std::vector<int> arr, int r)
{
	int mid;
	int l = 0;
	int h = arr.size() - 1;
	while (l < h)
	{
		mid = (l+h)/2;
		(r > arr[mid]) ? (l = mid + 1) : (h = mid);
	}
	return (arr[l] >= r) ? l : -1;
}

// Returns 'n' random value from the vector 'values' according to the distribution given by the vector 'freqs'
template<typename T>
std::vector<T> randomSelectionFromDistribution(std::vector<T> values, std::vector<int> freqs, int n)
{
	std::vector<int> freqCumSum = calculateCumulativeSum(freqs);
	std::vector<T>   outputVals;
	outputVals.reserve(n);
	for (int i=0; i<n; i++)
	{
		int rand = getRandomNumber(1,freqCumSum.back()); // calculate a random number between 1 and sum(freqs)
		int ceilingIndex = findCeil(freqCumSum,rand);    // determine the index of values to which the random number rounds up to
		outputVals.push_back(values[ceilingIndex]);
	}
	return outputVals;
}

// main() takes one input argument: the name of the input image
int main(int argc, char** argv)
{
	// 1. Load image
	const std::string inputImageName = argv[1];
	cv::Mat imgIn  = cv::imread(inputImageName,cv::IMREAD_COLOR);
	if(imgIn.empty())
	{
		std::cout << "Error opening image: " << inputImageName << '\n';
		return -1;
	}
	if(argc>2) std::cout << "Ignoring additional inputs";

	cv::Size sizeIn = imgIn.size();
	int NpixelsIn = sizeIn.width*sizeIn.height;

	// 1.1 Preprocessing
	cv::GaussianBlur(imgIn, imgIn, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT); // Reduce noise with Gaussian blur (kernel size = 3)
	cv::cvtColor(imgIn, imgIn, cv::COLOR_BGR2GRAY);                           // Convert to greyscale

	// 2. Extract keypoints
	//  - Select some random points
	//  - Some randomisation for aesthetic appeal
	cv::Mat imgFeatures;
	cv::Laplacian(imgIn,imgFeatures,CV_64F); // Edge detection

	std::srand(static_cast<unsigned int>(std::time(nullptr))); // set initial seed value to system clock
	// todo: use randomSelectionFromDistribution to take a random selection of pixel coordiantes from imgIn using imgFeatures as a distribution
	std::vector<int> pixels;
	pixels.resize(NpixelsIn);
	for (int i=0; i<NpixelsIn; i++)
		pixels[i] = i;

	std::vector<uchar> featuresVector;
	if(!imgFeatures.isContinuous())
		imgFeatures = imgFeatures.reshape(1,NpixelsIn);
	featuresVector.assign(imgFeatures.data,imgFeatures.data+imgFeatures.total()); // assumes only one channel (imgFeatures is greyscale)

	// ISSUES:
	//	1) depth of imgFeatures is 6, which corresponds to float64 values (not int) (?)
	// 	2) imgFeatures values can be negative, so I need to make them all positive

	int n = 10; // number of pixels to choose
	//std::vector<int> selectedPixels;
	//selectedPixels.resize(n);
	//selectedPixels = randomSelectionFromDistribution(pixels,featuresVector,n);


	// 3. Create polygons
	//  - Delaunay triangulation

	// 4. Colour polygons

	// 5. Export image
	//cv::Mat imgOut = imgFeatures.clone();
	cv::imwrite("media/output.jpg",imgIn);

	return 0;
}