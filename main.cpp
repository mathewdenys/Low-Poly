#include <iostream>
#include <array>  // for std::array
#include <vector> // for std::vector
#include <ctime>  // for std::time()
#include <numeric>// for std::iota()

#include <opencv2/core/mat.hpp>  // for basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgcodecs.hpp> // for reading and writing
#include <opencv2/imgproc.hpp>   // for GaussianBlur

// Calculate the cumulative sums of a std::vector of integers
// e.g. {1,2,3,-2} -> {1,3,6,4}
std::vector<int> calculateCumulativeSum(std::vector<int> input)
{
	std::vector<int> cumulativeSum(input.size());
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
	if(argc>2)
		std::cout << "Ignoring additional inputs";

	cv::Size sizeIn = imgIn.size();				// Image dimensions
	int NpixelsIn = sizeIn.width*sizeIn.height; // Total number of pixels (for later use)

	// 1.1 Preprocessing
	cv::GaussianBlur(imgIn, imgIn, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT); // Reduce noise with Gaussian blur (kernel size = 3)
	cv::cvtColor(imgIn, imgIn, cv::COLOR_BGR2GRAY);                           // Convert to greyscale

	// 2. Extract keypoints
	// 2.1 Edge detection
	if (!imgIn.depth()==0)
		std::cout << "Input image has depth!=0; this may need to be addressed";
	cv::Mat imgFeatures;
	int depth = CV_16S; 						  // Depth of the output image (input assumed to be CV_8U; don't want overflow; laplacian can be negative)
	cv::Laplacian(imgIn,imgFeatures,depth); 	  // Edge detection
	cv::convertScaleAbs(imgFeatures,imgFeatures); // Take absolute value of values and convert back to 8 bits

	// 2.2 Select pixels pseudo-randomly
	std::vector<int> pixels(NpixelsIn);					// Vector for storing pixel numbers
	std::iota(std::begin(pixels), std::end(pixels), 0); // Fill with 0, 1, ..., NpixelsIn

	std::vector<int> featuresVector;
	if(!imgFeatures.isContinuous())
		imgFeatures = imgFeatures.reshape(1,NpixelsIn);
	featuresVector.assign(imgFeatures.data,imgFeatures.data+imgFeatures.total()); // Assumes only one channel (imgFeatures is greyscale); uchars are cast to ints

	int n = 500; // number of pixels to choose
	std::vector<int> selectedPixels(n);
	std::srand(static_cast<unsigned int>(std::time(nullptr))); // Set initial seed value to system clock
	selectedPixels = randomSelectionFromDistribution(pixels,featuresVector,n);

	// 2.3 Add some randomisation / move pixels slightly (todo)

	// 2.4 Visualise selected points
	cv::Point center;
	for (int i=0;i<n;i++)
	{
		center = cv::Point{selectedPixels[i]%sizeIn.width,selectedPixels[i]/sizeIn.width};
		circle(imgFeatures,center,2,CV_RGB(255,255,255),3);
	}

	// 3. Create polygons
	//  - Delaunay triangulation

	// 4. Colour polygons

	// 5. Export image
	//cv::Mat imgOut = imgFeatures.clone();
	cv::imwrite("media/output.jpg",imgFeatures);

	return 0;
}