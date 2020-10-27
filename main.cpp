#include <iostream> // for std::cout
#include <array>    // for std::array
#include <vector>   // for std::vector
#include <ctime>    // for std::time()
#include <numeric>  // for std::iota()
#include <sstream>  // for std::stringstream

#include <opencv2/core/mat.hpp>  // for basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgcodecs.hpp> // for reading and writing
#include <opencv2/imgproc.hpp>   // for GaussianBlur

#include "delaunator.hpp"   // for Delaunay triangulation

// Calculate the cumulative sums of a std::vector of integers
// e.g. {1,2,3,-2} -> {1,3,6,4}
std::vector<int> calculateCumulativeSum(std::vector<int>& input)
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
int findCeil(std::vector<int>& arr, int r)
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
std::vector<T> randomSelectionFromDistribution(std::vector<T>& values, std::vector<int>& freqs, int n)
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

// main() takes two input arguments: the name of the input image and the number of pixels to randomly select
int main(int argc, char** argv)
{
	// 1. Deal with main() inputs
	// Deal with case of not enough inputs
	if (argc<3)
		std::cout << "Not enough inputs. Expected usage: <input image file name> <number of points to select>";

	// Load image
	const std::string inputImageName{argv[1]};
	cv::Mat imgIn{cv::imread(inputImageName,cv::IMREAD_COLOR)};
	if(imgIn.empty())
	{
		std::cout << "Error opening image: " << inputImageName << '\n';
		return -1;
	}

	// Parse number of pixels to select
	std::stringstream convert{argv[2]};
	int NSelectedPixels{};              // The number of pixels to pseudo-randomly select
	if (!(convert >> NSelectedPixels))  // Do the conversion
	{
		NSelectedPixels = 100; // If conversion fails, set NSelectedPixels to a default value
		std::cout << "Could not interpret second argument as an integer. Taking dafault value of N=" << NSelectedPixels << '\n';
	}

	// Deal with case of too many inputs
	if(argc>3)
		std::cout << "Ignoring additional inputs";

	// 1.1 Preprocess the image
	cv::Size sizeIn = imgIn.size();             // Image dimensions
	int NpixelsIn = sizeIn.width*sizeIn.height; // Total number of pixels (for later use)
	cv::GaussianBlur(imgIn, imgIn, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT); // Reduce noise with Gaussian blur (kernel size = 3)
	cv::cvtColor(imgIn, imgIn, cv::COLOR_BGR2GRAY);                           // Convert to greyscale

	// 2. Extract keypoints
	// 2.1 Edge detection
	if (!imgIn.depth()==0)
		std::cout << "Input image has depth!=0; this may need to be addressed";
	cv::Mat imgFeatures;
	int depth = CV_16S;                           // Depth of the output image (input assumed to be CV_8U; don't want overflow; laplacian can be negative)
	cv::Laplacian(imgIn,imgFeatures,depth);       // Edge detection
	cv::convertScaleAbs(imgFeatures,imgFeatures); // Take absolute value of values and convert back to 8 bits

	// 2.2 Select pixels pseudo-randomly
	std::vector<int> pixels(NpixelsIn);                 // Vector for storing pixel numbers
	std::iota(std::begin(pixels), std::end(pixels), 0); // Fill with 0, 1, ..., NpixelsIn

	std::vector<int> featuresVector;
	if(!imgFeatures.isContinuous())
		imgFeatures = imgFeatures.reshape(1,NpixelsIn);
	featuresVector.assign(imgFeatures.data,imgFeatures.data+imgFeatures.total()); // Assumes only one channel (imgFeatures is greyscale); uchars are cast to ints

	std::vector<int> selectedPixels(NSelectedPixels+4);                                      // Vector large enough to store NSelectedPixels randomly selected pixels, plus the four corners
	std::srand(static_cast<unsigned int>(std::time(nullptr)));                               // Set initial seed value to system clock
	selectedPixels = randomSelectionFromDistribution(pixels,featuresVector,NSelectedPixels); // Pseudo-random selection of pixels based on edge detection
	selectedPixels.push_back(0);                                                             // Top left corner
	selectedPixels.push_back(sizeIn.width-1);                                                // Top right corner
	selectedPixels.push_back(NpixelsIn-sizeIn.width+1);                                      // Bottom right corner
	selectedPixels.push_back(NpixelsIn-1);                                                   // Bottom left corner

	// 2.3 Add some randomisation / move pixels slightly (todo)

	// 2.4 Visualise selected points
	cv::Point center;
	for (int i=0; i<selectedPixels.size(); i++)
	{
		center = cv::Point{selectedPixels[i]%sizeIn.width,selectedPixels[i]/sizeIn.width};
		circle(imgFeatures,center,5,CV_RGB(255,255,255),3);
	}

	// 3. Create polygons
	std::vector<double> coords(2*selectedPixels.size()); // Holds coordinates of chosen points in the format {x1,y1,x2,y2,x3,y3,...} required by Delaunator below
	for (int i=0; i<selectedPixels.size(); i++)
	{
		coords.push_back(selectedPixels[i]%sizeIn.width); // x coordinate for ith point
		coords.push_back(selectedPixels[i]/sizeIn.width); // y coordinate for ith point
	}

	delaunator::Delaunator mesh(coords); // Performs the triangulation

	std::vector<std::array<cv::Point,3> > triangles; // Vector to store arrays of coordinates for each triangle
	//triangles.reserve(???); // todo: reserve some memory for `triangles` (how to know how many trianlges there are???)
	for (int nTriangle = 0; nTriangle*3 < mesh.triangles.size(); nTriangle++)
	{
		int nPoint = nTriangle*3;
		int x1 = mesh.coords[2*mesh.triangles[nPoint]];
		int y1 = mesh.coords[2*mesh.triangles[nPoint]   + 1];
		int x2 = mesh.coords[2*mesh.triangles[nPoint+1]];
		int y2 = mesh.coords[2*mesh.triangles[nPoint+1] + 1];
		int x3 = mesh.coords[2*mesh.triangles[nPoint+2]];
		int y3 = mesh.coords[2*mesh.triangles[nPoint+2] + 1];
		cv::Point p1 {x1, y1};
		cv::Point p2 {x2, y2};
		cv::Point p3 {x3, y3};
		triangles.push_back(std::array<cv::Point,3> {p1,p2,p3});
	}

	for (int i=0; i<triangles.size(); i++) // Draw each triangle onto the image
		fillPoly(imgFeatures, triangles.at(i), cv::Scalar(255-i, 255-i, 255-i), 8, 0); // meaninless colours for now (and could go below 0)

	// 5. Export image
	//cv::Mat imgOut = imgFeatures.clone();
	cv::imwrite("media/output.jpg",imgFeatures);

	return 0;
}