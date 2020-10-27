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

using triangleArray = std::vector<std::array<cv::Point,3> >;

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

// Returns the indices of `n` random elements of the `freqs` vector.
// Larger values in `freqs` are more likely to be chosen.
std::vector<int> randomSelectionFromDistribution(std::vector<int>& freqs, int n)
{
	std::vector<int> freqCumSum = calculateCumulativeSum(freqs);
	std::vector<int> outputVals;
	outputVals.reserve(n);
	for (int i=0; i<n; i++)
	{
		int rand = getRandomNumber(1,freqCumSum.back()); // calculate a random number between 1 and sum(freqs)
		int ceilingIndex = findCeil(freqCumSum,rand);    // determine the index of `values` to which the random number rounds up to
		outputVals.push_back(ceilingIndex);
	}
	return outputVals;
}

// Select Npixels pixels from imgFeatures. Returns a vector of selected pixel "numbers" (counting from top left to bottom right).
// imgFeatures in a greyscale image in which the intensity indicates how "important" that pixel is.
// In practice, imgFeatures is the output of edge detection, but this is not necessarily required
std::vector<int> selectPixels(cv::Mat& imgFeatures, int Npixels)
{
	std::vector<int> featuresVector;
	if(!imgFeatures.isContinuous())
		imgFeatures = imgFeatures.reshape(1,imgFeatures.total());
	featuresVector.assign(imgFeatures.data,imgFeatures.data+imgFeatures.total());    // Assumes only one channel (imgFeatures is greyscale); uchars are cast to ints

	std::vector<int> selectedPixels(Npixels+4);                                      // Vector large enough to store NSelectedPixels randomly selected pixels, plus the four corners
	std::srand(static_cast<unsigned int>(std::time(nullptr)));                       // Set initial seed value to system clock
	selectedPixels = randomSelectionFromDistribution(featuresVector,Npixels); // Pseudo-random selection of pixels based on edge detection
	selectedPixels.push_back(0);                                                     // Top left corner
	selectedPixels.push_back(imgFeatures.size().width-1);                            // Top right corner
	selectedPixels.push_back(imgFeatures.total()-imgFeatures.size().width+1);        // Bottom right corner
	selectedPixels.push_back(imgFeatures.total()-1);                                 // Bottom left corner

	return selectedPixels;
}

// Take a Delaunator and return a vector containing an array of vertices for each triangle in the Delaunay mesh
triangleArray makeTrianglesFromMesh(const delaunator::Delaunator& mesh)
{
	triangleArray triangles;                    // Vector to store arrays of coordinates for each triangle
	triangles.reserve(mesh.triangles.size()/3); // mesh.triangles lists vertices for each triangle
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

	return triangles;
}

// Draw the triangles stored in the `triangles` vector on imgOut.
// The colour of each triangle is determined by the mean colour of the corresponding pixels in imgIn.
// `triangles` is assumed to correspond at least to an image of the same size as imgIn and imgOut.
void drawTriangles(const cv::Mat& imgIn, cv::Mat& imgOut, const triangleArray& triangles)
{
	cv::Mat mask;
	for (auto triangle : triangles)                                  // Draw each triangle onto the image (with colour)
	{
		cv::Mat mask = cv::Mat::zeros(imgIn.size(), CV_8U);          // Reset mask for each triangle
		fillPoly(mask,   triangle, cv::Scalar(255,255,255), 8, 0);   // Make a triangle shaped mask (for finding the average colour below)
		fillPoly(imgOut, triangle, cv::mean(imgIn,mask),    8, 0);   // Draw coloured triangles on imgOut
	}
}

// main() takes two input arguments: the name of the input image and the number of pixels to randomly select
int main(int argc, char** argv)
{
	// PARSING INPUT ARGUMENTS
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
		NSelectedPixels = 100;          // If conversion fails, set NSelectedPixels to a default value
		std::cout << "Could not interpret second argument as an integer. Taking dafault value of N=" << NSelectedPixels << '\n';
	}

	// Deal with case of too many inputs
	if(argc>3)
		std::cout << "Ignoring additional inputs";

	// LOW POLYFICATION
	cv::Mat imgProcessed;
	cv::GaussianBlur(imgIn, imgProcessed, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT); // Reduce noise with Gaussian blur (kernel size = 3)
	cv::cvtColor(imgProcessed, imgProcessed, cv::COLOR_BGR2GRAY);                    // Convert to greyscale

	if (!imgIn.depth()==0)
		std::cout << "Input image has depth!=0; this may need to be addressed";
	int depth = CV_16S;                                          // Depth of the output image (input assumed to be CV_8U; don't want overflow; laplacian can be negative)
	cv::Mat imgFeatures;
	cv::Laplacian(imgProcessed,imgFeatures,depth);               // Edge detection
	cv::convertScaleAbs(imgFeatures,imgFeatures);                // Take absolute value of pixel values and convert back to 8 bits

	std::vector<int> selectedPixels = selectPixels(imgFeatures, NSelectedPixels); // Select pixels (for triangle vertices) pseudo-randomly

	std::vector<double> coords(2*selectedPixels.size());         // Holds coordinates of chosen points in the format {x1,y1,x2,y2,x3,y3,...} required by Delaunator below
	for (int pixel : selectedPixels)
	{
		coords.push_back(pixel%imgIn.size().width);  // x coordinate for ith point
		coords.push_back(pixel/imgIn.size().width);  // y coordinate for ith point
	}

	delaunator::Delaunator mesh(coords);                         // Perform the triangulation
	triangleArray triangles = makeTrianglesFromMesh(mesh);       // Extract triangle vertices from Delaunator
	cv::Mat imgOut = cv::Mat::zeros(imgIn.size(), imgIn.type());
	drawTriangles(imgIn,imgOut,triangles);                       // Draw the (coloured) triangles on imgOut

	cv::imwrite("media/output.jpg",imgOut);                      // Export image

	return 0;
}