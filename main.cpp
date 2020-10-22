#include <opencv2/core/mat.hpp>  // for basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgcodecs.hpp> // for reading and writing

// main() takes one input argument: the name of the input image
int main( int argc, char** argv )
{
    // 1. Load image
    const std::string inputImageName = argv[1];
    cv::Mat imgIn = cv::imread(inputImageName,cv::IMREAD_COLOR);

    // 2. Extract keypoints
    //  - Edge detection
    //  - Some randomisation for aesthetic appeal

    // 3. Create polygons
    //  - Delaunay triangulation

    // 4. Colour polygons

    // 5. Export image
    cv::Mat imgOut = imgIn.clone();
    cv::imwrite("media/output.jpg",imgOut);

	return 0;
}