#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <CL/cl2.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cl;

/* ========================================
 * ------------OpenCL Functions------------
 * ========================================
 */

cl::Device getDefaultDevice();

void initializeDevice();

/* ========================================
 * ------------OpenCV Functions------------
 * ========================================
 */

Mat readImage(const string filename);
Animation makeGif();

/* ========================================
 * -----------Global Variables-------------
 * ========================================
 */

cl::Program program;
cl::Context context;
cl::Device device;

/* ========================================
 * -------------Main Function--------------
 * ========================================
 */

int main(){
    
    initializeDevice();

    Mat image = readImage("../BabyGirl.png");

    Mat rgb[3];
    cv::split(image, rgb);


    Mat gray(image.rows, image.cols, CV_32F);
    image.convertTo(image, CV_32F);

    size_t total_size = image.rows * image.cols * sizeof(float);
    Buffer redBuffer(context, CL_MEM_READ_ONLY, total_size);
    Buffer greenBuffer(context, CL_MEM_READ_ONLY, total_size);
    Buffer blueBuffer(context, CL_MEM_READ_ONLY, total_size);
    Buffer grayBuffer(context, CL_MEM_READ_ONLY, total_size);

    Kernel kernel(program, "rgb2gray");
    kernel.setArg(0, redBuffer);
    kernel.setArg(1, greenBuffer);
    kernel.setArg(2, blueBuffer);
    kernel.setArg(3, grayBuffer);

    CommandQueue queue(context, QueueProperties::Profiling);
    queue.enqueueWriteBuffer(
            redBuffer,
            CL_TRUE,
            0,
            total_size,
            rgb[2].ptr<float>()
    );
    queue.enqueueWriteBuffer(
            greenBuffer,
            CL_TRUE,
            0,
            total_size,
            rgb[1].ptr<float>()
    );
    queue.enqueueWriteBuffer(
            blueBuffer,
            CL_TRUE,
            0,
            total_size,
            rgb[0].ptr<float>()
    );

    Event event;
    queue.enqueueNDRangeKernel(
        kernel,
        NullRange,
        NDRange((int)(image.rows*image.cols)),
        NullRange,
        nullptr,
        &event
    );

    queue.enqueueReadBuffer(
        grayBuffer,
        CL_TRUE,
        0,
        total_size,
        gray.ptr<float>()
    );


    return 0;
}


/* ========================================
 * ------------OpenCL Functions------------
 * ========================================
 */

cl::Device getDefaultDevice(){

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if(platforms.empty())
    {
        std::cerr << "No platforms detected!" << std::endl;
        exit(1);
    }

    auto platform = platforms.front();
    std::vector<cl::Device> devices;

    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if(devices.empty())
    {
        std::cerr << "No devices found!" << std::endl;
        exit(1);
    }

    std::cout << devices.front().getInfo<CL_DEVICE_NAME>() << " found" << std::endl;

    return devices.front();
}

void initializeDevice(){

    device = getDefaultDevice();

    std::cout << "Initializing device..." << std::endl;

    std::ifstream kernelFile("../rgb2gray_kernel.cl");

    std::string source(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));

    //std::cout << "Loaded kernel file:\n" << source << std::endl;

    cl::Program::Sources sources;
    sources.push_back({ source.c_str(), source.length() + 1 });

    context = cl::Context(device);
    program = cl::Program(context, sources);

    auto err = program.build(device);
    if(err != CL_BUILD_SUCCESS)
    {
       std::cerr << "ERROR\nBuild Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) << "\nBuild Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
       exit(1);
    }
}

/* ========================================
 * ------------OpenCV Functions------------
 * ========================================
 */

Mat readImage(const String filename){

    std::cout << "Reading image..." << std::endl;
    Mat image = imread(filename);
    if(image.empty())
    {
        std::cout << "Image: " << filename << " couldn't be loaded" << std::endl;
        exit(1);
    }

    return image;
}
