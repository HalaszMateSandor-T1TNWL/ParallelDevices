#include <CL/cl.h>
#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <CL/cl2.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <string>

using namespace std;
using namespace cv;
using namespace cl;

int divisor(int number, int max);

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

    size_t total_size = image.rows * image.cols * sizeof(float);
    Buffer redBuffer(context, CL_MEM_READ_ONLY, total_size);
    Buffer greenBuffer(context, CL_MEM_READ_ONLY, total_size);
    Buffer blueBuffer(context, CL_MEM_READ_ONLY, total_size);
    Buffer grayBuffer(context, CL_MEM_WRITE_ONLY, total_size);

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

    cout << "Image dimensions: " << image.rows << " x " << image.cols << endl;

    int local_work = ((int)divisor(image.rows*image.cols, kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device)));
    cout << "Max work item sizes:\t" << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0] << endl;
    cout << "Max work group size:\t" << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;
    cout << "Max individual work group size:\t" << kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << endl;
    cout << "Local work size chosen:\t" << local_work << endl;

    Event event;
    auto err = queue.enqueueNDRangeKernel(
        kernel,
        NullRange,
        NDRange((int)(image.rows*image.cols)),
        NDRange(local_work),
        nullptr,
        &event
    );
    if(err != CL_SUCCESS)
    {
        cout << "ERROR with enqueueNDRangeKernel\t" << err << endl;
        exit(1);
    }

    queue.enqueueReadBuffer(
        grayBuffer,
        CL_TRUE,
        0,
        total_size,
        gray.ptr<float>()
    );

    queue.finish();
    queue.flush();

    double time_taken = ((double)event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - (double)(event.getProfilingInfo<CL_PROFILING_COMMAND_START>())) / 1000000.0;

    image.convertTo(image, CV_8U);
    imshow("Image RGB", image);

    gray.convertTo(gray, CV_8U);
    imshow("Image Gray", gray);

    cout << "Time taken for operation:\t" << time_taken << endl;


    waitKey(0);

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

    image.convertTo(image, CV_32F);

    if(image.empty())
    {
        std::cout << "Image: " << filename << " couldn't be loaded" << std::endl;
        exit(1);
    }

    return image;
}

int divisor(int number, int max)
{
    int i;
    for(i = number / 2; i >= 1; i--)
    {
        if(number % i == 0 && i <= max)
        {
            break;
        }
    }
    return i;
}
