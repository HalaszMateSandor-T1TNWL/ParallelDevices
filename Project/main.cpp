#include <string>
#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
//#include <opencv2/core/cvstd.hpp>
#include <CL/opencl.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/traits.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <string>

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

namespace fs = std::filesystem;

using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace cl;

/* ========================================
 * ----------Secondary Functions-----------
 * ========================================
 */

int divisor(int number, int max);
void rgb2gray_sequential(const string filename);
std::vector<string> get_images_from_directory(const string dirname);
char character_to_grayscale(float grayscale);

/* ========================================
 * ------------OpenCL Functions------------
 * ========================================
 */

cl::Device getDefaultDevice();

void initializeDevice();

Mat rgb2gray_parallel(const string filename);
Mat gray2Ascii(Mat grayImage);

/* ========================================
 * ------------OpenCV Functions------------
 * ========================================
 */

Mat readImage(const string filename);

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

int main(int argc, const char** argv){

    /*if(argc != 2)
    {
        cout << "Please provide the path to an image." << endl;
        return 1;
    }*/

    initializeDevice();

    std::vector<string> files = get_images_from_directory(".");

    cv::Animation animation;
    std::vector<cv::Mat> images;

    for(int i = 0; i < files.size(); i++)
    {
        images.push_back(rgb2gray_parallel(files[i]));
    }

    Mat image = rgb2gray_parallel("BuryArizona.png");

    //Mat asciiImage = gray2Ascii(images[0]);
    Mat ascii(image.rows, image.cols, CV_8UC4);

    for(int y = 0; y < image.cols; y++)
    {
        for(int x = 0; x < image.rows; x++)
        {
            ascii.at<char>(x,y) = character_to_grayscale(image.at<char>(x,y));
        }
    }

    Mat asciiImage(image.rows, image.cols, CV_8UC3, Scalar(0));

    for(int y = 0; y < asciiImage.cols; y++)
    {
        for(int x = 0; x < asciiImage.rows; x++)
        {
            string s(1, ascii.at<char>(x,y));
            putText(asciiImage,
                    s,
                    cv::Point(x+10,y+10),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    CV_RGB(118, 185, 55),
                    1
            );
        }
    }

    //putText(asciiImage, string(1, ascii.at<char>(0,0)), cv::Point(0,10), cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(118, 185, 55), 1);


    imshow("Ascii", asciiImage);
    imshow("Image", image);

    waitKey(0);

    return 0;
}

char character_to_grayscale(float grayscale) {
    if (grayscale < 25) return '@';
    if (grayscale < 50) return '%';
    if (grayscale < 75) return '#';
    if (grayscale < 100) return '*';
    if (grayscale < 125) return '+';
    if (grayscale < 150) return '=';
    if (grayscale < 175) return '-';
    if (grayscale < 200) return ':';
    if (grayscale < 225) return '.';
    return ' ';
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

    cl::Program::Sources sources;
    sources.push_back({ source.c_str(), source.length() + 1 });

    cout << sources[0] << endl;

    context = cl::Context(device);
    program = cl::Program(context, sources);

    auto err = program.build(device);
    if(err != CL_BUILD_SUCCESS)
    {
       std::cerr << "[ERROR] Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) << "\nBuild Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
       exit(1);
    }
}

Mat rgb2gray_parallel(const string filename){

    Mat image = readImage(filename);

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

    int local_work = (divisor(image.rows*image.cols, kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device)));
    cout << "Max work item sizes:\t" << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0] << endl;
    cout << "Max work group size:\t" << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;
    cout << "Max individual work group size:\t" << kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << endl;
    cout << "Local work size chosen:\t" << local_work << endl;

    Event event;
    auto err = queue.enqueueNDRangeKernel(
        kernel,
        NullRange,
        NDRange((int)(image.rows*image.cols)),
        //NullRange,
        NDRange(local_work),
        nullptr,
        &event
    );
    if(err != CL_SUCCESS)
    {
        cout << "[ERROR] enqueueNDRangeKernel\t" << err << endl;
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

    cout << "Time taken for operation:\t" << time_taken << " milliseconds" << endl;

    gray.convertTo(gray, CV_8UC4);

    return gray;
}

Mat gray2Ascii(Mat grayImage)
{
    cout << "Starting ascii conversion..." << endl;
    Mat asciis(grayImage.rows, grayImage.cols, CV_8U);

    size_t floatBuffer_size = grayImage.rows * grayImage.cols * sizeof(float);
    size_t charBuffer_size = grayImage.rows * grayImage.cols * sizeof(char);

    Buffer grayBuffer(context, CL_MEM_READ_ONLY, floatBuffer_size);
    Buffer asciiBuffer(context, CL_MEM_WRITE_ONLY, charBuffer_size);

    Kernel kernel(program, "gray2Ascii");
    kernel.setArg(0, grayBuffer);
    kernel.setArg(1, asciiBuffer);

    CommandQueue queue(context, QueueProperties::Profiling);
    queue.enqueueWriteBuffer(
        grayBuffer,
        CL_TRUE,
        0,
        floatBuffer_size,
        grayImage.ptr<float>()
    );

    cout << "Image dimensions: " << grayImage.rows << " x " << grayImage.cols << endl;

    int local_work = (divisor(grayImage.rows*grayImage.cols, kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device)));
    cout << "Max work item sizes:\t" << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0] << endl;
    cout << "Max work group size:\t" << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;
    cout << "Max individual work group size:\t" << kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << endl;
    cout << "Local work size chosen:\t" << local_work << endl;

    Event event;
    auto err = queue.enqueueNDRangeKernel(
        kernel,
        NullRange,
        NDRange((int)(grayImage.rows*grayImage.cols)),
        //NullRange,
        NDRange(local_work),
        nullptr,
        &event
    );
    if(err != CL_SUCCESS)
    {
        cout << "[ERROR] enqueueNDRangeKernel\t" << err << endl;
        exit(1);
    }

    queue.enqueueReadBuffer(
        asciiBuffer,
        CL_TRUE,
        0,
        charBuffer_size,
        asciis.ptr<char>()
    );

    queue.finish();
    queue.flush();

    double time_taken = ((double)event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - (double)(event.getProfilingInfo<CL_PROFILING_COMMAND_START>())) / 1000000.0;

    cout << "Time taken for operation:\t" << time_taken << " milliseconds" << endl;

    return asciis;
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

/* ========================================
 * ----------Secondary Functions-----------
 * ========================================
 */

void rgb2gray_sequential(const string filename) {

    auto start = high_resolution_clock::now();

    Mat image = readImage(filename);

    Mat rgb[3]; //colour channels go: B G R

    cv::split(image, rgb);

    Mat gray(image.rows, image.cols, CV_32F);

    for(int i = 0; i < image.cols; i++)
    {
        for(int j = 0; j <  image.rows; j ++)
        {
            gray.at<float>(j,i) = rgb[2].at<float>(j,i) * 0.2125 + rgb[1].at<float>(j,i) * 0.7154 + rgb[0].at<float>(j,i) * 0.0721;
        }
    }

    gray.convertTo(gray, CV_8U);

    auto end = high_resolution_clock::now();

    auto duration = duration_cast<nanoseconds>(end - start);

    cout << "Time taken:\t" << duration.count() / 1000000.0 << " milliseconds" << endl;

    imshow("Sequential", gray);
}


int divisor(int number, int max){

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

std::vector<string> get_images_from_directory(const string dirname)
{
    std::vector<string> files;

    fs::path searchDir;
    if(dirname.empty())
    {
        cout << "No directory given.\nPlease provide a directory with the images you'd like to read in" << endl;
        exit(1);
    } else {
        searchDir = dirname;
    }

    for(const auto &entry : fs::directory_iterator(searchDir))
    {
        if(entry.is_regular_file())
        {
            if(entry.path().extension() == ".png")
            {
                cout << "Reading in:\t" << entry.path().filename() << endl;
                files.push_back(entry.path().string());
            }
        }
    }

    return files;
}
