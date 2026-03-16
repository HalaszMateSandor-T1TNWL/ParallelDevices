#include "include/read_kernel.h"
#include <CL/cl.h>

int SAMPLE_SIZE = 1024;

int main(int argc, char **argv)
{
    const char* kernel_source = read_kernel("kernels/kernel_code.cl");

    int* number_array = (int*)malloc(sizeof(int)*SAMPLE_SIZE);
    for(int i = 0; i < SAMPLE_SIZE; i++)
    {
        number_array[i] = rand() % 100;
    }
    int* relativity = (int*)malloc(sizeof(int)*SAMPLE_SIZE);

    cl_int err;

    cl_uint n_platforms;
    cl_platform_id platform_id;

    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
    if(err != CL_SUCCESS)
    {
        printf("[ERROR] Failed loading platform id\n");
        return 0;
    }

    cl_uint n_devices;
    cl_device_id device_id;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &n_devices);
    if(err != CL_SUCCESS)
    {
        printf("[ERROR] Failed loading device id");
        return 0;
    }

    cl_context context = clCreateContext(NULL, n_devices, &device_id, NULL, NULL, &err);
    if(err != CL_SUCCESS)
    {
        printf("[ERROR] Unable to create context");
        return 0;
    }

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    if(err != CL_SUCCESS)
    {
        printf("[ERROR] Couldn't create program");
        return 0;
    }

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err != CL_SUCCESS)
    {
        printf("[ERROR] Couldn't build the program: %d\n", err);
        return 0;
    }

    cl_kernel kernel = clCreateKernel(program, "kernel_relative", &err);
    if(err != CL_SUCCESS)
    {
        printf("[ERROR] Couldn't create kernel");
        return 0;
    }

    cl_mem d_Array = clCreateBuffer(context, CL_MEM_READ_ONLY, SAMPLE_SIZE*sizeof(int), NULL, &err);
    if(err != CL_SUCCESS)
    {
        printf("[ERROR] Couldn't create buffer, line: 69\n");
        return 0;
    }
    cl_mem d_Relativity = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SAMPLE_SIZE*sizeof(int), NULL, &err);
    if(err != CL_SUCCESS)
    {
        printf("[ERROR] Couldn't create buffer, line: 75: %d\n", err);
        return 0;
    }

    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &err);
    if(err != CL_SUCCESS)
    {
        printf("[ERROR] Couldn't create command queue");
        return 0;
    }

    err = clEnqueueWriteBuffer(
        command_queue,
        d_Array,
        CL_FALSE,
        0,
        SAMPLE_SIZE * sizeof(int),
        number_array,
        0,
        NULL,
        NULL
    );
    if(err != CL_SUCCESS)
    {
        printf("[ERROR] Couldn't create write buffer");
        return 0;
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_Array);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_Relativity);
    clSetKernelArg(kernel, 2, sizeof(int), (void*)&SAMPLE_SIZE);

    size_t local_work = 5;
    size_t n_work_groups = (SAMPLE_SIZE + local_work) / local_work;
    size_t global_work = local_work + n_work_groups;

    printf("Local Work: %d, Number of work groups: %d, Global Work %d\n", local_work, n_work_groups, global_work);

    err = clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        1,
        0,
        &global_work,
        &local_work,
        0,
        NULL,
        NULL
    );
    if(err != CL_SUCCESS)
    {
        printf("[ERROR] EnqueueNDRangeKernel: %d\n", err);
        return 0;
    }

    err = clEnqueueReadBuffer(
        command_queue,
        d_Relativity,
        CL_TRUE,
        0,
        SAMPLE_SIZE*sizeof(int),
        relativity,
        0,
        NULL,
        NULL
    );
    if(err != CL_SUCCESS)
    {
        printf("[ERROR] Error occoured while reading buffer: %d", err);
        return 0;
    }

    clFinish(command_queue);

    for(int i = 0; i < 101; i++)
    {
        printf("[%d]: %d\n", i, relativity[i]);
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device_id);

    free(number_array);
    free(relativity);

    return 0;
}