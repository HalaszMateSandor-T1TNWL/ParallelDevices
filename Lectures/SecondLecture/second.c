#include <stdio.h>
#include <CL/cl.h>

int SAMPLE_SIZE = 1024;

char* read_kernel_code(const char* filename)
{
    FILE* ptr;
    char* buffer;
    long size;

    ptr = fopen(filename, "r");
    fseek(ptr, 0, SEEK_END);
    size = ftell(ptr);

    rewind(ptr);

    buffer = (char*)malloc(size * sizeof(char));
    fread(buffer, sizeof(char), size, ptr);

    buffer[size] = 0;

    return buffer;

}

int main(int argc, char **argv)
{
    float* A = (float*)malloc(SAMPLE_SIZE*sizeof(float));
    float* B = (float*)malloc(SAMPLE_SIZE*sizeof(float));
    
    for (int i = 0; i < SAMPLE_SIZE; i++)
    {
        A[i] = i + 1;
        B[i] = (i+1)*2;
    }

    float* result = (float*)malloc(SAMPLE_SIZE*sizeof(float));

    const char* kernel_code = read_kernel_code("kernel.cl");

    cl_int err;

    cl_uint n_platforms;
    cl_platform_id platform_id;
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
    if(err != CL_SUCCESS)
    {
        printf("[ERROR] There was an error getting the platform IDs\n");
        return 0;
    }

    cl_device_id device_id;
    cl_uint n_devices;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &n_devices);
    if(err != CL_SUCCESS)
    {
        printf("[ERROR] There was an error getting device IDs\n");
        return 0;
    }

    cl_context context = clCreateContext(NULL, n_devices, &device_id, NULL, NULL, NULL);
    
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, NULL);
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if(err != CL_SUCCESS)
    {
        printf("[ERROR] There was an error building the program\n");
        return 0;
    }

    cl_kernel kernel = clCreateKernel(program, "kernel_code", NULL);

    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, NULL);

    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, SAMPLE_SIZE*sizeof(int), NULL, NULL);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, SAMPLE_SIZE*sizeof(int), NULL, NULL);
    cl_mem d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SAMPLE_SIZE*sizeof(int), NULL, NULL);

    clEnqueueWriteBuffer(
        command_queue,
        d_A,
        CL_TRUE,
        0,
        SAMPLE_SIZE * sizeof(int),
        A,
        0,
        NULL,
        NULL
    );

    clEnqueueWriteBuffer(
        command_queue,
        d_B,
        CL_TRUE,
        0,
        SAMPLE_SIZE * sizeof(float),
        B,
        0,
        NULL,
        NULL
    );

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_result);

    size_t local_work_size = 256;
    size_t n_work_groups = (SAMPLE_SIZE + local_work_size + 1) / local_work_size;
    size_t global_work_size = local_work_size * n_work_groups;


    clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL
    );

    clEnqueueReadBuffer(
        command_queue,
        d_result,
        CL_TRUE,
        0,
        SAMPLE_SIZE * sizeof(float),
        result,
        0,
        NULL,
        NULL
    );

    for (int i = 0; i < SAMPLE_SIZE; i++)
    {
        printf("[%d]: %f + %f = %f\n", i, A[i], B[i], result[i]);
    }
    
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device_id);

    free(A);
    free(B);
    free(result);

    return 0;
}
