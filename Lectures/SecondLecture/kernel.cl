__kernel void kernel_code(__global float* a, __global float* b, __global float* c, const int size)
{
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}