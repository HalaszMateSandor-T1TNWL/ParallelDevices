__kernel void rgb2gray(
    __global float* R,
    __global float* G,
    __global float* B,
    __global float* GrayScale
)
{
    int i = get_global_id(0);
    GrayScale[i] = 0.2125 * R[i] + 0.7154 * G[i] + 0.0721 * B[i];
}
