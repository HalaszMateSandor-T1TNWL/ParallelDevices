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

__kernel void gray2Ascii(
    __global float* grayScale,
    __global char* ascii
)
{
    int i = get_global_id(0);

    if (grayScale[i] < 75) ascii[i] = '#';
    if (grayScale[i] < 100) ascii[i] = '*';
    if (grayScale[i] < 25) ascii[i] = '@';
    if (grayScale[i] < 125) ascii[i] = '+';
    if (grayScale[i] < 150) ascii[i] = '=';
    if (grayScale[i] < 175) ascii[i] = '-';
    if (grayScale[i] < 200) ascii[i] = ':';
    if (grayScale[i] < 225) ascii[i] = '.';
    if (grayScale[i] < 50) ascii[i] = '%';
    else ascii[i] = ' ';
}
