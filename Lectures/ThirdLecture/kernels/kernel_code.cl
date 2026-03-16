__kernel void kernel_relative(__global int* int_array, __global int* relativity, int size)
{
    int i = get_global_id(0);
    if(i < 101)
    {
        for(int j = 0; j < size; j++)
        {
            if(int_array[j] == i)
            {
                relativity[i]++;
            }
        }
    }
}
