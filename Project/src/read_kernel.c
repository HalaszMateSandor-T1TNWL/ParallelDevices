#include "include/read_kernel.h"

char* read_kernel(char* filename)
{
    FILE* ptr;
    char* buffer;
    long size;

    ptr = fopen(filename, "r");
    if(ptr == NULL)
    {
        printf("[ERROR] Cannot read in file data.\n");
        return NULL;
    }

    fseek(ptr, 0, SEEK_END);
    size = ftell(ptr);

    buffer = malloc(size * sizeof(char));
    rewind(ptr);

    fread(buffer, sizeof(char), size, ptr);

    printf("%s\n", buffer);

    buffer[size] = 0;

    return buffer;
}