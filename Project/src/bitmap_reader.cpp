#include <iostream>
#include <CL/opencl.hpp>


typedef unsigned int color_t;

typedef struct Bitmap {
    char identifier[2];
    unsigned int size;
    unsigned int offset;
    int height;
    int width;
    color_t* pixels;
    unsigned char red;
    unsigned char blue;
    unsigned char green;
} Bitmap;

void read_header(struct Bitmap* bitmap, FILE* file, const char* filename)
{
    file = fopen(filename, "rb");
    if(file == NULL)
    {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fread(&bitmap->identifier, 2, 1, file);
    if(bitmap->identifier[0] != 'B' || bitmap->identifier[1] != 'M')
    {
        fprintf(stderr, "(%s) Not a BMP file.\n", filename);
        exit(EXIT_FAILURE);
    }
    else {
        printf("(%s) Success, BM read, Bitmap file identified\n...Reading in data\n", filename);
    }

    fread(&bitmap->size, sizeof(unsigned int), 1, file);
    printf("(%s) Size of image: %d\n", filename, bitmap->size);
}