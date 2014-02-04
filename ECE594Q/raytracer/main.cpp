#include "rayimage.h"
#include <stdlib.h>
#include <iostream>

int main(int argc, char const *argv[])
{
    RayBuffer rb(2, 2);
    rb.setPixel(0, 0, 255, 255, 255);
    rb.setPixel(1, 0, 170, 170, 170);
    rb.setPixel(0, 1, 85, 85, 85);
    rb.setPixel(1, 1, 0, 0, 0);

    RayImage ri;
    ri.createImage(rb, "test.bmp");
    return 0;
}