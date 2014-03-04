#ifndef _RIMAGE_H_
#define _RIMAGE_H_

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "CImg.h"
#include "raybuffer.h"
#include "material.h"

#define BMP_MAX_VAL     255.0f
#define EXR_MAX_VAL     65535.0f

class RImage {
public:
    cimg_library::CImg<float> img;
    std::string name;

public:
    int width();
    int height();
    void loadImage(std::string);
    void displayImage();
    void createImage(RayBuffer, std::string);
    SColor getSample(int, int, float);
};

#endif // _RIMAGE_H_