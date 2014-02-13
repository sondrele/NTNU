#ifndef _RAYIMAGE_H_
#define _RAYIMAGE_H_

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "CImg.h"
#include "raybuffer.h"

class RayImage {
public:
    CImg<unsigned char> img;    

public:
    int width();
    int height();
    void loadTexture(std::string);
    void createImage(RayBuffer, const char *name);
    PX_Color getSample(int, int);
};

#endif // _RAYIMAGE_H_