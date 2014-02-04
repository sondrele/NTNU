#ifndef _RAYIMAGE_H_
#define _RAYIMAGE_H_

#include "CImg.h"
#include "raybuffer.h"

class RayImage {

public:
    void createImage(RayBuffer, const char *name);
};

#endif // _RAYIMAGE_H_