#ifndef _TEXTURE_H_
#define _TEXTURE_H_

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "material.h"
#include "raybuffer.h"
#include "rayimage.h"

class Texture {
private:
    RayImage img;

public:
    Texture() {}
    ~Texture() {}

    void loadTexture(std::string);
    SColor getTexel(float, float);

};

#endif // _TEXTURE_H_