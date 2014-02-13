#ifndef _TEXTURE_H_
#define _TEXTURE_H_

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "rayimage.h"
#include "rayscene_shapes.h"

class Texture {
private:
    RayImage img;

public:
    Texture();

    void loadTexture(std::string);
    SColor getTexel(float, float);

};

#endif // _TEXTURE_H_