#ifndef _TEXTURE_H_
#define _TEXTURE_H_

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "material.h"
#include "raybuffer.h"
#include "rimage.h"

class Texture : public RImage {
public:
    Texture() {}
    ~Texture() {}

    SColor getTexel(float, float);
};

#endif // _TEXTURE_H_