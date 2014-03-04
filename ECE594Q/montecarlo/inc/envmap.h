#ifndef _ENVMAP_H_
#define _ENVMAP_H_

#include <string>

#include "material.h"
#include "ray.h"
#include "raybuffer.h"
#include "rimage.h"

class EnvMap : public RImage {
public:
    EnvMap() {}
    ~EnvMap() {}

    SColor getTexel(Vect);
};

#endif // _ENVMAP_H_