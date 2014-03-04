#ifndef _ENVMAP_H_
#define _ENVMAP_H_

#include <string>

#include "material.h"
#include "ray.h"
#include "raybuffer.h"
#include "rayimage.h"

class EnvMap {
private:
    RayImage img;

public:
    EnvMap() {}
    ~EnvMap() {}

    void loadMap(std::string);
    void show();
    SColor getTexel(Vect);
};

#endif