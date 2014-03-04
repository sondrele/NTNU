#include "envmap.h"

void EnvMap::loadMap(std::string name) {
    img.loadImage(name);
}

void EnvMap::show() {
    img.displayImage();
}

SColor EnvMap::getTexel(Vect dir) {
    float u = (float) (0.5 + atan2(dir.getZ(), dir.getX()) / (2 * M_PI));
    float v = (float) (0.5 - asin(dir.getY()) / M_PI);

    int x = (int) floor(u * (float) img.width());
    int y = (int) floor(v * (float) img.height());
    return img.getSample(x, y, EXR_MAX_VAL);
}
