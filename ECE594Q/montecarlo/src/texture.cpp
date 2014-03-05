#include "texture.h"

SColor Texture::getTexel(float u, float v) {
    int x = (int) floor(u * (float) width());
    int y = (int) floor(v * (float) height());
    return getSample(x, y, BMP_MAX_VAL);
}
