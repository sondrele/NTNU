#include "texture.h"

SColor Texture::getTexel(float u, float v) {
    int x = (int) floor(u * (float)img.width());
    int y = (int) floor(v * (float)img.height());
    return getSample(x, y, BMP_MAX_VAL);
}
