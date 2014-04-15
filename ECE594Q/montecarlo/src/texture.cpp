#include "texture.h"

SColor Texture::getTexel(float u, float v) {
    int x = (int) floor(u * (float) width());
    int y = (int) floor(v * (float) height());
    // Flip y coordinate in order to get right texture sample
    y = height() - y;
    return getSample(x, y, BMP_MAX_VAL);
}
