#include "texture.h"

void Texture::loadTexture(std::string imgName) {
    img.loadImage(imgName);
}

SColor Texture::getTexel(float u, float v) {
    int x = (int) floor(u * (float)img.width());
    int y = (int) floor(v * (float)img.height());
    return img.getSample(x, y, BMP_MAX_VAL);
}
