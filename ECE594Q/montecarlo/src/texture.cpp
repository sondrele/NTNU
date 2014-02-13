#include "texture.h"

Texture::Texture() {

}

void Texture::loadTexture(std::string imgName) {
    img.loadTexture(imgName);   
}

SColor Texture::getTexel(float u, float v) {
    int x = (int) floor(u * (float)img.width());
    int y = (int) floor(v * (float)img.height());
    PX_Color c = img.getSample(x, y);
    SColor sc;
    sc.R(c.R / 255.0f);
    sc.G(c.G / 255.0f);
    sc.B(c.B / 255.0f);
    return sc;
}
