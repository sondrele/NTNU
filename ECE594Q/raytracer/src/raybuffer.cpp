#include "raybuffer.h"

RayPixel::RayPixel(uint x, uint y) {
    X = x;
    Y = y;
}

void RayPixel::setColor(PX_Color c) {
    color = c;
}

void RayPixel::setColor(unsigned char R, unsigned char G, unsigned char B) {
    color.R = R;
    color.G = G;
    color.B = B;
}

RayBuffer::RayBuffer(uint width, uint height) {
    WIDTH = width;
    HEIGHT = height;

    for (uint y = 0; y < HEIGHT; y++) {
        for (uint x = 0; x < WIDTH; x++) {
            pixels.push_back(RayPixel(x, y));
        }
    }
}

void RayBuffer::setPixel(uint x, uint y, PX_Color color) {
    pixels[PX(x, y)].setColor(color);
}

void RayBuffer::setPixel(uint x, uint y, unsigned char R, unsigned char G, unsigned char B) {
    PX_Color c = {R, G, B};
    pixels[PX(x, y)].setColor(c);
}

RayPixel RayBuffer::getPixel(uint x, uint y) {
    return pixels[PX(x, y)];
}
