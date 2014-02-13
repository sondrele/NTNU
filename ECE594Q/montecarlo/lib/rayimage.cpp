#include "rayimage.h"

int RayImage::width() {
    return img.width();
}

int RayImage::height() {
    return img.height();
}

void RayImage::loadTexture(std::string name) {
    img = cimg_library::CImg<unsigned char>(name.c_str());
}

PX_Color RayImage::getSample(int x, int y) {
    y = height() - y;
    PX_Color c;
    c.R = img(x, y, 0, 0);
    c.G = img(x, y, 0, 1);
    c.B = img(x, y, 0, 2);
    return c;
}

void RayImage::createImage(RayBuffer buffer, const char *name) {
    cimg_library::CImg<unsigned char> image(buffer.getWidth(), buffer.getHeight(), 1, 3, 0);

    for (uint y = 0; y < buffer.getHeight(); y++) {
        for (uint x  = 0; x < buffer.getWidth(); x++) {
            RayPixel px = buffer.getPixel(x, y);
            PX_Color c0 = px.getColor();
            unsigned char color[3] = {
                (unsigned char) c0.R,
                (unsigned char) c0.G,
                (unsigned char) c0.B
            };
            image.draw_point(px.getX(), buffer.getHeight() - 1 - px.getY(), color);
        }
    }

    image.save_bmp(name);
}
