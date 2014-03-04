#include "rayimage.h"

#include <iostream>

int RayImage::width() {
    return img.width();
}

int RayImage::height() {
    return img.height();
}

void RayImage::loadImage(std::string n) {
    name = n;
    img.load(name.c_str());
}

void RayImage::displayImage() {
    cimg_library::CImgDisplay main_disp(img, name.c_str());
    while (!main_disp.is_closed())
        ;
}

SColor RayImage::getSample(int x, int y, float maxval) {
    // y = height() - y;
    // TODO: y must be changed for textures

    SColor c;
    c.R(img(x, y, 0, 0) / maxval);
    c.G(img(x, y, 0, 1) / maxval);
    c.B(img(x, y, 0, 2) / maxval);
    return c;
}

// SColor RayImage::getSColor(int x, int y) {
//     float r = img(x, y, 0, 0) / 65535.0f;
//     float g = img(x, y, 0, 1) / 65535.0f;
//     float b = img(x, y, 0, 2) / ;
//     SColor c;
//     c.R(r);
//     c.G(g);
//     c.B(b);
//     return c;
// }

void RayImage::createImage(RayBuffer buffer, std::string name) {
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

    image.save_bmp(name.c_str());
}
