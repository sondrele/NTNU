#include "rayimage.h"

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
