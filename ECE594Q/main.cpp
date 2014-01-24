#include "Mesh.h"
#include "FrameBuffer.h"
#include <cstdlib>
#include <iostream>

int main(int argc, char const *argv[]) {
    cimg_library::CImg<unsigned char> image(500, 500, 1, 3, 0);
    const unsigned char c[3] = {255, 255, 255};
    image.draw_point(0, 0, c);
    image.save_jpeg("asdas");
    return 0;
}
