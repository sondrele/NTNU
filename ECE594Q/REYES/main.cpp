#include "Mesh.h"
#include "FrameBuffer.h"
#include <cstdlib>
#include <iostream>

int main(int argc, char const *argv[]) {
    FrameBuffer fb(500, 500, 2, 2);
    RiSphere s(20, 32);
    s.rotate('z', 180);
    s.rotate('x', 90);
    s.translate(0, 0, 100);
    fb.addMesh(s);
    
    RiSphere s2(10, 32);
    s2.translate(10, -10, 50);
    fb.addMesh(s2);

    RiSphere s3(10, 32);
    s3.rotate('x', -90);
    s3.translate(-10, -10, 50);
    fb.addMesh(s3);

    fb.drawShapes("./imgs/fb_test_sphere.jpg");
    return 0;
}
