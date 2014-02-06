#include "raytracer.h"

RayTracer::RayTracer(uint width, uint height, Vect viewDir, Vect orthoUp) {
    WIDTH = width;
    HEIGHT = height;
    buffer = RayBuffer(WIDTH, HEIGHT);

    camera.setPos(Vect(0, 0, 0));
    camera.setVerticalFOV((float)M_PI / 2.0f);
    camera.setViewDir(viewDir);
    camera.setOrthoUp(orthoUp);
    // scene.setCamera(camera);

    scaleConst = 2;

    calculateImagePlane();
}

RayTracer::RayTracer(uint width, uint height) {
    WIDTH = width;
    HEIGHT = height;
    scaleConst = 2;
    buffer = RayBuffer(WIDTH, HEIGHT);

    camera.setPos(Vect(0, 0, 0));
    camera.setVerticalFOV((float)M_PI / 2.0f);
    camera.setViewDir(Vect(0, 0, -1));
    camera.setOrthoUp(Vect(0, 1, 0));
    // scene.setCamera(camera);

    calculateImagePlane();
}

void RayTracer::setCameraPos(Vect camPos) {
    camPos.normalize();
    camera.setPos(camPos);
}

void RayTracer::setViewDirection(Vect viewDir) {
    viewDir.normalize();
    camera.setViewDir(viewDir);

    calculateImagePlane();
}

void RayTracer::setOrthogonalUp(Vect orthoUp) {
    orthoUp.normalize();
    camera.setOrthoUp(orthoUp);
    
    calculateImagePlane();
}

void RayTracer::calculateImagePlane() {
    Vect viewDir = getViewDirection();
    Vect orthoUp = getOrthogonalUp();
    Vect cameraPos = getCameraPos();

    parallelRight = viewDir.crossProduct(orthoUp);
    parallelUp = parallelRight.crossProduct(viewDir);
    parallelRight.normalize();
    parallelUp.normalize();

    imageCenter = cameraPos + viewDir.linearMult(scaleConst);
}

Vect RayTracer::getImageCenter() {
    return imageCenter;
}

double RayTracer::getHorizontalFOV() {
    double horiFov = ((float) WIDTH / (float) HEIGHT) * getVerticalFOV();
    if (horiFov >= M_PI)
        throw "FOV too large";
    return horiFov;
}

Vect RayTracer::vertical() {
    float f = (float) tan(getVerticalFOV() / 2) * scaleConst;
    Vect vert = parallelUp.linearMult(f);
    return vert;
}

Vect RayTracer::horizontal() {
    float f = (float) tan(getHorizontalFOV() / 2) * scaleConst;
    Vect hori = parallelRight.linearMult(f);
    return hori;
}

Point_2D RayTracer::computePoint(uint x, uint y) {
    if (!(x < WIDTH && y < HEIGHT))
        throw "Coords out of bounds";

    Point_2D pt;
    pt.x = (float)(x + 0.5) * (1 / (float) WIDTH);
    pt.y = (float)(y + 0.5) * (1 / (float) HEIGHT);
    return pt;
}

Vect RayTracer::computeDirection(uint x, uint y) {
    Point_2D p = computePoint(x, y);
    Vect dx = (horizontal()).linearMult(2 * p.x - 1);
    Vect dy = (vertical()).linearMult(2 * p.y - 1);

    Vect dir = imageCenter + dx + dy;
    // dir.normalize(); TODO: fix tests
    return dir;
}

Ray RayTracer::computeRay(uint x, uint y) {
    Vect dir = computeDirection(x, y);
    Ray r(getCameraPos(), dir);
    return r;
}

RayBuffer RayTracer::traceRays() {
    for (uint y = 0; y < HEIGHT; y++) {
        for (uint x = 0; x < WIDTH; x++) {
            Ray r = computeRay(x, y);
            Intersection is = scene.calculateRayIntersection(r);
            if (is.hasIntersected()) {
                PX_Color c = {255, 255, 255};
                buffer.setPixel(x, y, c);
            } else {
                PX_Color c = {0, 0, 0};
                buffer.setPixel(x, y, c);
            }
        }
    }
    return buffer;
}
