#include "raytracer.h"

Ray::Ray() {
    ;
}

Ray::Ray(Vect o, Vect d) {
    origin = o;
    direction = d;
}

RayTracer::RayTracer(uint width, uint height) {
    WIDTH = width;
    HEIGHT = height;

    verticalFOV = M_PI / 2.0;
    scaleConst = 2.0;
    // rt.setViewDirection(Vect(0, 0, 1));
    // rt.setOrthogonalUp(Vect(0, 1, 0));
}

RayTracer::RayTracer(uint width, uint height, Vect viewDir, Vect orthoUp) {
    WIDTH = width;
    HEIGHT = height;

    verticalFOV = M_PI / 2.0;
    scaleConst = 2.0;

    setViewDirection(viewDir);
    setOrthogonalUp(orthoUp);
}

void RayTracer::setCamera(Vect cam) {
    cameraPos = cam;
    cameraPos.normalize();
}

void RayTracer::setViewDirection(Vect viewDir) {
    viewDirection = viewDir;
    viewDirection.normalize();

    calculateImagePlane();
}

void RayTracer::setOrthogonalUp(Vect orthoUp) {
    orthogonalUp = orthoUp;
    orthogonalUp.normalize();
    
    calculateImagePlane();
}

void RayTracer::calculateImagePlane() {
    parallelRight = viewDirection.crossProduct(orthogonalUp);
    parallelUp = parallelRight.crossProduct(viewDirection);
    parallelRight.normalize();
    parallelUp.normalize();

    imageCenter = cameraPos + viewDirection.linearMult(scaleConst);
}

double RayTracer::getHorizontalFOV() {
    double horiFov = ((float) WIDTH / (float) HEIGHT) * verticalFOV;
    if (horiFov >= M_PI)
        throw "FOV too large";
    return horiFov;
}

Vect RayTracer::vertical() {
    float f = (float) tan(verticalFOV / 2) * scaleConst;
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

    Vect pt = imageCenter + dx + dy;
    return pt;
}

Ray RayTracer::computeRay(uint x, uint y) {
    Vect dir = computeDirection(x, y);
    Ray r(cameraPos, dir);
    return r;
}
