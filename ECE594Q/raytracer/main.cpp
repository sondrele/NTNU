#include <stdlib.h>
#include <iostream>

#include "raytracer.cpp"

int main(int argc, char const *argv[]) {

    Shape *s = RaySceneFactory::NewSphere(10, Vect(0, 0, -20));
    RayScene rayScene;

    RayTracer rayTracer(IMAGE_WIDTH, IMAGE_HEIGHT);
    rayTracer.setScene(rayScene);

    RayBuffer rayBuffer = rayTracer.traceRays();

    RayImage img;
    img.createImage(rayBuffer, "scene1.bmp");
    return 0;
}