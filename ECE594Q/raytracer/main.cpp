#include <stdio.h>
#include <stdlib.h>
#include "scene_io.h"

#include "raytracer.h"
#include "rayscene_factory.h"
#include "rayimage.h"

#define IMAGE_WIDTH     200
#define IMAGE_HEIGHT    200

int main(int argc, char const *argv[]) {

    std::vector<Shape *> v;
    v.push_back(RaySceneFactory::NewSphere(10, Vect(0, 0, -20)));
    v.push_back(RaySceneFactory::NewSphere(10, Vect(10, 10, -20)));
    RayScene *rayScene = new RayScene();
    rayScene->setShapes(v);

    RayTracer rayTracer(IMAGE_WIDTH, IMAGE_HEIGHT);
    Camera cam;
    cam.setPos(Vect(0, 0, 10));
    cam.setVerticalFOV((float)M_PI / 2.0f);
    cam.setViewDir(Vect(0, 0, -1));
    cam.setOrthoUp(Vect(0, 1, 0));

    rayTracer.setCamera(cam);
    rayTracer.setScene(rayScene);

    RayBuffer rayBuffer = rayTracer.traceRays();

    RayImage img;
    img.createImage(rayBuffer, "output.bmp");

    return 0;
}
