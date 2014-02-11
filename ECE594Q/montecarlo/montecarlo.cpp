#include <stdio.h>
#include <stdlib.h>
#include "scene_io.h"

#include "raytracer.h"
#include "rayscene_factory.h"
#include "rayimage.h"

#define PATH            "./scenes-whitted/"
#define ASCII           ".ascii"
#define IMG             ".bmp"
#define IMAGE_WIDTH     220
#define IMAGE_HEIGHT    220

typedef unsigned char u08;

SceneIO *scene = NULL;

uint w, h, d;
std::string in;
std::string out;

static void loadScene(const char *name, RayTracer &rayTracer) {
    /* load the scene into the SceneIO data structure using given parsing code */
    scene = readScene(name);
    
    RayScene *rayScene = new RayScene();
    RaySceneFactory::CreateScene(*rayScene, *scene);

    Camera cam;
    RaySceneFactory::CreateCamera(cam, *(scene->camera));
    rayTracer.setCamera(cam);
    rayTracer.setScene(rayScene);
}

static void render(RayTracer &rayTracer) {
    RayBuffer rayBuffer = rayTracer.traceRays();
    RayImage img;
    img.createImage(rayBuffer, out.c_str());
}

static void parse_input(int argc, char *argv[]) {
    if (argc == 5) {
        in = std::string(PATH) + std::string(argv[1]) + std::string(ASCII);
        out = std::string(argv[1]) + std::string(IMG);
        w = atoi(argv[2]);
        h = atoi(argv[3]);
        d = atoi(argv[4]);
    } else {
        cout << "usage: ./montecarlo test1.ascii w h d" << endl;
        w = IMAGE_WIDTH;
        h = IMAGE_HEIGHT;
        d = 2;
        in = std::string(PATH) + std::string("test1.ascii");
        out = std::string("test1.bmp");
    }
}

int main(int argc, char *argv[]) {
    parse_input(argc, argv);

    try {

        RayTracer rayTracer(w, h, d);
        loadScene(in.c_str(), rayTracer);

        /* write your ray tracer here */
        render(rayTracer);

        /* cleanup */
        if (scene != NULL) {
            deleteScene(scene);
        }
    } catch (const char *str) {
        cout << str << endl;
    }

    return 1;
}
