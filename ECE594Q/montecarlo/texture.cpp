#include <stdio.h>
#include <stdlib.h>
#include "scene_io.h"

#include "raytracer.h"
#include "rayscene_factory.h"
#include "rayimage.h"

#define WHITTED         "./scenes-whitted/"
#define MONTE           "./scenes-montecarlo/"
#define ASCII           ".ascii"
#define IMG             ".bmp"
#define IMAGE_WIDTH     100
#define IMAGE_HEIGHT    100

typedef unsigned char u08;

SceneIO *scene = NULL;

uint w, h, d;
std::string in;
std::string out;

static void loadTextures(RayScene *rayScene) {
    std::string t = std::string("earth.bmp");
    Texture *text = new Texture();
    text->loadTexture(t);
    // Sphere *s = (Sphere *) rayScene->getShape(0);
    Sphere *s0 = (Sphere *) rayScene->getShape(0);
    // s0->setIShader(new CheckIShader());
    s0->setTexture(text);
    s0->setCShader(new CheckCShader());
}

static void loadScene(const char *name, RayTracer &rayTracer) {
    /* load the scene into the SceneIO data structure using given parsing code */
    scene = readScene(name);
    
    RayScene *rayScene = new RayScene();
    RaySceneFactory::CreateScene(*rayScene, *scene);

    loadTextures(rayScene);

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
    if (argc == 6) {
        in = (*argv[5] == 'm') ? std::string(MONTE) : std::string(WHITTED);
        in += std::string(argv[1]) + std::string(ASCII);
        out = std::string(argv[1]) + std::string(IMG);
        w = atoi(argv[2]);
        h = atoi(argv[3]);
        d = atoi(argv[4]);
    } else {
        cout << "usage: ./montecarlo flaot:w/m name:test1.ascii width:100 height:100 depth:10" << endl;
        w = IMAGE_WIDTH;
        h = IMAGE_HEIGHT;
        d = 2;
        in = std::string(MONTE) + std::string("test1.ascii");
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
