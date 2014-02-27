#include <stdio.h>
#include <stdlib.h>
// #include <chrono>
#include "scene_io.h"

#include "raytracer.h"
#include "rayscene_factory.h"
#include "rayimage.h"

#define WHITTED         "./scenes-whitted/"
#define MONTE           "./scenes-montecarlo/"
#define ASCII           ".ascii"
#define IMG             ".bmp"
#define IMAGE_WIDTH     500
#define IMAGE_HEIGHT    500

typedef unsigned char u08;

SceneIO *scene = NULL;

uint w, h, d, m = 1;
bool shaders = false;
std::string in;
std::string out;

static void loadScene(const char *name, RayTracer &rayTracer) {
    // auto start = std::chrono::system_clock::now();
    
    /* load the scene into the SceneIO data structure using given parsing code */
    scene = readScene(name);
    RayScene *rayScene = new RayScene();
    RaySceneFactory::CreateScene(*rayScene, *scene);

    Camera cam;
    RaySceneFactory::CreateCamera(cam, *(scene->camera));
    rayTracer.setCamera(cam);
    rayTracer.setScene(rayScene);

    // auto end = std::chrono::system_clock::now();
    // auto elapsed =
    //     std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "Preprocess time: " << elapsed.count() << std::endl;
}

static void loadShaderScene(const char *name, RayTracer &rayTracer) {
    /* load the scene into the SceneIO data structure using given parsing code */
    scene = readScene(name);
    RayScene *rayScene = new RayScene();
    RaySceneFactory::CreateScene(*rayScene, *scene);
    
    // Load the textures and shaders for the demo scene
    Texture *jupterTexture = new Texture();
    jupterTexture->loadTexture(std::string("textures/jupiter.bmp"));
    Sphere *s0 = (Sphere *) rayScene->getShape(0);
    s0->setTexture(jupterTexture);

    Texture *earthTexture = new Texture();
    earthTexture->loadTexture(std::string("textures/earth.bmp"));
    Sphere *s1 = (Sphere *) rayScene->getShape(1);
    s1->setTexture(earthTexture);

    Sphere *s2 = (Sphere *) rayScene->getShape(2);
    s2->setCShader(new FunCShader());


    Texture *venusTexture = new Texture();
    venusTexture->loadTexture(std::string("textures/venus.bmp"));
    Sphere *s3 = (Sphere *) rayScene->getShape(3);
    s3->setTexture(venusTexture);
    // s3->setIShader(new CheckIShader());

    Camera cam;
    RaySceneFactory::CreateCamera(cam, *(scene->camera));
    rayTracer.setCamera(cam);
    rayTracer.setScene(rayScene);
}

static void render(RayTracer &rayTracer) {
    rayTracer.setM(m);

    // auto start = std::chrono::system_clock::now();

    RayBuffer rayBuffer = rayTracer.traceRaysWithAntiAliasing();
    
    // auto end = std::chrono::system_clock::now();
    // auto elapsed =
    //     std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "Tracing time: " << elapsed.count() << std::endl;

    RayImage img;
    img.createImage(rayBuffer, out.c_str());
}

static void parseInput(int argc, char *argv[]) {
    cout << "Normal usage: Run with 5 arguments: scene, width, height, depth, m || w" << endl;
    cout << "Example:      ./montecarlo test1 100 100 10 w" << endl;
    cout << "Shader demo:  ./montecarlo shaders" << endl;

    if (argc >= 6) {
        in = (*argv[5] == 'm') ? std::string(MONTE) : std::string(WHITTED);
        in += std::string(argv[1]) + std::string(ASCII);
        out = std::string(argv[1]) + std::string(IMG);
        w = atoi(argv[2]);
        h = atoi(argv[3]);
        d = atoi(argv[4]);
    } else if (argc == 2) {
        w = IMAGE_WIDTH;
        h = IMAGE_HEIGHT;
        d = 10;
        in = std::string(WHITTED) + std::string("shaders.ascii");
        out = std::string("shaders.bmp");
        shaders = true;
    } else {
        w = IMAGE_WIDTH;
        h = IMAGE_HEIGHT;
        d = 2;
        in = std::string(MONTE) + std::string("test1.ascii");
        out = std::string("test1.bmp");
    }

    if (argc == 7) {
        m = 2;
    }
}

int main(int argc, char *argv[]) {
    parseInput(argc, argv);

    // auto start = std::chrono::system_clock::now();

    try {
        RayTracer rayTracer(w, h, d);
        // Load the scene
        if (shaders) {
            loadShaderScene(in.c_str(), rayTracer);
        } else {
            loadScene(in.c_str(), rayTracer);
        }

        // Render the scene
        render(rayTracer);

        /* cleanup */
        if (scene != NULL) {
            deleteScene(scene);
        }
    } catch (const char *str) {
        cout << str << endl;
    }

    // auto end = std::chrono::system_clock::now();
    // auto elapsed =
    //     std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "Total time: " << elapsed.count() << std::endl;
    return 0;
}
