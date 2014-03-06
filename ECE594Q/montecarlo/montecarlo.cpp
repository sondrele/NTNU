#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#include "scene_io.h"
#include "tiny_obj_loader.h"
#include "rand.h"

#include "raytracer.h"
#include "rscenefactory.h"
#include "rimage.h"

#define SCENES          "./scenes/"
#define ASCII           ".ascii"
#define OBJ             ".obj"
#define IMG             ".bmp"
#define IMAGE_WIDTH     500
#define IMAGE_HEIGHT    500

typedef unsigned char u08;

SceneIO *scene = NULL;

uint w, h, d, m = 1;
std::string in;
std::string inObj;
std::string out;

bool shaders = false;
bool objScene = false;
bool pathTracing = false;

static void loadScene(const char *name, RayTracer &rayTracer) {
    // auto start = std::chrono::system_clock::now();
    
    /* load the scene into the SceneIO data structure using given parsing code */
    scene = readScene(name);
    RScene *rayScene = new RScene();
    RSceneFactory::CreateScene(*rayScene, *scene);

    Camera cam;
    RSceneFactory::CreateCamera(cam, *(scene->camera));
    rayTracer.setCamera(cam);
    rayTracer.setScene(rayScene);
    rayTracer.loadEnvMap("textures/uffizi_latlong.exr");

    // auto end = std::chrono::system_clock::now();
    // auto elapsed =
    //     std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "Preprocess time: " << elapsed.count() << std::endl;
}

static void loadObjScene(const char *name, RayTracer &rayTracer) {
    std::vector<tinyobj::shape_t> shapes;
    scene = readScene(name);
    std::string err = tinyobj::LoadObj(shapes, inObj.c_str(), SCENES);

    if (!err.empty()) {
      std::cerr << err << std::endl;
      exit(1);
    }

    RScene *rayScene = new RScene();
    RSceneFactory::CreateSceneFromObj(rayScene, shapes);


    std::vector<Light *> lts;
    RSceneFactory::CreateLights(lts, *(scene->lights));
    rayScene->setLights(lts);

    Camera cam;
    RSceneFactory::CreateCamera(cam, *(scene->camera));
    rayTracer.setCamera(cam);
    rayTracer.setScene(rayScene);
    rayTracer.loadEnvMap("textures/doge2_latlong.exr");

    // auto end = std::chrono::system_clock::now();
    // auto elapsed =
    //     std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "Preprocess time: " << elapsed.count() << std::endl;
}

static void loadShaderScene(const char *name, RayTracer &rayTracer) {
    /* load the scene into the SceneIO data structure using given parsing code */
    scene = readScene(name);
    RScene *rayScene = new RScene();
    RSceneFactory::CreateScene(*rayScene, *scene);
    
    // Load the textures and shaders for the demo scene
    Texture *jupterTexture = new Texture();
    jupterTexture->loadImage(std::string("textures/jupiter.bmp"));
    Sphere *s0 = (Sphere *) rayScene->getShape(0);
    s0->setTexture(jupterTexture);

    Texture *earthTexture = new Texture();
    earthTexture->loadImage(std::string("textures/earth.bmp"));
    Sphere *s1 = (Sphere *) rayScene->getShape(1);
    s1->setTexture(earthTexture);

    Sphere *s2 = (Sphere *) rayScene->getShape(2);
    s2->setCShader(new FunCShader());


    Texture *venusTexture = new Texture();
    venusTexture->loadImage(std::string("textures/venus.bmp"));
    Sphere *s3 = (Sphere *) rayScene->getShape(3);
    s3->setTexture(venusTexture);
    // s3->setIShader(new CheckIShader());

    Camera cam;
    RSceneFactory::CreateCamera(cam, *(scene->camera));
    rayTracer.setCamera(cam);
    rayTracer.setScene(rayScene);
}

static void render(RayTracer &rayTracer) {
    rayTracer.setM(m);
    rayTracer.setNumSamples(m);

    auto start = std::chrono::system_clock::now();

    RayBuffer rayBuffer;
    if (!pathTracing) {
        rayBuffer = rayTracer.traceRays();
    } else {
        rayBuffer = rayTracer.tracePaths();
    }
    
    auto end = std::chrono::system_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Tracing time: " << elapsed.count() << std::endl;

    RImage img;
    img.createImage(rayBuffer, out);
}

static void parseInput(int argc, char *argv[]) {
    cout << "Normal usage:\nRun with 5 arguments: scene, width, height, depth, numSamples" << endl;
    cout << "Example (whitted illumination):  ./montecarlo test1 100 100 10" << endl;
    cout << "Example (path tracing):          ./montecarlo test1 100 100 10 10" << endl;
    cout << "Shader demo:  ./montecarlo shaders" << endl << endl;

    if (argc >= 5) {
        in = std::string(SCENES) + std::string(argv[1]) + std::string(ASCII);
        out = std::string(argv[1]) + std::string(IMG);
        w = atoi(argv[2]);
        h = atoi(argv[3]);
        d = atoi(argv[4]);
        if (argc >= 6) {
            inObj = std::string(SCENES) + std::string(argv[1]) + std::string(OBJ);
            cout << "Obj: " << inObj << endl;
            objScene = true;
        }
        if (argc >= 7) {
            m = atoi(argv[5]);
            pathTracing = true;
        }
    } else if (argc == 2) {
        w = IMAGE_WIDTH;
        h = IMAGE_HEIGHT;
        d = 10;
        in = std::string(SCENES) + std::string("shaders.ascii");
        out = std::string("shaders.bmp");
        shaders = true;
    } else {
        w = IMAGE_WIDTH;
        h = IMAGE_HEIGHT;
        d = 2;
        in = std::string(SCENES) + std::string("whitted1.ascii");
        out = std::string("whitted1.bmp");
    }
    cout << "Scene: " << in << endl;
    cout << "Output: " << out << endl;

    cout << endl;
}

int main(int argc, char *argv[]) {
    parseInput(argc, argv);

    // auto start = std::chrono::system_clock::now();

    try {
        RayTracer rayTracer(w, h, d);
        // Load the scene
        if (objScene) {
            loadObjScene(in.c_str(), rayTracer);
        } else if (shaders) {
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
