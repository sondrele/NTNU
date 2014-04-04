#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#include "scene_io.h"
#include "tiny_obj_loader.h"
#include "rand.h"
#include "argparser.h"

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

uint w, h, d, numSamples = 1;
float fattjScale = 1.0f;
std::string in;
std::string inObj;
std::string out;

bool shaders = false;
bool area = false;
bool objScene = false;
bool pathTracing = false;
bool bidirectional = false;
bool environmentMap = false;

static void loadScene(const char *name, RayTracer *rayTracer) {
    scene = readScene(name);
    RScene *rayScene = new RScene();
    RSceneFactory::CreateScene(*rayScene, *scene);

    if (area) {
        Light *l = new Light();
        l->setIntensity(SColor(1, 1, 1));
        l->setArea(Vect(1, 5, -2), Vect(1.2f, 5, -2.2f));
        rayScene->addLight(l);
    }

    Camera cam;
    RSceneFactory::CreateCamera(cam, *(scene->camera));
    rayTracer->setCamera(cam);
    rayTracer->setScene(rayScene);
    if (environmentMap) {
        rayTracer->loadEnvMap("textures/uffizi_latlong.exr");
    }
}

static void loadObjScene(const char *name, RayTracer *rayTracer) {
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
    if (area) {
        Light *l = new Light();
        l->setIntensity(SColor(1, 1, 1));
        l->setArea(Vect(213, 540, 227), Vect(343, 540, 332));
        rayScene->addLight(l);
    }

    Camera cam;
    RSceneFactory::CreateCamera(cam, *(scene->camera));
    rayTracer->setCamera(cam);
    rayTracer->setScene(rayScene);
    if (environmentMap) {
        rayTracer->loadEnvMap("textures/doge2_latlong.exr");
    }
}

static void loadShaderScene(const char *name, RayTracer *rayTracer) {
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
    rayTracer->setCamera(cam);
    rayTracer->setScene(rayScene);
}

static void render(RayTracer *rayTracer) {
    rayTracer->setNumSamples(numSamples);

    auto start = std::chrono::system_clock::now();
    RayBuffer rayBuffer = rayTracer->traceScene();
    
    auto end = std::chrono::system_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Tracing time: " << elapsed.count() << std::endl;

    RImage img;
    img.createImage(rayBuffer, out);
}

static void parseCommandLine(int argc, char *argv[]) {
    cout << "help: ./tracer -h" << endl;
    if (ArgParser::CmdOptExists(argv, argv+argc, "-h")) {
        cout << "Run with following parameters: " << endl;
        cout << "-h:                Display help message" << endl;
        cout << "-scene scene_name: Specify which scene to load" << endl;
        cout << "-out img_name:     Specify the name of the output image" << endl;
        cout << "-obj:              Set if the scene is in .obj format" << endl;
        cout << "-bi:               Specifies that bidirectional path tracing should be used" << endl;
        cout << "-env:              Load uffizi environment map" << endl;
        cout << "-size num_pixels:  Image size, width and height" << endl;
        cout << "-d ray_depth:      Specify depth of reflective rays" << endl;
        cout << "-n num_samples:    Specify number of samples for pathtracing" << endl;
        cout << "-f:                Specify a scale constant for Fattj" << endl;
        exit(0);
    }

    if (ArgParser::CmdOptExists(argv, argv+argc, "-area")) {
        area = true;
    }

    char *size = ArgParser::GetCmdOpt(argv, argv + argc, "-size");
    if (size) {
        w = atoi(size);
        h = atoi(size);
    } else {
        w = IMAGE_WIDTH;
        h = IMAGE_HEIGHT;
    }

    if (ArgParser::CmdOptExists(argv, argv+argc, "-bi")) {
        bidirectional = true;
    }

    if (ArgParser::CmdOptExists(argv, argv+argc, "-obj")) {
        objScene = true;
    }

    if (ArgParser::CmdOptExists(argv, argv+argc, "-env")) {
        environmentMap = true;
    }

    char *scene = ArgParser::GetCmdOpt(argv, argv + argc, "-scene");
    if (scene) {
        if (objScene) {
            inObj = std::string(SCENES) + std::string(scene) + std::string(OBJ);
        }
        in = std::string(SCENES) + std::string(scene) + std::string(ASCII);
        out = std::string(scene) + std::string(IMG);
    } else {
        in = std::string(SCENES) + std::string("whitted1.ascii");
        out = std::string("whitted1.bmp");
    }

    char *outname = ArgParser::GetCmdOpt(argv, argv + argc, "-out");
    if (outname) {
        out = std::string(outname) + std::string(IMG);
    }

    char *depth = ArgParser::GetCmdOpt(argv, argv + argc, "-d");
    if (depth) {
        d = atoi(depth);
    } else {
        d = 5;
    }

    char *samples = ArgParser::GetCmdOpt(argv, argv + argc, "-n");
    if (samples) {
        numSamples = atoi(samples);
        pathTracing = true;
    } else {
        numSamples = 1;
    }

    char *fs = ArgParser::GetCmdOpt(argv, argv + argc, "-f");
    if (fs) {
        fattjScale = atof(fs);
    }
}

int main(int argc, char *argv[]) {
    parseCommandLine(argc, argv);

    try {
        RayTracer *rayTracer;
        if (pathTracing) {
            PathTracer *pathTracer = new PathTracer(w, h, d);
            if (bidirectional) {
                pathTracer->setBidirectional(true);
            }
            rayTracer = pathTracer;
        } else {
            rayTracer = new WhittedTracer(w, h, d);
        }
        rayTracer->setFattjScale(fattjScale);

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
            delete rayTracer;
        }
    } catch (const char *str) {
        cout << str << endl;
    }
    return 0;
}
