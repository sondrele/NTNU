#include <stdio.h>
#include <stdlib.h>
#include "scene_io.h"

#include "raytracer.h"
#include "rayscene_factory.h"
#include "rayimage.h"

#define IMAGE_WIDTH     220
#define IMAGE_HEIGHT    220


typedef unsigned char u08;

SceneIO *scene = NULL;


static void loadScene(const char *name, RayTracer &rt) {
    /* load the scene into the SceneIO data structure using given parsing code */
    scene = readScene(name);

    /* hint: use the Visual Studio debugger ("watch" feature) to probe the
       scene data structure and learn more about it for each of the given scenes */
    RayScene *rayScene = new RayScene();
    RaySceneFactory::CreateScene(*rayScene, *scene);

    Camera cam;
    RaySceneFactory::CreateCamera(cam, *(scene->camera));
    rt.setCamera(cam);
    rt.setScene(rayScene);
    return;
}


/* just a place holder, feel free to edit */
void render(void) {


}



int main(int argc, char *argv[]) {
    // Timer total_timer;
    // total_timer.startTimer();
    try {
        RayTracer rayTracer(IMAGE_WIDTH, IMAGE_HEIGHT, 10);

        loadScene("./scenes/test1.ascii", rayTracer);

        /* write your ray tracer here */
        render();

        RayBuffer rayBuffer = rayTracer.traceRays();
        RayImage img;
        img.createImage(rayBuffer, "test1.bmp");

        /* cleanup */
        if (scene != NULL) {
            deleteScene(scene);
        }
    } catch (const char *str) {
        cout << str << endl;
    }

    // total_timer.stopTimer();
    // fprintf(stderr, "Total time: %.5lf \n", total_timer.getTime());
    
    return 1;
}
