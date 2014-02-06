#include <stdio.h>
#include <stdlib.h>
#include "scene_io.h"

#include "raytracer.h"
#include "rayscene_factory.h"
#include "rayimage.h"

#define IMAGE_WIDTH     500
#define IMAGE_HEIGHT    500


typedef unsigned char u08;

SceneIO *scene = NULL;


static void loadScene(const char *name) {
    /* load the scene into the SceneIO data structure using given parsing code */
    scene = readScene(name);

    /* hint: use the Visual Studio debugger ("watch" feature) to probe the
       scene data structure and learn more about it for each of the given scenes */
    RayScene rayScene;
    RaySceneFactory::CreateScene(rayScene, *scene);

    RayTracer rayTracer(IMAGE_WIDTH, IMAGE_HEIGHT);
    rayTracer.setScene(rayScene);

    RayBuffer rayBuffer = rayTracer.traceRays();

    RayImage img;
    img.createImage(rayBuffer, "scene1.bmp");
    return;
}


/* just a place holder, feel free to edit */
void render(void) {


}



int main(int argc, char *argv[]) {
    // Timer total_timer;
    // total_timer.startTimer();
    try {
        loadScene("./scenes/test1.ascii");

        /* write your ray tracer here */
        render();

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
