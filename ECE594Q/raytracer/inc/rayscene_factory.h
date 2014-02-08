#ifndef _RAYSCENE_FACTORY_H_
#define _RAYSCENE_FACTORY_H_ value

#include <vector>

#include "scene_io.h"
#include "raybuffer.h"
#include "ray.h"
#include "camera.h"
#include "rayscene.h"
#include "rayscene_shapes.h"

class RaySceneFactory {
public:
    static PX_Color ColorToPX_Color(Color);
    static Vect ColorToVect(Color c);
    static Vect PointToVect(Point);
    static Sphere * NewSphere(float, Vect);
    static void CreateMaterial(Material &m, MaterialIO &mio);
    static void CreateSphere(Sphere &s, SphereIO &sio);
    static void CreateTriangle(Triangle &t, PolygonIO &pio);
    static void CreateMesh(Mesh &m, PolySetIO &pio);
    static void CreateLight(Light &, LightIO &);
    static void CreateLights(std::vector<Light> &, LightIO &);
    static void CreateCamera(Camera &c, CameraIO &cio);
    static void CreateScene(RayScene &s, SceneIO &sio);
    static Shape * CreateShape(ObjIO &oio);
    static void CreateShapes(std::vector<Shape *> &shps, ObjIO &oio);
};

#endif
