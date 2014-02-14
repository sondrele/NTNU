#ifndef _RAYSCENE_FACTORY_H_
#define _RAYSCENE_FACTORY_H_ value

#include <vector>

#include "scene_io.h"
#include "raybuffer.h"
#include "ray.h"
#include "camera.h"
#include "rayscene.h"
#include "material.h"
#include "rayscene_shapes.h"

class RaySceneFactory {
public:
    static PX_Color ColorToPX_Color(Color);
    static Vect PointToVect(Point);
    static Vertex PointToVertex(Point);
    
    static Light * CreateLight(LightIO &);
    static void CreateLights(std::vector<Light *> &, LightIO &);
    
    static Sphere * NewSphere(float, Vect);
    static void CreateSphere(Sphere &s, SphereIO &sio);
    
    static Vertex CreateVertex(VertexIO &);
    static Vertex CreateVertexWithBindings(VertexIO &);
    static Triangle* CreateTriangleWithBindings(PolygonIO &);
    static Triangle * CreateTriangle(PolygonIO &);
    
    static void CreateMesh(Mesh &, PolySetIO &, std::vector<Material*>);
    
    static Material * CreateMaterial(MaterialIO &mio);
    static std::vector<Material *> CreateMaterials(MaterialIO *mio, long numMaterials);
    static void AddMaterials(Shape*, std::vector<Material*>);
    
    static Shape * CreateShape(ObjIO &oio);
    static void CreateShapes(std::vector<Shape *> &shps, ObjIO &oio);
    
    static void CreateCamera(Camera &c, CameraIO &cio);
    
    static void CreateScene(RayScene &s, SceneIO &sio);
};

#endif
