#ifndef _RSCENEFACTORY_H_
#define _RSCENEFACTORY_H_ value

#include <vector>
#include <assert.h>

#include "scene_io.h"
#include "tiny_obj_loader.h"

#include "raybuffer.h"
#include "ray.h"
#include "camera.h"
#include "rscene.h"
#include "material.h"
#include "shapes.h"

class RSceneFactory {
public:
    // Parse scene_io object
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
    static void CreateScene(RScene &s, SceneIO &sio);

    // Parse obj
    static Mesh * CreateMeshFromObj(tinyobj::mesh_t, Material *);
    static Material * CreateMaterialFromObj(tinyobj::material_t);
    static Shape * CreateShapeFromObj(tinyobj::shape_t);
    static void CreateShapesFromObj(std::vector<Shape *> &, std::vector<tinyobj::shape_t> &);
    static void CreateSceneFromObj(RScene *, std::vector<tinyobj::shape_t> &);
};

#endif // _RSCENEFACTORY_H_
