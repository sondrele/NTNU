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
    static void CreateSphere(Sphere &, SphereIO &);
    static Vertex CreateVertex(VertexIO &);
    static Vertex CreateVertexWithBindings(VertexIO &);
    static Triangle* CreateTriangleWithBindings(PolygonIO &);
    static Triangle * CreateTriangle(PolygonIO &);
    static void CreateMesh(Mesh &, PolySetIO &, std::vector<Material *>);
    static Material * CreateMaterial(MaterialIO &);
    static std::vector<Material *> CreateMaterials(MaterialIO *, long);
    static void AddMaterials(Shape*, std::vector<Material*>);
    static Shape * CreateShape(ObjIO &);
    static void CreateShapes(std::vector<Shape *> &, ObjIO &);
    static void CreateCamera(Camera &, CameraIO &);
    static void CreateScene(RScene &, SceneIO &);

    // Parse obj
    static Mesh * CreateMeshFromObj(tinyobj::mesh_t, Material *);
    static Material * CreateMaterialFromObj(tinyobj::material_t);
    static Shape * CreateShapeFromObj(tinyobj::shape_t);
    static void CreateShapesFromObj(std::vector<Shape *> &, std::vector<tinyobj::shape_t> &);
    static void CreateSceneFromObj(RScene *, std::vector<tinyobj::shape_t> &);
};

#endif // _RSCENEFACTORY_H_