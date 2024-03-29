#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"
#include "rscene.h"
#include "rscenefactory.h"

SphereIO sphere;
PolygonIO triangle0, triangle1;
PolySetIO trimesh;
PolySetIO singleTrimesh;
ObjIO objects;
MaterialIO material;
LightIO light0;
LightIO light1;
CameraIO camera;
SceneIO scene;

TEST_GROUP(RSceneFactory) {
    void setup() {
        sphere.origin[0] = -2.11537f;
        sphere.origin[1] = -0.766425f;
        sphere.origin[2] = -3.86329f;
        sphere.radius = 1.33453f;

        sphere.xaxis[0] = 1;
        sphere.xaxis[1] = 0;
        sphere.xaxis[2] = 0;
        sphere.xlength = 1.33453f;
        
        sphere.yaxis[0] = 0;
        sphere.yaxis[1] = 1;
        sphere.yaxis[2] = 0;
        sphere.ylength = 1.33453f;

        sphere.zaxis[0] = 0;
        sphere.zaxis[1] = 0;
        sphere.zaxis[2] = 1;
        sphere.zlength = 1.33453f;

        triangle0.numVertices = 3;
        triangle0.vert = new VertexIO[3];
        triangle0.vert[1].pos[0] = 0.4f;
        triangle0.vert[1].pos[1] = 0.5f;
        triangle0.vert[1].pos[2] = 0.6f;
        triangle0.vert[1].materialIndex = 0;
        triangle0.vert[0].materialIndex = 0;
        triangle0.vert[0].norm[0] = 0;
        triangle0.vert[0].norm[1] = 2;
        triangle0.vert[0].norm[2] = 0;
        triangle0.vert[1].norm[0] = 1;
        triangle0.vert[1].norm[1] = 0;
        triangle0.vert[1].norm[2] = 0;
        triangle0.vert[0].materialIndex = 0;
        triangle0.vert[1].materialIndex = 1;
        triangle0.vert[2].materialIndex = 1;


        triangle1.numVertices = 3;
        triangle1.vert = new VertexIO[3];
        triangle1.vert[0].pos[0] = 0.4f;
        triangle1.vert[0].pos[1] = 0.5f;
        triangle1.vert[0].pos[2] = 0.6f;
        triangle1.vert[0].norm[0] = 0;
        triangle1.vert[0].norm[1] = 2;
        triangle1.vert[0].norm[2] = 0;
        triangle1.vert[1].norm[0] = 1;
        triangle1.vert[1].norm[1] = 0;
        triangle1.vert[1].norm[2] = 0;
        triangle1.vert[0].materialIndex = 0;
        triangle1.vert[1].materialIndex = 1;
        triangle1.vert[2].materialIndex = 2;

        trimesh.numPolys = 2;
        trimesh.materialBinding = PER_OBJECT_MATERIAL;
        trimesh.poly = new PolygonIO[2];
        trimesh.poly[0] = triangle0;
        trimesh.poly[1] = triangle1;


        light1.next = NULL;
        light1.type = POINT_LIGHT;
        light1.position[0] = 0;
        light1.position[1] = 1;
        light1.position[2] = 2;
        light1.color[0] = 0;
        light1.color[1] = 0.5;
        light1.color[2] = 1;

        light0.next = &light1;
        light0.type = DIRECTIONAL_LIGHT;
        light0.direction[0] = 2;
        light0.direction[1] = 2;
        light0.direction[2] = 2;
        light0.color[0] = 0;
        light0.color[1] = 0.5;
        light0.color[2] = 1;

        camera.position[0] = 1;
        camera.position[1] = 2;
        camera.position[2] = 3;
        camera.viewDirection[0] = 3;
        camera.viewDirection[1] = 3;
        camera.viewDirection[2] = 3;
        camera.focalDistance = 0.1f;
        camera.orthoUp[0] = 4;
        camera.orthoUp[1] = 4;
        camera.orthoUp[2] = 4;
        camera.verticalFOV = 0.2f;

        scene.camera = &camera;
        scene.lights = &light0;

        material.diffColor[0] = 0.2f;
        material.diffColor[1] = 0.2f;
        material.diffColor[2] = 0.2f;
        material.shininess = 1;
        material.ktran = 0.5f;
    }
    void teardown() {
        delete [] triangle0.vert;
        delete [] triangle1.vert;
        delete [] trimesh.poly;
    }
};

TEST(RSceneFactory, can_init_sphere) {
    Sphere s;
    RSceneFactory::CreateSphere(s, sphere);

    CHECK_EQUAL(1.33453f, s.getRadius());
    CHECK_EQUAL(1.33453f, s.getXlen());
    CHECK_EQUAL(1.33453f, s.getYlen());
    CHECK_EQUAL(1.33453f, s.getZlen());
    Vect x = s.getX();
    CHECK_EQUAL(1, x.getX());
    CHECK_EQUAL(0, x.getY());
    CHECK_EQUAL(0, x.getZ());
}

TEST(RSceneFactory, can_convert_color_to_PX_Color) {
    Color c;
    c[0] = 0;
    c[1] = 0.5;
    c[2] = 1;
    PX_Color color = RSceneFactory::ColorToPX_Color(c);
    CHECK_EQUAL(0, color.R);
    CHECK_EQUAL(127, color.G);
    CHECK_EQUAL(255, color.B);
}

TEST(RSceneFactory, can_init_triangle) {
    Triangle *t = RSceneFactory::CreateTriangle(triangle0);
    Vect b = t->getB();
    DOUBLES_EQUAL(0.4, b.getX(), 0.00001);
    DOUBLES_EQUAL(0.5, b.getY(), 0.00001);
    DOUBLES_EQUAL(0.6, b.getZ(), 0.00001);

    delete t;
}

TEST(RSceneFactory, can_init_triangle_with_bindings) {
    Triangle *t = RSceneFactory::CreateTriangleWithBindings(triangle1);
    
    Vect norm = t->getA().getSurfaceNormal();
    CHECK_EQUAL(0, norm.getX());
    CHECK_EQUAL(1, norm.getY());
    CHECK_EQUAL(0, norm.getZ());

    norm = t->getB().getSurfaceNormal();
    CHECK_EQUAL(1, norm.getX());
    CHECK_EQUAL(0, norm.getY());
    CHECK_EQUAL(0, norm.getZ());

    delete t;
}

TEST(RSceneFactory, can_create_material) {
    Material *m = RSceneFactory::CreateMaterial(material);

    DOUBLES_EQUAL(1, m->getShininess(), 0.00001);
    DOUBLES_EQUAL(0.5f, m->getTransparency(), 0.00001);

    delete m;
}

TEST(RSceneFactory, can_init_mesh) {
    Material *ms = RSceneFactory::CreateMaterial(material);
    std::vector<Material *> v;
    v.push_back(ms);

    Mesh m;
    RSceneFactory::AddMaterials(&m, v);
    RSceneFactory::CreateMesh(m, trimesh, v);
    Material *mat = m.getMaterial();
    POINTERS_EQUAL(ms, mat);

    Triangle *t0 = m.getTriangle(0);
    Vect b = t0->getB();
    DOUBLES_EQUAL(0.4, b.getX(), 0.00001);
    DOUBLES_EQUAL(0.5, b.getY(), 0.00001);
    DOUBLES_EQUAL(0.6, b.getZ(), 0.00001);
    mat = t0->getMaterial();
    CHECK(ms == mat);
    DOUBLES_EQUAL(1, mat->getShininess(), 0.00001);
    DOUBLES_EQUAL(0.5f, mat->getTransparency(), 0.00001);

    Triangle *t1 = m.getTriangle(1);
    Vect a = t1->getA();
    DOUBLES_EQUAL(0.4, a.getX(), 0.00001);
    DOUBLES_EQUAL(0.5, a.getY(), 0.00001);
    DOUBLES_EQUAL(0.6, a.getZ(), 0.00001);
}

TEST(RSceneFactory, mesh_with_per_surface_material) {
    std::vector<Material *> v;
    v.push_back(RSceneFactory::CreateMaterial(material));
    material.diffColor[0] = 10;
    material.diffColor[1] = 0.1f;
    v.push_back(RSceneFactory::CreateMaterial(material));
    v.push_back(RSceneFactory::CreateMaterial(material));

    trimesh.materialBinding = PER_VERTEX_MATERIAL;
    trimesh.numPolys = 1;

    Mesh m;
    RSceneFactory::AddMaterials(&m, v);
    RSceneFactory::CreateMesh(m, trimesh, v);
    Triangle *t = m.getTriangle(0);
    Material *m0 = t->getMaterial('a');
    Material *m1 = t->getMaterial('b');
    Material *m2 = t->getMaterial('c');

    CHECK(m0 != m1);
    CHECK(m0 != m2);
    CHECK(m2 == m1);
    
    DOUBLES_EQUAL(1, m2->getDiffColor().R(), 0.00001);
    DOUBLES_EQUAL(0.1, m2->getDiffColor().G(), 0.00001);
    trimesh.materialBinding = PER_OBJECT_MATERIAL;
    trimesh.numPolys = 2;
}


TEST(RSceneFactory, can_init_light) {
    Light *l = RSceneFactory::CreateLight(light0);

    Vect d = l->getDir();
    CHECK_EQUAL(2, d.getX());
    CHECK_EQUAL(2, d.getY());
    CHECK_EQUAL(2, d.getZ());
    delete l;
}

TEST(RSceneFactory, can_init_lights) {
    std::vector<Light *> lts;
    RSceneFactory::CreateLights(lts, light0);

    CHECK_EQUAL(2, lts.size());

    Vect p = lts[1]->getPos();
    CHECK_EQUAL(0, p.getX());
    CHECK_EQUAL(1, p.getY());
    CHECK_EQUAL(2, p.getZ());
    delete lts[0];
    delete lts[1];
}

TEST(RSceneFactory, can_init_camera) {
    Camera cam;
    RSceneFactory::CreateCamera(cam, camera);

    Vect p = cam.getPos();
    CHECK_EQUAL(1, p.getX());
    DOUBLES_EQUAL(0.2, cam.getVerticalFOV(), 0.00001);
}

TEST(RSceneFactory, can_create_scene) {
    RScene s;
    RSceneFactory::CreateScene(s, scene);

    Light *l = s.getLight(0);
    CHECK_EQUAL(DIRECTIONAL_LIGHT, l->getType());
    l = s.getLight(1);
    CHECK_EQUAL(POINT_LIGHT, l->getType());
}
