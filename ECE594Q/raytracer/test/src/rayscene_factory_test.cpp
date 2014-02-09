#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"
#include "rayscene.h"
#include "rayscene_factory.h"

SphereIO sphere;
PolygonIO triangle0, triangle1;
PolySetIO trimesh;
ObjIO objects;
MaterialIO material;
LightIO light0;
LightIO light1;
CameraIO camera;
SceneIO scene;

TEST_GROUP(RaySceneFactory) {
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

        triangle1.numVertices = 3;
        triangle1.vert = new VertexIO[3];
        triangle1.vert[0].pos[0] = 0.4f;
        triangle1.vert[0].pos[1] = 0.5f;
        triangle1.vert[0].pos[2] = 0.6f;

        trimesh.numPolys = 2;
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
    }
    void teardown() {
        delete [] triangle0.vert;
        delete [] triangle1.vert;
        delete [] trimesh.poly;
    }
};

TEST(RaySceneFactory, can_init_sphere) {
    Sphere s;
    RaySceneFactory::CreateSphere(s, sphere);

    CHECK_EQUAL(1.33453f, s.getRadius());
    CHECK_EQUAL(1.33453f, s.getXlen());
    CHECK_EQUAL(1.33453f, s.getYlen());
    CHECK_EQUAL(1.33453f, s.getZlen());
    Vect x = s.getX();
    CHECK_EQUAL(1, x.getX());
    CHECK_EQUAL(0, x.getY());
    CHECK_EQUAL(0, x.getZ());
}

TEST(RaySceneFactory, can_convert_color_to_PX_Color) {
    Color c;
    c[0] = 0;
    c[1] = 0.5;
    c[2] = 1;
    PX_Color color = RaySceneFactory::ColorToPX_Color(c);
    CHECK_EQUAL(0, color.R);
    CHECK_EQUAL(127, color.G);
    CHECK_EQUAL(255, color.B);
}

TEST(RaySceneFactory, can_init_triangle) {
    Triangle t;
    RaySceneFactory::CreateTriangle(t, triangle0);
    Vect b = t.getB();
    DOUBLES_EQUAL(0.4, b.getX(), 0.00001);
    DOUBLES_EQUAL(0.5, b.getY(), 0.00001);
    DOUBLES_EQUAL(0.6, b.getZ(), 0.00001);
}

TEST(RaySceneFactory, can_init_mesh) {
    Mesh m;
    RaySceneFactory::CreateMesh(m, trimesh);
    Triangle *t0 = m.getTriangle(0);
    Vect b = t0->getB();
    DOUBLES_EQUAL(0.4, b.getX(), 0.00001);
    DOUBLES_EQUAL(0.5, b.getY(), 0.00001);
    DOUBLES_EQUAL(0.6, b.getZ(), 0.00001);

    Triangle *t1 = m.getTriangle(1);
    Vect a = t1->getA();
    DOUBLES_EQUAL(0.4, a.getX(), 0.00001);
    DOUBLES_EQUAL(0.5, a.getY(), 0.00001);
    DOUBLES_EQUAL(0.6, a.getZ(), 0.00001);
}

TEST(RaySceneFactory, can_init_light) {
    Light l;
    RaySceneFactory::CreateLight(l, light0);

    Vect d = l.getDir();
    CHECK_EQUAL(2, d.getX());
    CHECK_EQUAL(2, d.getY());
    CHECK_EQUAL(2, d.getZ());
}

TEST(RaySceneFactory, can_init_lights) {
    std::vector<Light> lts;
    RaySceneFactory::CreateLights(lts, light0);

    CHECK_EQUAL(2, lts.size());

    Vect p = lts[1].getPos();
    CHECK_EQUAL(0, p.getX());
    CHECK_EQUAL(1, p.getY());
    CHECK_EQUAL(2, p.getZ());
}

TEST(RaySceneFactory, can_init_camera) {
    Camera cam;
    RaySceneFactory::CreateCamera(cam, camera);

    Vect p = cam.getPos();
    CHECK_EQUAL(1, p.getX());
    DOUBLES_EQUAL(0.2, cam.getVerticalFOV(), 0.00001);
}

TEST(RaySceneFactory, can_create_scene) {
    RayScene s;
    RaySceneFactory::CreateScene(s, scene);

    Light l = s.getLight(0);
    CHECK_EQUAL(DIRECTIONAL_LIGHT, l.getType());
    l = s.getLight(1);
    CHECK_EQUAL(POINT_LIGHT, l.getType());
}
