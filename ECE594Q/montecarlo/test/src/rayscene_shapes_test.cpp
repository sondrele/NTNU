#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"
#include "rayscene_shapes.h"

Triangle *t;
Material *mat0;
Material *mat1;
Material *mat2;


TEST_GROUP(RaySceneShapesTest) {
    void setup() {
        mat0 = new Material(); mat0->setDiffColor(SColor(0, 0, 0));
        mat1 = new Material(); mat1->setDiffColor(SColor(0, 0, 0));
        mat2 = new Material(); mat2->setDiffColor(SColor(1, 1, 1));
        t = new Triangle();
        Vertex v0(-1, 0, 0); Vertex v1(1, 0, 0); Vertex v2(0, 0, -1);
        v0.setMaterial(mat0); v1.setMaterial(mat1); v2.setMaterial(mat2);
        v0.setSurfaceNormal(Vect(0, 1, 0));
        v1.setSurfaceNormal(Vect(0, 1, 0));
        v2.setSurfaceNormal(Vect(0, 0, -1));
        t->setA(v0); t->setB(v1); t->setC(v2);
    }
    void teardown() {
        delete t;
        delete mat0; delete mat1; delete mat2;
    }
};

TEST(RaySceneShapesTest, ray_can_intersect_with_triangle) {
    Triangle t0;
    t0.setA(Vertex(0, 0, 1));
    t0.setB(Vertex(-1, -1, 2));
    t0.setC(Vertex(1, -1, 2));

    Intersection is;
    Ray r0(Vect(0, 0, 0), Vect(0, -0.25, 1));
    is = t0.intersects(r0);
    CHECK(is.hasIntersected());

    Ray r1(Vect(0, 0, 0), Vect(0, -0.5, 2));
    is = t0.intersects(r1);
    CHECK(is.hasIntersected());
    DOUBLES_EQUAL(1.37437, is.getIntersectionPoint(), 0.00001);

    Ray r2(Vect(0, 0, 0), Vect(0, 0.1f, 2));
    is = t0.intersects(r2);
    CHECK_FALSE(is.hasIntersected());

    Ray r3(Vect(0, 0, 0), Vect(0.25, -0.1f, 2));
    is = t0.intersects(r3);
    CHECK_FALSE(is.hasIntersected());
}

TEST(RaySceneShapesTest, ray_can_intersect_with_mesh) {
    Material *m0 = new Material();
    Triangle *t0 = new Triangle();
    t0->addMaterial(m0);
    t0->setA(Vertex(0, 0, 1));
    t0->setB(Vertex(-1, -1, 2));
    t0->setC(Vertex(1, -1, 2));

    Material *m1 = new Material();
    Triangle *t1 = new Triangle();
    t1->addMaterial(m1);
    t1->setA(Vertex(0, 0, 1));
    t1->setB(Vertex(-1, 1, 2));
    t1->setC(Vertex(1, 1, 2));

    Mesh m;
    m.addTriangle(t0);
    m.addTriangle(t1);

    Ray r0(Vect(0, 0, 0), Vect(0, -0.25, 1));
    Intersection is = m.intersects(r0);
    CHECK(is.hasIntersected());

    Ray r1(Vect(0, 0, 0), Vect(1, 1, 2));
    is = m.intersects(r1);
    CHECK(is.hasIntersected());

    Ray r2(Vect(0, 0, 0), Vect(1, 1, -2));
    is = m.intersects(r2);
    CHECK_FALSE(is.hasIntersected());

}

TEST(RaySceneShapesTest, triangle_has_per_vertex_normals) {
    Vect n0 = t->getA().getSurfaceNormal();
    Vect n1 = t->getB().getSurfaceNormal();
    Vect n2 = t->getC().getSurfaceNormal();
    CHECK_EQUAL(1, n0.getY());
    CHECK_EQUAL(1, n1.getY());
    CHECK_EQUAL(-1, n2.getZ());
}

TEST(RaySceneShapesTest, triangle_has_area) {
    Triangle t0;
    t0.setA(Vertex(0, 0, -4));
    t0.setB(Vertex(2, 2, -4));
    t0.setC(Vertex(2, 0, -4));
    float area = t0.getArea();
    DOUBLES_EQUAL(2, area, 0.00001);
}

TEST(RaySceneShapesTest, can_get_interpolation_of_triangle_surface) {
    Ray r(Vect(0, 1, -1), Vect(0, -1, 0));
    Intersection in = t->intersects(r);
    CHECK(in.hasIntersected());

    t->setPerVertexNormal(true);
    Vect n = t->interpolatedNormal(Vect(0, 0, -1));
    CHECK_EQUAL(0, n.getX());
    CHECK_EQUAL(0, n.getY());
    CHECK_EQUAL(-1, n.getZ());

    n = t->interpolatedNormal(Vect(-1, 0, 0));
    CHECK_EQUAL(0, n.getX());
    CHECK_EQUAL(1, n.getY());
    CHECK_EQUAL(0, n.getZ());

    n = t->surfaceNormal(Vect(0, -1, 0), Vect(0, 0, -0.5f));
    CHECK_EQUAL(0, n.getX());
    DOUBLES_EQUAL(0.707101f, n.getY(), 0.00001);
    DOUBLES_EQUAL(-0.707101f, n.getZ(), 0.00001);
}

TEST(RaySceneShapesTest, can_get_interpolated_color) {
    t->setPerVertexMaterial(true);

    SColor color = t->getColor(Vect(0, 0, -1));
    CHECK_EQUAL(1, color.R());
    CHECK_EQUAL(1, color.G());
    CHECK_EQUAL(1, color.B());

    color = t->getColor(Vect(0, 0, -0.5f));
    CHECK_EQUAL(0.5f, color.R());
    CHECK_EQUAL(0.5f, color.G());
    CHECK_EQUAL(0.5f, color.B());
}

TEST(RaySceneShapesTest, can_get_long_and_lat_from_sphere) {
    Sphere s;
    s.setOrigin(Vect(0, -1, -1));
    s.setRadius(1);

    Point_2D pt = s.getUV(Vect(0, 0, -1));
    DOUBLES_EQUAL(0.5, pt.x, 0.00001);
    CHECK_EQUAL(1, pt.y);

    pt = s.getUV(Vect(0, -2, -1));
    DOUBLES_EQUAL(0.5, pt.x, 0.00001);
    DOUBLES_EQUAL(0, pt.y, 0.00001);
}

TEST(RaySceneShapesTest, can_compare_shapes) {
    Sphere s;
    s.setOrigin(Vect(0, -1, -1));
    s.setRadius(2);
    CHECK(s < *t);
    CHECK(Shape::CompareX(&s, t));
    
    s.setRadius(0.1f);
    CHECK(*t < s);
}

TEST(RaySceneShapesTest, ray_can_intersect_with_boundingBox) {
    BBox box;
    box.setMin(Vect(0, 0, 0));
    box.setMax(Vect(1, 1, 1));

    Ray r(Vect(0, 0, 2), Vect(0, 0, -1));
    CHECK(box.intersects(r));

    r = Ray(Vect(0, 0, 2), Vect(0, -1, -1));
    CHECK_FALSE(box.intersects(r));

    r = Ray(Vect(2, 2, 2), Vect(-1, -1, -1));
    CHECK(box.intersects(r));

    r = Ray(Vect(2, 0, 2), Vect(-2, 1, -2));
    CHECK(box.intersects(r));

    r = Ray(Vect(2, 0, 2), Vect(-2, -1, -2));
    CHECK_FALSE(box.intersects(r));
}

TEST(RaySceneShapesTest, mesh_boundingbox_updates_for_every_triangle) {
    Vertex a0(0, 0, 0); Vertex a1(1, 0, 0); Vertex a2(0, 0, -1);
    Vertex b0(0, 1, 0);
    Vertex c0(-2, -2, -3); Vertex c1(-2, 3, 1.5f); Vertex c2(3, 1.5f, -1);
    Triangle *t0 = new Triangle(); t0->setA(a0); t0->setB(a1); t0->setC(a2);
    Triangle *t1 = new Triangle(); t1->setA(a0); t1->setB(b0); t1->setC(a2);
    Triangle *t2 = new Triangle(); t2->setA(c0); t2->setB(c1); t2->setC(c2);
    Mesh m; m.addTriangle(t0);
    BBox box = m.getBBox();
    CHECK_EQUAL(0, box.getMin().getX());
    CHECK_EQUAL(0, box.getMin().getY());
    CHECK_EQUAL(-1, box.getMin().getZ());
    CHECK_EQUAL(1, box.getMax().getX());
    CHECK_EQUAL(0, box.getMax().getY());
    CHECK_EQUAL(0, box.getMax().getZ());

    m.addTriangle(t1);
    box = m.getBBox();
    CHECK_EQUAL(0, box.getMin().getX());
    CHECK_EQUAL(0, box.getMin().getY());
    CHECK_EQUAL(-1, box.getMin().getZ());
    CHECK_EQUAL(1, box.getMax().getX());
    CHECK_EQUAL(1, box.getMax().getY());
    CHECK_EQUAL(0, box.getMax().getZ());

    m.addTriangle(t2);
    box = m.getBBox();
    CHECK_EQUAL(-2, box.getMin().getX());
    CHECK_EQUAL(-2, box.getMin().getY());
    CHECK_EQUAL(-3, box.getMin().getZ());
    CHECK_EQUAL(3, box.getMax().getX());
    CHECK_EQUAL(3, box.getMax().getY());
    DOUBLES_EQUAL(1.5f, box.getMax().getZ(), 0.000001);

}

TEST(RaySceneShapesTest, can_get_bbox_centroid) {
    BBox x;
    x.setMin(Vect(-2, -2, -2));
    x.setMax(Vect(1, 1, 1));
    Vect centroid = x.getCentroid();
    CHECK_EQUAL(-0.5f, centroid.getX());
    CHECK_EQUAL(-0.5f, centroid.getY());
    CHECK_EQUAL(-0.5f, centroid.getZ());
}
