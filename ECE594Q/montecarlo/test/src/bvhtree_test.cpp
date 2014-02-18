#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"

#include <vector>
#include "rayscene_shapes.h"
#include "rayscene_factory.h"
#include "bvhtree.h"


// std::vector<Shape *> shapes;
static Shape *s0;
static Shape *s1;
static Shape *s2;
static Shape *s3;
static std::vector<Shape *> v;

TEST_GROUP(BVHTreeTest) {
    void setup() {
        s0 = RaySceneFactory::NewSphere(1, Vect(0, 0, 0));
        s1 = RaySceneFactory::NewSphere(1, Vect(-1, 2, 1));
        s2 = RaySceneFactory::NewSphere(1, Vect(-2, -2, 2));
        s3 = RaySceneFactory::NewSphere(1, Vect(2, 2, -1));
    }
    void teardown() {
        delete s0;
        delete s1;
        delete s2;
        delete s3;
    }
};

TEST(BVHTreeTest, can_init_kdtree) {
    BVHTree *tree = new BVHTree();
    CHECK_FALSE(Shape::CompareX(s0, s1));
    delete tree;
}

TEST(BVHTreeTest, can_build_tree_with_four_shapes) {
    std::vector<Shape *> shapes;
    shapes.push_back(s0); shapes.push_back(s1);
    shapes.push_back(s2); shapes.push_back(s3);

    BVHTree tree;
    BVHNode *root = tree.buildTree(shapes);
    CHECK(root == tree.getRoot());

    BBox x = root->getBBox(); // Split on S0
    Vect pmin = x.getMin(); Vect pmax = x.getMax();
    CHECK_EQUAL(-3, pmin.getX());
    CHECK_EQUAL(-3, pmin.getY());
    CHECK_EQUAL(-2, pmin.getZ());
    CHECK_EQUAL(3, pmax.getX());
    CHECK_EQUAL(3, pmax.getY());
    CHECK_EQUAL(3, pmax.getZ());

    BVHNode *l = root->getLeft(); // S2 + S1
    x = l->getBBox();
    pmin = x.getMin(); pmax = x.getMax();
    CHECK_FALSE(l->isLeaf());
    CHECK_EQUAL(-3, pmin.getX());
    CHECK_EQUAL(-3, pmin.getY());
    CHECK_EQUAL(0, pmin.getZ());
    CHECK_EQUAL(0, pmax.getX());
    CHECK_EQUAL(3, pmax.getY());
    CHECK_EQUAL(3, pmax.getZ());

    
    BVHNode *r = root->getRight(); // S0 + S3
    x = r->getBBox();
    pmin = x.getMin(); pmax = x.getMax();
    CHECK_FALSE(r->isLeaf());
    CHECK_EQUAL(-1, pmin.getX());
    CHECK_EQUAL(-1, pmin.getY());
    CHECK_EQUAL(-2, pmin.getZ());
    CHECK_EQUAL(3, pmax.getX());
    CHECK_EQUAL(3, pmax.getY());
    CHECK_EQUAL(1, pmax.getZ());

    BVHNode *ll = l->getLeft(); BVHNode *lr = l->getRight();
    CHECK(ll->getShape() == s2);
    CHECK(lr->getShape() == s1);

    BVHNode *rl = r->getLeft(); BVHNode *rr = r->getRight();
    CHECK(rl->getShape() == s0);
    CHECK(rr->getShape() == s3);
}

TEST(BVHTreeTest, can_search_for_intersection_in_tree) {
    std::vector<Shape *> shapes;
    shapes.push_back(s0); shapes.push_back(s1);
    shapes.push_back(s2); shapes.push_back(s3);

    BVHTree tree;
    tree.buildTree(shapes);

    Ray r(Vect(2, 2, 2), Vect(0, 0, -1));
    Intersection ins = tree.intersects(r);
    CHECK(ins.hasIntersected());
    CHECK(ins.getShape() == s3);

    r = Ray(Vect(-1, -1, 1), Vect(-1, -1, 1));
    ins = tree.intersects(r);
    CHECK(ins.hasIntersected());
    CHECK(ins.getShape() == s2);

    r = Ray(Vect(-1, -1, 1), Vect(0, 0, 1));
    ins = tree.intersects(r);
    CHECK_FALSE(ins.hasIntersected());    
}
