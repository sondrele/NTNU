#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"

#include <vector>
#include "rayscene_shapes.h"
#include "rayscene_factory.h"
#include "kdtree.h"


// std::vector<Shape *> shapes;
static Shape *s0;
static Shape *s1;
static Shape *s2;
static Shape *s3;
static std::vector<Shape *> v;

TEST_GROUP(KDTreeTest) {
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

TEST(KDTreeTest, can_init_kdtree) {
    KDTree *tree = new KDTree();
    CHECK_FALSE(Shape::CompareX(s0, s1));
    delete tree;
}

TEST(KDTreeTest, can_build_tree_with_four_shapes) {
    std::vector<Shape *> shapes;
    shapes.push_back(s0); shapes.push_back(s1);
    shapes.push_back(s2); shapes.push_back(s3);

    KDTree tree;
    tree.setShapesPerLeaf(1);
    Node *root = tree.buildTree(shapes);
    CHECK(root == tree.getRoot());

    Vect x = root->getPoint(); // Split on S0
    CHECK_EQUAL(-1, x.getX());
    CHECK_EQUAL(-1, x.getY());
    CHECK_EQUAL(-1, x.getZ());

    Node *l = root->getLeft();
    CHECK_FALSE(l->isLeaf());
    CHECK_EQUAL(-2, l->getPoint().getX());
    CHECK_EQUAL(1, l->getPoint().getY());
    CHECK_EQUAL(0, l->getPoint().getZ());
    
    Node *ll = l->getLeft(); Node *lr = l->getRight();
    CHECK(ll->getShape(0) == s2);
    CHECK(lr->getShape(0) == s1);

    Node *r = root->getRight();
    CHECK_FALSE(r->isLeaf());
    CHECK_EQUAL(1, r->getPoint().getX());
    CHECK_EQUAL(1, r->getPoint().getY());
    CHECK_EQUAL(-2, r->getPoint().getZ());
    
    Node *rl = r->getLeft(); Node *rr = r->getRight();
    CHECK(rl->getShape(0) == s0);
    CHECK(rr->getShape(0) == s3);

}
