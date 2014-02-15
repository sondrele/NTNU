#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"

#include <vector>
#include "rayscene_shapes.h"
#include "rayscene_factory.h"
#include "kdtree.h"


// std::vector<Shape *> shapes;
Shape *s;
TEST_GROUP(KDTreeTest) {
    void setup() {
        // shapes.push_back();
        s = RaySceneFactory::NewSphere(1, Vect(-1, 1, -1));
        // shapes.push_back(RaySceneFactory::NewSphere(1, Vect(-1, -1, -1)));
        // shapes.push_back(RaySceneFactory::NewSphere(1, Vect(1, -1, -1)));
        // shapes.push_back(RaySceneFactory::NewSphere(1, Vect(1, 1, -1)));
        // shapes.push_back(RaySceneFactory::NewSphere(1, Vect(-1, 1, 1)));
        // shapes.push_back(RaySceneFactory::NewSphere(1, Vect(-1, -1, 1)));
        // shapes.push_back(RaySceneFactory::NewSphere(1, Vect(1, -1, 1)));
        // shapes.push_back(RaySceneFactory::NewSphere(1, Vect(1, 1, 1)));
    }
    void teardown() {
        // for (uint i = 0; i < shapes.size(); i++) {
        //     delete shapes[i];
        // }
        delete s;
    }
};

TEST(KDTreeTest, can_init_kdtree) {
    KDTree *tree = new KDTree(3);
    delete tree;
}
