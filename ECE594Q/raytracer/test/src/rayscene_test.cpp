#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"
#include "rayscene.h"

TEST_GROUP(RaySceneTest) {
    void setup() {

    }
    void teardown() {

    }
};

TEST(RaySceneTest, can_add_shapes) {
    RayScene s;

    Shape *s0 = new Sphere();
    s.addShape(s0);

    Shape *s1 = s.getShape(0);
    CHECK(s0 == s1);

    delete s0;
}
