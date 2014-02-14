#include "CppUTest/CommandLineTestRunner.h"
#include "CppUTestExt/MockSupport.h"
#include "shader.h"

TEST_GROUP(ShadersTest) {
    void setup() {
    }
    void teardown() {
    }
};

TEST(ShadersTest, can_init_shader_with_check_pattern) {
    CShader *s = new CShader();
    delete s;
}
