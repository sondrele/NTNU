
#include "ctest.h"

void test_true() {
    ASSERT_TRUE(1);
}

void testsuite() {
    TEST_CASE(test_true);
}

int main() {
    int a = RUN_TEST_SUITE(testsuite);
}