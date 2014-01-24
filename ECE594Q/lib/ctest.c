#include "ctest.h"

static int num_tests = 0;
static int num_tests_failed = 0;
static int num_tests_passed = 0;
static int num_assertions_passed = 0;
static int num_assertions_failed = 0;
static int testcase_failed = 0;

static char test_suite_name[CTEST_ERROR_BUF];
static char error_message[CTEST_ERROR_BUF];
static char test_report[CTEST_REPORT_BUF];

void ctest_assertion_failed(const char *func, int line) {
    sprintf(test_report + strlen(test_report), "# ctest: %1$sFailure in %2$s%4$s() %1$sline%2$s %5$d %3$s\n", RED_COLOR, CYAN_COLOR, STD_COLOR, func, line);
    num_assertions_failed += 1;
}

void ctest_assertion_passed(const char *func) {
    num_assertions_passed += 1;
}

void ctest_generate_report(int passed, const char *error_message, const char *func, int line) {
    if (passed) {
        ctest_assertion_passed(func);
    } else {
        testcase_failed = 1;
        ctest_assertion_failed(func, line);
        sprintf(test_report + strlen(test_report), "# ctest: %s %s\n#\n", error_message, STD_COLOR);
    }
}

/* ASSERTIONS */
void assert_same(int passed, void *ptr_1, void *ptr_2, const char *func, int line) {
    int diff = (long)ptr_2 - (long)ptr_1;
    diff = diff >= 0 ? diff : -diff;
    sprintf(error_message, "%4$sExpected %5$s%1$p%4$s to be equal to %5$s%2$p%4$s, they differ by %5$s%3$d%4$s bytes", ptr_1, ptr_2, diff, RED_COLOR, CYAN_COLOR);
    ctest_generate_report(passed, error_message, func, line);
}

void assert_true(int passed, const char *func, int line) {
    sprintf(error_message, "%sExpected %strue", RED_COLOR, CYAN_COLOR);
    ctest_generate_report(passed, error_message, func, line);
}

void assert_false(int passed, const char *func, int line) {
    sprintf(error_message, "%sExpected %sfalse", RED_COLOR, CYAN_COLOR);
    ctest_generate_report(passed, error_message, func, line);
}

/* INTEGER ASSERTIONS */
void assert_equal_int(int passed, int actual, int expected, const char *func, int line) {
    sprintf(error_message, "%3$sExpected %4$s%2$d%3$s, but was %4$s%1$d%3$s", actual, expected, RED_COLOR, CYAN_COLOR);
    ctest_generate_report(passed, error_message, func, line);
}

void assert_not_equal_int(int passed, int actual, const char *func, int line) {
    sprintf(error_message, "%1$sExpected integers to %2$snot%1$s be equal: %2$s%3$d", RED_COLOR, CYAN_COLOR, actual);
    ctest_generate_report(passed, error_message, func, line);
}

/* FLOAT ASSERTIONS */
void assert_equal_double(double passed, double actual, double expected, double delta, const char *func, int line) {
    passed = (passed < 0.0) ? -passed : passed;
    delta = (delta < 0.0) ? -delta : delta;
    sprintf(error_message, "%3$sExpected %4$s%2$f%3$s, but was %4$s%1$f%3$s", actual, expected, RED_COLOR, CYAN_COLOR);
    ctest_generate_report(passed <= delta, error_message, func, line);
}

void assert_equal_float(float passed, float actual, float expected, float delta, const char *func, int line) {
    passed = (passed < 0.0) ? -passed : passed;
    delta = (delta < 0.0) ? -delta : delta;
    sprintf(error_message, "%3$sExpected %4$s%2$f%3$s, but was %4$s%1$f%3$s", actual, expected, RED_COLOR, CYAN_COLOR);
    ctest_generate_report(passed <= delta, error_message, func, line);
}

/* STRING ASSERTIONS */
void assert_equal_string(int passed, char *actual, char *expected, const char *func, int line) {
    sprintf(error_message, "%3$sExpected %4$s%2$s%3$s, but was %4$s%1$s%3$s", actual, expected, RED_COLOR, CYAN_COLOR);
    ctest_generate_report(passed, error_message, func, line);
}

void assert_not_equal_string(int passed, char *actual, const char *func, int line) {
    sprintf(error_message, "%1$sExpected strings to %2$snot%1$s be equal: %2$s%3$s", RED_COLOR, CYAN_COLOR, actual);
    ctest_generate_report(passed, error_message, func, line);   
}

void assert_contains_string(int passed, char *string, char *find, const char *func, int line) {
    sprintf(error_message, "%3$sDid not find occurence of %4$s%2$s%3$s in string %4$s%1$s%3$s", string, find, RED_COLOR, CYAN_COLOR);
    ctest_generate_report(passed, error_message, func, line);
}

/* HELPER FUNCTIONS */
void get_test_suite_name(const char *func) {
    sprintf(test_suite_name, "%s", func);
}

void increase_num_tests() {
    num_tests += 1;
}

int has_failed() {
    return testcase_failed;
}

void increase_num_tests_failed() {
    num_tests_failed += 1;
}

void increase_num_tests_passed() {
    num_tests_passed += 1;
}

void reset_testcase() {
    testcase_failed = 0;
}

/* TESTSUITE RUNNER */
int run_testsuite(void (*test_suite)(void)) {
    clock_t start, end;
    start = clock();
    test_suite();
    end = clock();
    float diff = (((float)end - (float)start) / CLOCKS_PER_SEC) * 1000;

    printf("==========================================================================\n");
    printf("# ctest: %sRUNNING %s%s\n", FAT_COLOR, test_suite_name, STD_COLOR);
    printf("--------------------------------------------------------------------------\n");
    printf("%s", test_report);
    printf("--------------------------------------------------------------------------\n");
    printf("# ctest: %s%d tests finished in: %fms%s\n", FAT_COLOR, num_tests, diff, STD_COLOR);
    if (num_tests_passed) {
        printf("# ctest: %sPASSED TESTS (%d)%s", GREEN_COLOR, num_tests_passed, STD_COLOR);
        printf(", %sPASSED ASSERTIONS (%d)%s\n", GREEN_COLOR, num_assertions_passed, STD_COLOR);
    }
    if (num_assertions_failed) {
        printf("# ctest: %sFAILED TESTS (%d)%s", RED_COLOR, num_tests_failed, STD_COLOR);
        printf(", %sFAILED ASSERTIONS (%d)%s", RED_COLOR, num_assertions_failed, STD_COLOR);
        printf(", %sPASSED ASSERTIONS (%d)%s\n", RED_COLOR, num_assertions_passed, STD_COLOR);
        printf("# ctest: %1$sTestsuite %2$s%4$s%1$s failed!%3$s\n", RED_COLOR, CYAN_COLOR, STD_COLOR, test_suite_name);
    } else {
        printf("# ctest: %1$sTestsuite %2$s%4$s%1$s passed!%3$s\n", GREEN_COLOR, CYAN_COLOR, STD_COLOR, test_suite_name);
    }
    printf("==========================================================================\n");

    cleanup_testsuite();
    return num_tests_failed ? 1 : 0;
}

void cleanup_testsuite() {
    testcase_failed = 0;
    num_tests = 0;
    num_tests_failed = 0;
    num_assertions_failed = 0;
    num_tests_passed = 0;
    num_assertions_passed = 0;
    sprintf(test_suite_name, "\n");
    sprintf(test_report, "\n");
}
