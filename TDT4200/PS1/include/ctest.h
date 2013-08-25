#ifndef CTEST
#define CTEST
#include <malloc.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define CTEST_ERROR_BUF         100
#define CTEST_REPORT_BUF        1000000

/***************************************
* Colors applied to stdio
***************************************/
#define STD_COLOR               "\033[0;37m"
#define RED_COLOR               "\033[1;31m"
#define GREEN_COLOR             "\033[1;32m"
#define BLUE_COLOR              "\033[0;34m"
#define CYAN_COLOR              "\033[1;36m"

/***************************************
* TEST-RUNNER MACROS
***************************************/
/* Runs a testcase, given as a function reference */
#define TEST_CASE(testcase) do {\
    get_test_suite_name(__func__);\
    increase_num_tests();\
    testcase();\
    if (has_failed()) {\
        increase_num_tests_failed();\
        reset_testcase();\
    } else {\
        increase_num_tests_passed();\
    }\
} while (0)

/* Runs a testsuite, given as a function reference */
#define RUN_TEST_SUITE(testsuite) run_testsuite(testsuite)

/***************************************
* ASSERT MACROS
* Calls the appropriate function with information about testcase and line 
***************************************/
#define ASSERT_TRUE(test) do { assert_true((test), __func__, __LINE__); } while (0)
#define ASSERT_FALSE(test) do { assert_false(!(test), __func__, __LINE__); } while (0)

#define ASSERT_EQUAL_INT(actual, expected) do { assert_equal_int((actual == expected), actual, expected, __func__, __LINE__); } while (0)
#define ASSERT_NOT_EQUAL_INT(actual, expected) do { assert_not_equal_int((actual != expected), actual, __func__, __LINE__); } while (0)

#define ASSERT_EQUAL_DOUBLE(actual, expected, delta) do { assert_equal_double((actual - expected), actual, expected, delta, __func__, __LINE__); } while (0)
#define ASSERT_EQUAL_FLOAT(actual, expected, delta) do { assert_equal_float((actual - expected), actual, expected, delta, __func__, __LINE__); } while (0)

#define ASSERT_EQUAL_STRING(actual, expected) do { assert_equal_string((strcmp(actual, expected) == 0), actual, expected, __func__, __LINE__); } while (0)
#define ASSERT_NOT_EQUAL_STRING(actual, expected) do { assert_not_equal_string((strcmp(actual, expected) != 0), actual, __func__, __LINE__); } while (0)
#define ASSERT_CONTAINS_STRING(string, find) do { assert_contains_string((strstr(string, find) != NULL), string, find, __func__, __LINE__); } while (0)

/***************************************
* CTEST FUNCTIONS
***************************************/
/* Runs testsuite and prints summary */
int run_testsuite(void (*test_suite)(void));
void cleanup_testsuite();

/* Functions used to generate report summary */
void ctest_assertion_failed(const char *func, int line);
void ctest_assertion_passed(const char *func);
void generate_report(int passed, const char *message, const char *func, int line);

/* Assert functions, these should only be called through the macros */
void assert_true(int passed, const char *func, int line);
void assert_false(int passed, const char *func, int line);

void assert_equal_int(int passed, int actual, int expected, const char *func, int line);
void assert_not_equal_int(int passed, int actual, const char *func, int line);

void assert_equal_double(double passed, double actual, double expected, double delta, const char *func, int line);
void assert_equal_float(float passed, float actual, float expected, float delta, const char *func, int line);

void assert_equal_string(int passed, char *actual, char *expected, const char *func, int line);
void assert_not_equal_string(int passed, char *actual, const char *func, int line);
void assert_contains_string(int passed, char *string, char *find, const char *func, int line);

/* Helper functions used by the TESTCASE macro in order to generate a meaningful report */
void get_test_suite_name(const char *func);
void increase_num_tests();
int has_failed();
void increase_num_tests_failed();
void increase_num_tests_passed();
void reset_testcase();
#endif
