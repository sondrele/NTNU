#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <string.h>

/***************************************
* Colors applied to stdio
***************************************/
#define red   "\033[1;31m"
#define green "\033[1;32m"
#define blue  "\033[0;34m"
#define cyan  "\033[1;36m"

/***************************************
* ASSERT MACROS
***************************************/

/* 
Error message, printing the failing function and the what line 
the assertion failed at (including some nice colors :)
*/
#define FAIL() printf("\n%sfailure in %s%s() %1$sline %2$s%d\n%1$s", red, cyan, __func__, __LINE__)

#define ERROR_MSG(actual, expected, type) {\
    switch (type) {\
    case CHAR:\
        printf("%3$sExpected %4$s%2$s%3$s, but was %4$s%1$s%3$s\n", actual, expected, red, cyan);\
        return;\
    case INT:\
        printf("%3$sExpected %4$s%2$d%3$s, but was %4$s%1$d%3$s\n", actual, expected, red, cyan);\
        return;\
    }\
} while (0)

#define ASSERT_TRUE(test) do {\
    if (!(test)) {\
        FAIL();\
        return 1;\
    }\
} while (0)

#define ASSERT_FALSE(test) do {\
    if ((test)) {\
        FAIL();\
        return 1;\
    }\
} while (0)

#define ASSERT_EQUAL_STRING(actual, expected) do {\
    int test = (strcmp(actual, expected) == 0);\
    if (!(test)) {\
        FAIL();\
        ERROR_MSG(actual, expected, CHAR);\
        return 1;\
    }\
} while (0)

#define ASSERT_NOT_EQUAL_INT(actual, not_expected) do {\
    int test = (actual != not_expected);\
    if (!(test)) {\
        FAIL();\
        error_msg(actual, not_expected, INT);\
        return 1;\
    }\
} while (0)

#define ASSERT_EQUAL_INT(actual, expected) do {\
    int test = (actual == expected);\
    if (!(test)) {\
        FAIL();\
        error_msg(actual, expected, INT);\
        return 1;\
    }\
} while (0)

#define ASSERT_NOT_EQUAL(actual, not_expected, type) do {\
    int test = (actual != not_expected);\
    if (type == CHAR) {\
        test = (strcmp(actual, not_expected) == 0);\
    }\
    if (!(test)) {\
        FAIL();\
        error_msg(actual, not_expected, type);\
        return 1;\
    }\
} while (0)

#define _verify(test) do {\
    int r=test();\
    tests_run++;\
    if (r)\
        return r;\
} while (0)

// switch (type) {
// case CHAR:
// printf("%3$sExpected %4$s%2$s%3$s, but was %4$s%1$s%3$s\n", (char*)actual, (char*)expected, red, cyan);
// return;
// case INT:
// printf("%3$sExpected %4$s%2$d%3$s, but was %4$s%1$d%3$s\n", (int)actual, (int)expected, red, cyan);
// return;
// }

void error_msg(void *actual, void *expected, int type);

int tests_run = 0;
void print_summary(int);

enum Types { CHAR, INT, DOUBLE };

/* Stub functions */
int stdout_fd;

void stub_printf();

char *read_printf();

void restore_printf();
