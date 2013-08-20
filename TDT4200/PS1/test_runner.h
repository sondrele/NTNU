
/* Colors that can be applied to stdio*/
#define red   "\033[1;31m"
#define green "\033[1;32m"
#define blue  "\033[0;34m"
#define cyan  "\033[1;36m"


#define FAIL() printf("\n%sfailure in %s%s() %1$sline %2$s%d\n%1$s", red, cyan, __func__, __LINE__)

#define _assert(test) do {\
    if (!(test)) {\
        FAIL();\
        return 1;\
    }\
} while (0)

#define _assert_equals(actual, expected, type) do {\
	int test = (actual == expected);\
	if (type == CHAR) {\
		test = (strcmp(actual, expected) == 0);\
	}\
    _assert(test);\
} while (0)

#define _verify(test) do {\
    int r=test();\
    tests_run++;\
    if (r)\
        return r;\
} while (0)

void error_msg(void *actual, void *expected, int type);

int tests_run = 0;
void print_summary(int);

enum Types { CHAR, INT, DOUBLE };

/* Stub functions */
int stdout_fd;

void stub_printf();

char *read_printf();

void restore_printf();
