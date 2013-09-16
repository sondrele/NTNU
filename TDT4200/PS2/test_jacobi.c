#include <stdio.h>
#include <stdlib.h>
#include "ctest.h"
#include "global.h"

#define LP(row, col) ((row) + border) * (local_width + 2 * border) + ((col) + border)

int imageSize = 3;
int local_width = 3;
int local_height = 3;
int north = 0,
    south = 0,
    east = 0,
    west = 0;
int rank = 0;
int border = 1;
float *local_pres,
    *local_pres0,
    *local_diverg;

void setup() {
    local_pres = (float *) malloc(sizeof(float) * ((local_width + 2) * (local_height + 2)));
    local_pres0 = (float *) malloc(sizeof(float) * ((local_width + 2) * (local_height + 2)));
    for (int i = 0; i < ((local_width + 2) * (local_height + 2)); i++) {
        local_pres[i] = 1.0;
    }
    local_pres[6] = 2.1;
    local_pres[7] = 2.2;
    local_pres[8] = 2.3;
    local_pres[11] = 2.4;
    local_pres[12] = 3.0;
    local_pres[13] = 2.5;
    local_pres[16] = 2.6;
    local_pres[17] = 2.7;
    local_pres[18] = 2.8;
    
    local_diverg = (float *) malloc(sizeof(float));
    *local_diverg = 0.0f;
}

// void test_get_row() {
//     float *row = get_row(1);
    
//     ASSERT_EQUAL_FLOAT(row[0], 1.0, 0.001);
//     ASSERT_EQUAL_FLOAT(row[1], 2.4, 0.001);
//     ASSERT_EQUAL_FLOAT(row[2], 3.0, 0.001);
//     ASSERT_EQUAL_FLOAT(row[3], 2.5, 0.001);
//     ASSERT_EQUAL_FLOAT(row[4], 1.0, 0.001);
// }

// void test_get_col() {
//     float *col = get_col(1);

//     ASSERT_EQUAL_FLOAT(col[0], 1.0, 0.001);
//     ASSERT_EQUAL_FLOAT(col[1], 2.2, 0.001);
//     ASSERT_EQUAL_FLOAT(col[2], 3.0, 0.001);
//     ASSERT_EQUAL_FLOAT(col[3], 2.7, 0.001);
//     ASSERT_EQUAL_FLOAT(col[4], 1.0, 0.001);
// }

void test_indexing() {
    int i = LP(0, 0);
    ASSERT_EQUAL_INT(i, 6);
    i = LP(-1, -1);
    ASSERT_EQUAL_INT(i, 0);
    i = LP(3, 3);
    ASSERT_EQUAL_INT(i, 24);
}

void test_calculate_jacobi() {
    float ans = calculate_jacobi(1, 1),
        ans2 = calculate_jacobi(0, 0),
        ans3 = calculate_jacobi(-1, -1),
        ans4 = calculate_jacobi(3, 3);

    ASSERT_EQUAL_FLOAT(ans, 2.45, 0.001);
    ASSERT_EQUAL_FLOAT(ans2, 1.65, 0.001);
    ASSERT_EQUAL_FLOAT(ans3, 1.0, 0.001);
    ASSERT_EQUAL_FLOAT(ans4, 1.0, 0.001);
}

void test_jacobi_iteration() {
    jacobi_iteration();

    ASSERT_EQUAL_FLOAT(local_pres0[LP(0, 0)], 1.65, 0.001);
    ASSERT_EQUAL_FLOAT(local_pres0[LP(1, 1)], 2.45, 0.001);
    ASSERT_EQUAL_FLOAT(local_pres0[LP(1, 2)], 2.275, 0.00001);
}

void test_jacobi() {
    setup();
    TEST_CASE(test_indexing);
    TEST_CASE(test_calculate_jacobi);
    TEST_CASE(test_jacobi_iteration);
}

int main() {
    int a = RUN_TEST_SUITE(test_jacobi);
}
