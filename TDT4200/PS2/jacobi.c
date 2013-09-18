#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "global.h"

// Create unique tags for each message
#define TAG_DISTR_DIVERG            1000
#define TAG_DISTR_PRES              1000
#define TAG_EXCHANGE                2000
#define TAG_GATHER                  3000

// Indexing macro for local pres arrays
#define LP(row, col) ((row) + border) * (local_width + 2 * border) + ((col) + border)
#define IX(i,j) ((i)+(imageSize+2)*(j))

static int *displs;
static int *counts;

void print_statss() {
    printf("imageSize = %d\n", imageSize);
    printf("local_height = %d\n", local_height);
    printf("local_width = %d\n", local_width);
    printf("size = %d\n", size);

    double sum = 0;
    for (int j=1; j<=imageSize; j++) {
        for (int i=1; i<=imageSize; i++) {
            sum += pres[IX(i,j)];
        }
    }
    printf("------sum pres: %f\n", sum);
    sum = 0;
    for (int j=1; j<=imageSize; j++) {
        for (int i=1; i<=imageSize; i++) {
            sum += diverg[IX(i,j)];
        }
    }
    printf("------sum div: %f\n", sum);
    double sum1 = 0,
        sum2 = 0,
        sum3 = 0;
    for (int j=-1; j <= local_height; j++) {
        for (int i=-1; i<= local_width ; i++) {
            sum1 += local_pres[IX(i,j)];
            sum2 += local_pres0[IX(i,j)];
            if (j >= 0 && j < local_height && i >= 0 && i < local_width)
                sum3 += local_diverg[IX(i,j)];
        }
    }
    printf("------sum local_pres: %f\n", sum1);
    printf("------sum local_pres0: %f\n", sum2);
    printf("------sum local_diverg: %f\n", sum3);
}

void print_array(float *a, int s, int s2) {
    for (int i = 0; i < s; ++i) {
        for (int j = 0; j < s2; ++j) {
            printf("%f ", a[i*s2+j]);
        }
        printf("\n");
    }
}

// Distribute the diverg (bs) from rank 0 to all the processes
void distribute_diverg() {
    MPI_Scatterv(diverg+imageSize+3,
                counts,
                displs,
                matrix_t,
                local_diverg,
                1,
                local_diverg_t,
                0,
                cart_comm);
}

// Gather the results of the computation at rank 0
void gather_pres() {
    MPI_Gatherv(local_pres0 + local_width + 3,
                1,
                local_pres_t,
                pres + imageSize + 3,
                counts,
                displs,
                matrix_t,
                0,
                cart_comm);
}

// Exchange borders between processes during computation
void exchange_borders() {
    if (north >= 0) {
        float *out = (local_pres + LP(0, 0)),
            *in = (local_pres + LP(-1, 0));
        int tag = TAG_EXCHANGE + north + rank;
        MPI_Send(out, 1, border_row_t, north, tag, cart_comm);
        MPI_Recv(in, 1, border_row_t, north, tag, cart_comm, &status);
    }
    if (south >= 0) {
        float *out = (local_pres + LP(local_height - 1, 0)),
            *in = (local_pres + LP(local_height, 0));
        int tag = TAG_EXCHANGE + south + rank;
        MPI_Send(out, 1, border_row_t, south, tag, cart_comm);
        MPI_Recv(in, 1, border_row_t, south, tag, cart_comm, &status);
    }
    if (west >= 0) {
        float *out = (local_pres + LP(0, 0)),
            *in = (local_pres + LP(0, -1));
        int tag = TAG_EXCHANGE + west + rank;
        MPI_Send(out, 1, border_col_t, west, tag, cart_comm);
        MPI_Recv(in, 1, border_col_t, west, tag, cart_comm, &status);
    }
    if (east >= 0) {
        float *out = (local_pres + LP(0, local_width - 1)),
            *in = (local_pres + LP(0, local_width));
        int tag = TAG_EXCHANGE + east + rank;
        MPI_Send(out, 1, border_col_t, east, tag, cart_comm);
        MPI_Recv(in, 1, border_col_t, east, tag, cart_comm, &status);
    }
}

// Calculate the value for one element in the grid
float calculate_jacobi(int row, int col) {
    float x1 = local_pres0[LP(row, col - 1)],
        x2 = local_pres0[LP(row - 1, col)],
        x3 = local_pres0[LP(row, col + 1)],
        x4 = local_pres0[LP(row + 1, col)];
    if (north < 0 && row == 0) {
        x2 = local_pres0[LP(row, col)];
    }
    if (south < 0 && row == local_height) {
        x4 = local_pres0[LP(row, col)];
    }
    if (west < 0 && col == 0) {
        x1 = local_pres0[LP(row, col)];
    }
    if (east < 0 && col == local_width) {
        x3 = local_pres0[LP(row, col)];
    }
    return 0.25 * (x1 + x2 + x3 + x4 - local_diverg[row * local_width + col]);
}

// One jacobi iteration
void jacobi_iteration() {
    for (int row = 0; row < local_height; row++) {
        for (int col = 0; col < local_width; col++) {
            local_pres[LP(row, col)] = calculate_jacobi(row, col);
            // local_pres[LP(row, col)] = 0.2 * rank;
        }
    }
}

void init_local_pres() {
    displs = malloc(sizeof(int) * size);
    counts = malloc(sizeof(int) * size);

    for (int i = 0; i < dims[0]; i++) {
        // int row_addr = i * local_height * local_width * dims[1] + imageSize + 3;
        for (int j = 0; j < dims[1]; j++) {
            // int col_addr = j * local_width + i*2;
            displs[i * dims[1] + j] = i*local_height*(imageSize+2)+j*local_width;
            counts[i * dims[1] + j] = 1;
        }
    }

    for (int i = -1; i < local_height + 1; i++) {
        for (int j = -1; j < local_width + 1; j++) {
            int idx = LP(i, j);
            local_pres0[idx] = 0.0;
            local_pres[idx] = 0.0;
        }
    }
}

// Solve linear system with jacobi method
void jacobi(int iter) {
    init_local_pres();
    distribute_diverg();

    // // Jacobi iterations
    for (int k = 0; k < iter; k++) {    
        jacobi_iteration();
        exchange_borders();
        float *temp_ptr = local_pres0;
        local_pres0 = local_pres;
        local_pres = temp_ptr;
    }

    gather_pres();
}

// For debugging purposes
void print_local_pres(float *jacobi) {
    for(int i = -1; i < local_height + 1; i++) {
        for (int j = -1; j < local_width + 1; j++) {
            printf("%f ", jacobi[LP(i, j)]);
        }
        printf("\n");
    }
}
