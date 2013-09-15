#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "global.h"

// Indexing macro for local pres arrays
#define LP(row, col) ((row) + border) * (local_width + 2 * border) + ((col) + border)


// Distribute the diverg (bs) from rank 0 to all the processes
void distribute_diverg() {
    if (rank != 0) {
        MPI_Recv(local_diverg, (local_width * local_height), array_slice_t, 0, 1, cart_comm, &status);
        printf("%d: receiving divergence\n", rank);
        for (int i = 0; i < (local_width * local_height); i++) {
            printf("%f\n", local_diverg[i]);
        }
    } else {
        for (int node = 1; node < size; node++) {
            float *loc_div = (float *) malloc(sizeof(float) * local_width * local_height);
            for (int i = 0; i < (local_width * local_height); i++) {

            }
            MPI_Send(&diverg[node], 1, array_slice_t, node, 1, cart_comm);
            printf("distributing divergence to: %d\n", node);
        }
    }
}

// Gather the results of the computation at rank 0
void gather_pres() {
    printf("%d: gathering pressure\n", rank);
    if (rank != 0) {
        // MPI_Send()
    } else {
        for (int i = 0; i < size; i++) {

        }
    }
}

void print_array(float *array, int length) {
    for (int i = 0; i < length; i++)
        printf("%f", array[i]);
    printf("\n");
}

// Exchange borders between processes during computation
void exchange_borders() {
    printf("%d: exchanging borders\n", rank);
    if (north >= 0) {
        float *row = get_row(0),
            *in_row = (float *) malloc(sizeof(float) * local_width);
        MPI_Send(row, local_width, MPI_FLOAT, north, 1, cart_comm);
        MPI_Recv(&in_row, local_width, MPI_FLOAT, north, 1, cart_comm, &status);
        print_array(in_row, local_width);
    }

    if (south >= 0) {
        float *row = get_row(local_height - 1),
            *in_row = (float *) malloc(sizeof(float) * local_width);
        MPI_Send(row, local_width, MPI_FLOAT, south, 1, cart_comm);
        MPI_Recv(&in_row, local_width, MPI_FLOAT, south, 1, cart_comm, &status);
    }

    if (west >= 0) {
        float *col = get_col(0),
            *in_col = (float *) malloc(sizeof(float) * local_height);
        MPI_Send(col, local_height, MPI_FLOAT, west, 1, cart_comm);
        MPI_Recv(&in_col, local_height, MPI_FLOAT, west, 1, cart_comm, &status);
    }

    if (east >= 0) {
        float *col = get_col(local_width - 1),
            *in_col = (float *) malloc(sizeof(float) * local_height);
        MPI_Send(col, local_height, MPI_FLOAT, east, 1, cart_comm);
        MPI_Recv(&in_col, local_height, MPI_FLOAT, east, 1, cart_comm, &status);
    }
}

float *get_row(int x) {
    float *row = (float *) malloc(sizeof(float) * (local_width + 2));
    for (int i = -1; i < local_width + 1; i++) {
        row[i+1] = local_pres[LP(x, i)];
    }
    return row;
}

float *get_col(int x) {
    float *col = (float *) malloc(sizeof(float) * (local_height + 2));
    for (int i = -1; i < local_height + 1; i++) {
        col[i+1] = local_pres[LP(i, x)];
    }
    return col;
}

// One jacobi iteration
void jacobi_iteration() {
    for (int row = 0; row < local_height; row++) {
        for (int col = 0; col < local_width; col++) {
            local_pres0[LP(row, col)] = calculate_jacobi(row, col);
        }
    }
}

float calculate_jacobi(int row, int col) {
    int r2 = (row == -1) ? row : row - 1,
        r4 = (row == local_height) ? row : row + 1,
        c1 = (col == -1) ? col : col - 1,
        c3 = (col == local_width) ? col : col + 1;

    float x1 = local_pres[LP(row, c1)],
        x2 = local_pres[LP(r2, col)],
        x3 = local_pres[LP(row, c3)],
        x4 = local_pres[LP(r4, col)];
    return 0.25 * (x1 + x2 + x3 + x4 - *local_diverg);
}

void print_jacobi(float *jacobi) {
    for(int i = -1; i < local_height + 1; i++) {
        for (int j = -1; j < local_width + 1; j++) {
            printf("%f ", jacobi[LP(i, j)]);
        }
        printf("\n");
    }
}

// Solve linear system with jacobi method
void jacobi(int iter) {
    distribute_diverg();

    // Jacobi iterations
    for (int k = 0; k < iter; k++) {    
        jacobi_iteration();
    }

    gather_pres();
}
