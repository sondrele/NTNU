#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "global.h"

// Indexing macro for local pres arrays
#define LP(row, col) ((row) + border) * (local_width + 2 * border) + ((col) + border)


// Distribute the diverg (bs) from rank 0 to all the processes
void distribute_diverg() {
    if (rank != 0) 
        return;

}

// Gather the results of the computation at rank 0
void gather_pres() {
    if (rank != 0) 
        return;

}

// Exchange borders between processes during computation
void exchange_borders() {
    // if (north >= 0) {
    //     float *row = get_row(0),
    //         *in_row = (float *) malloc(sizeof(float) * local_width);
    //     MPI_Send(row, local_width, MPI_FLOAT, north, 1, cart_comm);
    //     MPI_Recv(&in_row, local_width, MPI_FLOAT, north, 1, cart_comm, &status);
    // }

    // if (south >= 0) {
    //     float *row = get_row(local_height - 1),
    //         *in_row = (float *) malloc(sizeof(float) * local_width);
    //     MPI_Send(row, local_width, MPI_FLOAT, south, 1, cart_comm);
    //     MPI_Recv(&in_row, local_width, MPI_FLOAT, south, 1, cart_comm, &status);
    // }

    // if (west >= 0) {
    //     float *col = get_col(0),
    //         *in_col = (float *) malloc(sizeof(float) * local_height);
    //     MPI_Send(col, local_height, MPI_FLOAT, west, 1, cart_comm);
    //     MPI_Recv(&in_col, local_height, MPI_FLOAT, west, 1, cart_comm, &status);
    // }

    // if (east >= 0) {
    //     float *col = get_col(local_width - 1),
    //         *in_col = (float *) malloc(sizeof(float) * local_height);
    //     MPI_Send(col, local_height, MPI_FLOAT, east, 1, cart_comm);
    //     MPI_Recv(&in_col, local_height, MPI_FLOAT, east, 1, cart_comm, &status);
    // }
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
    // TODO: hvis (row,col) er på edge, så må noe gjøres, ikke hvis (row-1,col) er på edge..
    int i1 = on_edge(row - 1, col) ? LP(row, col) : LP(row - 1, col),
        i2 = on_edge(row + 1, col) ? LP(row, col) : LP(row + 1, col),
        i3 = on_edge(row, col - 1) ? LP(row, col) : LP(row, col - 1),
        i4 = on_edge(row, col + 1) ? LP(row, col) : LP(row, col + 1);
    float x1 = local_pres[i1],
        x2 = local_pres[i2],
        x3 = local_pres[i3],
        x4 = local_pres[i4];
    return 0.25 * (x1 + x2 + x3 + x4 - *local_diverg);
}

int on_edge(int row, int col) {
    if (row == -1 || col == -1 || row == (local_height + 1) || col == (local_width + 1))
        return 1;
    return 0;
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
