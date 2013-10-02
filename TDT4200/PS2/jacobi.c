#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "global.h"

// Rank of the root node
#define ROOT                        0

// Indexing macro for local pres arrays
#define LP(row, col) ((row) + border) * (local_width + 2 * border) + ((col) + border)

// Arrays used by MPI_scatterv and MPI_gatherv for for each of the processes
static int *displs;
static int *counts;

// Distribute the diverg (bs) from root to all the processes
void distribute_diverg() {
    MPI_Scatterv(diverg + imageSize + 3,            // sendbuf
                counts,                             // sendcnts
                displs,                             // displs
                pres_and_diverg_t,                  // sendtype
                local_diverg,                       // recvbuf
                1,                                  // recvcnt
                local_diverg_t,                     // recvtype
                ROOT,                               // root
                cart_comm);                         // comm
}

// Gather the results of the computation at the root
void gather_pres() {
    MPI_Gatherv(local_pres0 + local_width + 3,      // sendbuf
                1,                                  // sendcount
                local_pres_t,                       // sendtype
                pres + imageSize + 3,               // recvebuf
                counts,                             // recvcounts
                displs,                             // displs
                pres_and_diverg_t,                  // recvtype
                ROOT,                               // root
                cart_comm);                         // comm
}

// Called from 'exchange_borders', handles the MPI procedure calls
void exchange_border(float *out, float *in, int dst, MPI_Datatype type) {
    int tag = dst + rank;
    MPI_Send(out, 1, type, dst, tag, cart_comm);
    MPI_Recv(in, 1, type, dst, tag, cart_comm, &status);
}

// Exchange borders between processes during computation
void exchange_borders() {
    float *lp = local_pres;
    if (north >= 0) {
        exchange_border(lp + LP(0, 0), lp + LP(-1, 0), north, border_row_t);
    }
    if (south >= 0) {
        exchange_border(lp + LP(local_height - 1, 0), lp + LP(local_height, 0), south, border_row_t);
    }
    if (west >= 0) {
        exchange_border(lp + LP(0, 0), lp + LP(0, -1), west, border_col_t);
    }
    if (east >= 0) {
        exchange_border(lp + LP(0, local_width - 1), lp + LP(0, local_width), east, border_col_t);
    }
}

// Calculate the value for one element in the grid
float calculate_jacobi(int row, int col) {
    float x1 = local_pres0[LP(row, col - 1)],
        x2 = local_pres0[LP(row - 1, col)],
        x3 = local_pres0[LP(row, col + 1)],
        x4 = local_pres0[LP(row + 1, col)];

    // If the element is on the border of 'pres', set to it's current value instead
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
    for (int i = 0; i < local_height; i++) {
        for (int j = 0; j < local_width; j++) {
            local_pres[LP(i, j)] = calculate_jacobi(i, j);
        }
    }
}

// Calculates the displacements used in'gather_pres' and 'distribute_diverg' 
// for each of the processes,
// and initializes 'local_pres' and 'local_pres0' to empty arrays
void init_local_pres() {
    displs = malloc(sizeof(int) * size);
    counts = malloc(sizeof(int) * size);

    for (int i = 0; i < dims[0]; i++) {
        for (int j = 0; j < dims[1]; j++) {
            int index = i * dims[1] + j;
            displs[index] = i * local_height * (imageSize + 2) + j * local_width;
            counts[index] = 1;
        }
    }

    for (int i = -1; i < local_height + 1; i++) {
        for (int j = -1; j < local_width + 1; j++) {
            int index = LP(i, j);
            local_pres0[index] = 0.0;
            local_pres[index] = 0.0;
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
        // Rearrange the pointers after each iteration
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
