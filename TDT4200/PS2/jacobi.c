#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "global.h"

// Indexing macro for local pres arrays
#define LP(row, col) ((row) + border) * (local_width + 2 * border) + ((col) + border)


// Distribute the diverg (bs) from rank 0 to all the processes
void distribute_diverg() {
    if (rank == 0) {
        int displs[size];
        for (int i = 0; i < size / local_height; i++) {
            for (int j = 0; j < local_height; j++) {
                displs[i * local_height + j] = i * imageSize * local_height + j * local_width;
            }
        }
        for (int i = 1; i < size; i++) {
            int offset = displs[i];
            for (int j = 0; j < local_height; j++) {
                offset += j * imageSize;
                MPI_Send((diverg + offset), 1, array_slice_t, i, 123, cart_comm);
            }
        }
        for (int i = 0; i < local_height; i++) {
            for(int j = 0; j < local_width; j++) {
                local_diverg[i * local_width + j] = diverg[i * imageSize + j];
            }
        }
    } else {
        for (int i = 0; i < local_height; i++) {
            MPI_Recv((local_diverg + local_width * i), 1, array_slice_t, 0, 123, cart_comm, &status);
        }
    }
}

// Gather the results of the computation at rank 0
void gather_pres() {
    if (rank != 0) {
        for (int j = 0; j < local_height; j++) {
            MPI_Send((local_pres + LP(j, 0)), 1, array_slice_t, 0, 123, cart_comm);
        }
    } else if (rank == 0) {
        int displs[size];
        for (int i = 0; i < size / local_height; i++) {
            for (int j = 0; j < local_height; j++) {
                displs[i * local_height + j] = i * imageSize * local_height + j * local_width;
            }
        }
        for (int i = 1; i < size; i++) {
            int offset = displs[i];
            for (int j = 0; j < local_height; j++) {
                offset += j * imageSize;
                MPI_Recv((pres + offset), 1, array_slice_t, i, 123, cart_comm, &status);
            }
        }
        for (int i = 0; i < local_height; i++) {
            for(int j = 0; j < local_width; j++) {
                pres[i * imageSize + j] = local_pres[LP(i, j)];
            }
        }
    }
}

// Exchange borders between processes during computation
void exchange_borders() {
    if (north >= 0) {
        float *out = (local_pres + LP(0, 0)),
            *in = (local_pres + LP(-1, 0));
        MPI_Send(out, 1, border_row_t, north, 1, cart_comm);
        MPI_Recv(in, 1, border_row_t, north, 1, cart_comm, &status);
    }
    if (south >= 0) {
        float *out = (local_pres + LP(local_height - 1, 0)),
            *in = (local_pres + LP(local_height, 0));
        MPI_Send(out, 1, border_row_t, south, 1, cart_comm);
        MPI_Recv(in, 1, border_row_t, south, 1, cart_comm, &status);
    }
    if (west >= 0) {
        float *out = (local_pres + LP(0, 0)),
            *in = (local_pres + LP(0, -1));
        MPI_Send(out, 1, border_col_t, west, 1, cart_comm);
        MPI_Recv(in, 1, border_col_t, west, 1, cart_comm, &status);
    }
    if (east >= 0) {
        float *out = (local_pres + LP(0, local_width - 1)),
            *in = (local_pres + LP(0, local_width));
        MPI_Send(out, 1, border_col_t, east, 1, cart_comm);
        MPI_Recv(in, 1, border_col_t, east, 1, cart_comm, &status);
    }
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
