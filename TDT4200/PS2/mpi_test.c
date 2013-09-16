#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "global.h"

#define LP(row, col) ((row) + border) * (local_width + 2 * border) + ((col) + border)

void print_arr(float*, int, int);

int iterations,             // Number of CFD iterations (not Jacobi iterations)
    imageSize;              // Width/height of the simulation domain/output image.
unsigned char* imageBuffer;     // Buffer to hold the image to be written to file

int rank,                       // Rank of this process
    size,                       // Total number of processes
    dims[2],                    // Process grid dimensions
    coords[2],                  // Process grid coordinates of current process
    periods[2] = {0,0},         // Periodicity of process grid
    north, south, east, west,   // Ranks of neighbours in process grid
    local_height, local_width,  // Size of local subdomain
    border = 1;                 // Border thickness

MPI_Comm cart_comm;             // Cartesian communicator

MPI_Status status;              // MPI status object 

// Global and local part of the pres array (stores the xs)
float test_pres[64] = {
    0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4,
    0.5, 0.6, 0.7, 0.8, 1.5, 1.6, 1.7, 1.8,
    2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4,
    2.5, 2.6, 2.7, 2.8, 3.5, 3.6, 3.7, 3.8,
    4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4,
    4.5, 4.6, 4.7, 4.8, 5.5, 5.6, 5.7, 5.8,
    6.1, 6.2, 6.3, 6.4, 7.1, 7.2, 7.3, 7.4,
    6.5, 6.6, 6.7, 6.8, 7.5, 7.6, 7.7, 7.8
};
float* pres;
float* local_pres;
float* local_pres0;

// Global, local part of the diverg array (stores the bs)
float* local_diverg;
float* diverg;
float test_diverg[64] = {
    0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4,
    0.5, 0.6, 0.7, 0.8, 1.5, 1.6, 1.7, 1.8,
    2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4,
    2.5, 2.6, 2.7, 2.8, 3.5, 3.6, 3.7, 3.8,
    4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4,
    4.5, 4.6, 4.7, 4.8, 5.5, 5.6, 5.7, 5.8,
    6.1, 6.2, 6.3, 6.4, 7.1, 7.2, 7.3, 7.4,
    6.5, 6.6, 6.7, 6.8, 7.5, 7.6, 7.7, 7.8
};

// MPI datatypes, you might need to add some
// remember to also include them in global.h to
// make them visible in other files
MPI_Datatype border_row_t,
             border_col_t,
             array_slice_t,
             array_slice_t2;

// Function to create and commit MPI datatypes
void create_types() {
    MPI_Type_contiguous(local_width,        // count
                        MPI_FLOAT,          // old_type
                        &border_row_t);     // newtype_p
    MPI_Type_commit(&border_row_t);

    MPI_Type_vector(local_height,           // count
                    1,                      // blocklength
                    local_width + 2,        // stride
                    MPI_FLOAT,              // old_type
                    &border_col_t);         // newtype_p
    MPI_Type_commit(&border_col_t);

    MPI_Type_contiguous(local_width,        // count
                    MPI_FLOAT,              // old_type
                    &array_slice_t);        // newtype_p
    MPI_Type_commit(&array_slice_t);

    // Used with MPI_Scatter, but does not work properly
    // MPI_Type_vector(local_height,           // count
    //                 local_width,            // blocklength
    //                 imageSize,              // stride
    //                 MPI_FLOAT,              // old_type
    //                 &array_slice_t2));      // newtype_p
    // MPI_Type_commit(&array_slice_t2);
}

void test_disiribute_diverg() {
    // if (rank == 0) {
    //     int displs[size];
    //     for (int i = 0; i < size / local_height; i++) {
    //         for (int j = 0; j < local_height; j++) {
    //             displs[i * local_height + j] = i * imageSize * local_height + j * local_width;
    //         }
    //     }

    //     for (int i = 1; i < size; i++) {
    //         int offset = displs[i];
    //         for (int j = 0; j < local_height; j++) {
    //             offset += j * imageSize;
    //             MPI_Send((diverg + offset), 1, array_slice_t, i, 123, cart_comm);
    //         }
    //     }

    //     for (int i = 0; i < local_height; i++) {
    //         for(int j = 0; j < local_width; j++) {
    //             local_diverg[i * local_width + j] = diverg[i * imageSize + j];
    //         }
    //     }
    // } else {
    //     for (int i = 0; i < local_height; i++) {
    //         MPI_Recv((local_diverg + local_width * i), 1, array_slice_t, 0, 123, cart_comm, &status);
    //     }
    // }
    if (rank == 0) {
        // each entry is the index for the 'grid' corresponding to each node
        int displs[size];
        for (int i = 0; i < size / local_height; i++) {
            for (int j = 0; j < local_height; j++) {
                displs[i * local_height + j] = i * imageSize * local_height + j * local_width;
            }
        }
        // distribute to every other process than itself (0)
        for (int i = 1; i < size; i++) {
            int offset = displs[i];
            for (int j = 0; j < local_height; j++) {
                offset += j * imageSize;
                MPI_Send((diverg + offset), 1, array_slice_t, i, 123, cart_comm);
                MPI_Send((pres + offset), 1, array_slice_t, i, 123, cart_comm);
            }
        }
        // fix local_diverg 'manually'
        for (int i = 0; i < local_height; i++) {
            for(int j = 0; j < local_width; j++) {
                local_diverg[i * local_width + j] = diverg[i * imageSize + j];
            }
        }
    } else {
        // receive local_diverg from process 0
        for (int i = 0; i < local_height; i++) {
            MPI_Recv((local_diverg + local_width * i), 1, array_slice_t, 0, 123, cart_comm, &status);
            MPI_Recv((local_pres + 1 + (i + 1) * (local_width + 2)), 1, array_slice_t, 0, 123, cart_comm, &status);
        }
    }
    printf("%d: start\n", rank);
    print_arr(local_pres, local_height+2, local_width+2);
    printf("%d: end\n", rank);
    // implementation using scatter
    // MPI_Scatter(diverg,
    //             1,
    //             array_slice_t2,
    //             local_diverg,
    //             local_width*local_height,
    //             MPI_FLOAT,
    //             rank,
    //             cart_comm);

    // print local_diverg
    // printf("%d:======\n", rank);
    // for (int i = 0; i < local_height*local_width; i++) {
    //         printf("%f ", local_diverg[i]);
    //     }
    // printf("\n======\n");
}

void test_exchange_border() {
    printf("%d: exchanging borders\n", rank);
    for (int i = 0; i < local_height; i++) {
        for (int j = 0; j < local_width; j++) {
            local_pres[LP(i, j)] = - rank - 0.1 * (i * local_width + j);
        }
    }

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
    print_jacobi(local_pres);
    printf("%d: done\n", rank);
}

void test_gather_pressure() {
    for (int i = 0; i < local_height; i++) {
        for (int j = 0; j < local_width; j++) {
            local_pres[LP(i, j)] = - rank - 0.1 * (i * local_width + j);
        }
    }
    printf("rank: %d, last: %f\n", rank, local_pres[LP(local_height-1, local_width-1)]);
    if (rank != 0) {
        for (int j = 0; j < local_height; j++) {
            MPI_Send((local_pres + LP(j, 0)), 1, array_slice_t, 0, 123, cart_comm);
        }
    } else if (rank == 0) {
        int displs[size];
        for (int i = 0; i < size / local_height; i++) {
            for (int j = 0; j < local_height; j++) {
                displs[i * local_height + j] = i * imageSize * local_height + j * local_width;
                printf("d: %d\n", displs[i*local_height+j]);
            }
        }

        for (int i = 1; i < size; i++) {
            int offset = displs[i];
            for (int j = 0; j < local_height; j++) {
                offset += j * imageSize;
                float *buf = (float*) malloc(sizeof(float)*local_width);
                MPI_Recv(buf, 1, array_slice_t, i, 123, cart_comm, &status);
                printf("received from %d: ", i);
                for(int i = 0; i < local_width; i++) {
                    printf("%f ", buf[i]);
                }
                printf("\n");
                // MPI_Recv((pres + offset), 1, array_slice_t, i, 123, cart_comm, &status);
            }
        }

        for (int i = 0; i < local_height; i++) {
            for(int j = 0; j < local_width; j++) {
                pres[i * imageSize + j] = local_pres[LP(i, j)];
            }
        }
        print_arr(pres, imageSize, imageSize);
    }
}

void print_arr(float *a, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%f ", a[i * col + j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv){
    // Reading command line arguments
    iterations = 8;
    imageSize = 8;
    if (argc == 3) {
        iterations = atoi(argv[1]);
        imageSize = atoi(argv[2]);
    }
    iterations = 1;
    imageSize = 8;

    // MPI initialization, getting rank and size
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Creating cartesian communicator
    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    // Finding neighbours processes
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);

    // Determining size of local subdomain
    local_height = imageSize/dims[0];
    local_width = imageSize/dims[1];

    // Creating and commiting MPI datatypes for message passing
    create_types();

    // Allocating memory for local arrays
    local_pres = (float*)malloc(sizeof(float)*(local_width + 2*border)*(local_height+2*border));
    local_pres0 = (float*)malloc(sizeof(float)*(local_width + 2*border)*(local_height+2*border));
    local_diverg = (float*)malloc(sizeof(float)*local_width*local_height);

    // Initializing the CFD computation, only one process should do this.
    if (rank == 0) {
        pres = &test_pres;
        diverg = &test_diverg;

        imageBuffer = (unsigned char*)malloc(sizeof(unsigned char)*imageSize*imageSize);

        printf("local_width: %d\n", local_width);
        printf("local_height: %d\n", local_height);
        printf("imageSize: %d\n", imageSize);
    }

    test_disiribute_diverg();
    // test_exchange_border();
    // test_gather_pressure();

    // Finalize
    MPI_Finalize();
}
