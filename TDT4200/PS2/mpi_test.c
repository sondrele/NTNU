#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "global.h"

#define LP(row, col) ((row) + border) * (local_width + 2 * border) + ((col) + border)

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

// MPI datatypes, you might need to add some
// remember to also include them in global.h to
// make them visible in other files
MPI_Datatype border_row_t,
             border_col_t,
             array_slice_t,
             test_t;

// Global and local part of the pres array (stores the xs)
float test_pres[100] = {
    1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0,
    1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0,
    1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0,
    1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0
};
float* pres;
float* local_pres;
float* local_pres0;

// Global, local part of the diverg array (stores the bs)
float* local_diverg;
float* diverg;
float test_diverg[100] = {
    0.7, 0.8, 0.9, 0.1, 5.5, 5.5, 5.5, 5.5,
    7.0, 8.0, 9.0, 1.0, 4.5, 4.5, 4.5, 4.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
};

// Function to create and commit MPI datatypes
void create_types() {
    MPI_Type_contiguous(local_width,                // count
                        MPI_FLOAT,                  // old_type
                        &border_row_t);             // newtype_p
    MPI_Type_commit(&border_row_t);

    MPI_Type_vector(local_height,                   // count
                     1,                             // blocklength
                     local_width + 2,               // stride
                     MPI_FLOAT,                     // old_type
                     &border_col_t);                // newtype_p
    MPI_Type_commit(&border_col_t);

    MPI_Type_contiguous(local_width,                // count
                    MPI_FLOAT,                      // old_type
                    &array_slice_t);                // newtype_p
    MPI_Type_commit(&array_slice_t);
}

void test_disiribute_diverg() {
    if (rank == 1) {
        printf("%d: receiving divergence\n", rank);
        local_diverg = (float*) malloc(sizeof(float) * local_width * local_height * 2);
        for (int i = 0; i < local_height; i++) {
            MPI_Recv((local_diverg + local_width * i), 1, array_slice_t, 0, 123, cart_comm, &status);
        }
        for (int i = 0; i < local_height*local_width; i++) {
            printf("%f ", local_diverg[i]);
        }
        printf("\n");
    }

    if (rank == 0) {
        printf("distributing divergence to: %d\n", 1);
        // offset for diverg er riktig
        for (int i = 0; i <local_height; i++) {
            MPI_Send((diverg + local_width + imageSize*i), 1, array_slice_t, 1, 123, cart_comm);
        }
    }
}

void test_exchange_border() {
    if (rank == 1 && 0) {
        printf("%d: exchanging border\n", rank);
        for (int i = 0; i < ((local_width + 2) * (local_height + 2)); i++)
            local_pres[i] = 1;
        local_pres[LP(0,0)] = 1.111111;
        local_pres[LP(1,0)] = 5.555555;
        print_jacobi(local_pres);

        printf("Sending: %f %f\n", (local_pres + LP(0, 0)), (local_pres + LP(0, 0) + 6));
        MPI_Send((local_pres + LP(0, 0)), 1, border_col_t, 0, 123, cart_comm);
        MPI_Recv((local_pres0 + LP(0, -1)), 1, border_col_t, 0, 123, cart_comm, &status);
        // printf("%d receiving %f %f\n", rank, local_pres0[LP(0,0)], local_pres0[LP(1,0)]);
        print_jacobi(local_pres0);
    }

    if (rank == 0) {
        printf("%d: exchanging border\n", rank);
        for (int i = 0; i < ((local_width + 2) * (local_height + 2)); i++)
            local_pres[i] = i;

        print_jacobi(local_pres);
        // printf("%d Sending: %f %f\n", rank, *(local_pres + LP(0, 3)), *(local_pres + LP(0, 3) + 6));
        // MPI_Send((local_pres + LP(0, 3)), 1, border_col_t, 1, 123, cart_comm);
        // MPI_Recv((local_pres0 + LP(0, 4)), 1, border_col_t, 1, 123, cart_comm, &status);


        MPI_Send((local_pres + LP(1, 0)), 1, border_row_t, 2, 123, cart_comm);
        MPI_Recv((local_pres0 + LP(2, 0)), 1, border_row_t, 2, 123, cart_comm, &status);
        print_jacobi(local_pres0);
    }

    if (rank == 2) {
        printf("%d: exchanging border\n", rank);
        for (int i = 0; i < ((local_width + 2) * (local_height + 2)); i++)
            local_pres[i] = i;
        print_jacobi(local_pres);

        // printf("%d Sending: %f %f\n", rank, *(local_pres + LP(0, 3)), *(local_pres + LP(0, 3) + 6));
        MPI_Send((local_pres + LP(0, 0)), 1, border_row_t, 0, 123, cart_comm);
        MPI_Recv((local_pres0 + LP(-1, 0)), 1, border_row_t, 0, 123, cart_comm, &status);
        print_jacobi(local_pres0);
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

    // Finalize
    MPI_Finalize();
}
