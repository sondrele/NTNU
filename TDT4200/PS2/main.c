#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "config.h"
#include "global.h"
#include "bmp.h"

struct Configuration config;    // Stores arrays needed for the CFD computation, should not be modified

int iterations,                 // Number of CFD iterations (not Jacobi iterations)
    imageSize;                  // Width/height of the simulation domain/output image.
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
            pres_and_diverg_t,
            local_diverg_t,
            local_pres_t;

// Global and local part of the pres array (stores the xs)
float* pres;
float* local_pres;
float* local_pres0;

// Global, local part of the diverg array (stores the bs)
float* local_diverg;
float* diverg;

// Function to create and commit MPI datatypes
// Each datatype is resized to size of float, it looks like this fixes 
// segmentation fault issues.
void create_types() {
    MPI_Datatype border_row_t0;
    MPI_Type_contiguous(local_width,            // count
                        MPI_FLOAT,              // old_type
                        &border_row_t0);        // newtype_p
    MPI_Type_create_resized(border_row_t0, 0, sizeof(float), &border_row_t);
    MPI_Type_commit(&border_row_t);

    MPI_Datatype border_col_t0;
    MPI_Type_vector(local_height,               // count
                    1,                          // blocklength
                    local_width + 2,            // stride
                    MPI_FLOAT,                  // old_type
                    &border_col_t0);            // newtype_p
    MPI_Type_create_resized(border_col_t0, 0, sizeof(float), &border_col_t);
    MPI_Type_commit(&border_col_t);

    MPI_Datatype pres_and_diverg_t0;
    MPI_Type_vector(local_height,               // count
                    local_width,                // blocklength
                    imageSize + 2,              // stride
                    MPI_FLOAT,                  // old_type
                    &pres_and_diverg_t0);       // newtype_p
    MPI_Type_create_resized(pres_and_diverg_t0, 0, sizeof(float), &pres_and_diverg_t);
    MPI_Type_commit(&pres_and_diverg_t);

    MPI_Datatype local_diverg_t0;
    MPI_Type_vector(local_height,               // count
                    local_width,                // blocklength
                    local_width,                // stride
                    MPI_FLOAT,                  // old_type
                    &local_diverg_t0);          // newtype_p
    MPI_Type_create_resized(local_diverg_t0, 0, sizeof(float), &local_diverg_t);
    MPI_Type_commit(&local_diverg_t);

    MPI_Datatype local_pres_t0;
    MPI_Type_vector(local_height,               // count
                    local_width,                // blocklength
                    local_width + 2,            // stride
                    MPI_FLOAT,                  // old_type
                    &local_pres_t0);            // newtype_p
    MPI_Type_create_resized(local_pres_t0, 0, sizeof(float), &local_pres_t);
    MPI_Type_commit(&local_pres_t);
}

// For debugging purposes
void print_stats() {
    printf("dims[0] = %d\ndims[1] = %d\n", dims[0], dims[1]);
    printf("imageSize = %d\n", imageSize);
    printf("local_height = %d\n", local_height);
    printf("local_width = %d\n", local_width);
    printf("size = %d\n", size);
}

int main (int argc, char **argv) {
    // Reading command line arguments
    iterations = 100;
    imageSize = 512;
    if (argc == 3) {
        iterations = atoi(argv[1]);
        imageSize = atoi(argv[2]);
    }

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
        initFluid(&config, imageSize, imageSize);
        pres = config.pres;
        diverg = config.div;

        imageBuffer = (unsigned char*)malloc(sizeof(unsigned char)*imageSize*imageSize);
        print_stats();
    }

    // Solving the CFD equations, one iteration for each timestep.
    // These are not the same iterations used in the Jacobi solver.
    // The solveFluid function call the Jacobi solver, wich runs for 
    // 100 iterations for each of these iterations.
    for (int i = 0; i < iterations; i++) {
        solveFluid(&config);
    }

    // Converting the density to an image and writing it to file.
    if (rank == 0) {
        // dens->pres
        densityToColor(imageBuffer, config.dens, config.N);
        write_bmp(imageBuffer, imageSize, imageSize);

        // Free fluid simulation memory
        freeFluid( &config );
    }

    // Finalize
    MPI_Finalize();
}
