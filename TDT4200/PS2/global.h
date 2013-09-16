
#include <mpi.h>

// Global variables from main.c
extern int imageSize;
extern int rank, size,
    dims[2],
    coords[2],
    periods[2],
    north, south, east, west,
    local_height, local_width,
    width, height, border;

extern MPI_Comm cart_comm;

extern MPI_Status status;

extern MPI_Datatype border_row_t,
                    border_col_t,
                    array_slice_t;

extern float* local_pres;
extern float* local_pres0;
extern float* pres;
extern float* local_diverg;
extern float* diverg;

// Functions
void distribute_diverg();
void gather_pres();
void exchange_borders();
void jacobi_iteration();
void jacobi(int iter);
int on_edge(int row, int col);
float calculate_jacobi(int row, int col);
float *get_row(int row);
float *get_col(int col);
void print_jacobi();
