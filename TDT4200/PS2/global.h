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
                    pres_and_diverg_t,
                    local_diverg_t,
                    local_pres_t;

extern float *local_pres;
extern float *local_pres0;
extern float *pres;
extern float *local_diverg;
extern float *diverg;

// Functions
void distribute_diverg();
void gather_pres();
void exchange_border(float *out, float *in, int dst);
void exchange_borders();
void jacobi_iteration();
void init_local_pres();
void jacobi(int iter);
void print_local_pres(float *jacobi);
