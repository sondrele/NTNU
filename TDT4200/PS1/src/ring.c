#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int size, rank, in;
    MPI_Status status;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int buf, dest;
    if (rank == 0) {
        buf = 0;
        MPI_Send(&buf, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
        MPI_Recv(&in, 1, MPI_INT, size - 1, 1, MPI_COMM_WORLD, &status);
        
        printf("Rank 0 received %d from rank %d\n", in, size - 1);
    } else {
        MPI_Recv(&in, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &status);
        buf = rank + in;
        dest = (rank == size - 1) ? 0 : rank + 1;
        MPI_Send(&buf, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
        
        printf("Rank %d received %d from rank %d and sent %d to rank %d\n",
            rank, in, rank - 1, buf, dest);
    }
    
    MPI_Finalize();
}
