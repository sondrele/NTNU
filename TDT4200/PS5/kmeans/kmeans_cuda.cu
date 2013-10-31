#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <cuda.h>
#include <curand.h>

// Type for points
typedef struct{
    float x;    // x coordinate
    float y;    // y coordinate
    int cluster; // cluster this point belongs to
} Point;

// Type for centroids
typedef struct{
    float x;    // x coordinate
    float y;    // y coordinate
    int nPoints; // number of points in this cluster
} Centroid;

// Global variables
int nPoints;   // Number of points
int nClusters; // Number of clusters/centroids

Point* points;       // Array containig all points
Centroid* centroids; // Array containing all centroids


// Reading command line arguments
void parse_args(int argc, char** argv){
    if(argc != 3){
        printf("Useage: kmeans nClusters nPoints\n");
        exit(-1);
    }
    nClusters = atoi(argv[1]);
    nPoints = atoi(argv[2]);

    if (nPoints < 64 || nClusters < 1) {
        printf("nClusters must be greater than 0\nnPoints must be greater than or equal to 64\n");
        exit(1);
    }
}


// Create random point
Point create_random_point(){
    Point p;
    p.x = ((float)rand() / (float)RAND_MAX) * 1000.0 - 500.0;
    p.y = ((float)rand() / (float)RAND_MAX) * 1000.0 - 500.0;
    p.cluster = rand() % nClusters;
    return p;
}

// Create random centroid
Centroid create_random_centroid(){
    Centroid p;
    p.x = ((float)rand() / (float)RAND_MAX) * 1000.0 - 500.0;
    p.y = ((float)rand() / (float)RAND_MAX) * 1000.0 - 500.0;
    p.nPoints = 0;
    return p;
}

__host__ Centroid device_create_random_centroid(){
    Centroid c;
    c.x = ((float)rand() / (float)RAND_MAX) * 1000.0 - 500.0;
    c.y = ((float)rand() / (float)RAND_MAX) * 1000.0 - 500.0;
    c.nPoints = 0;
    return c;
}


// Initialize random data
// Points will be uniformly distributed
void init_data(){
    points = (Point*)malloc(sizeof(Point)*nPoints);
    for(int i = 0; i < nPoints; i++){
        points[i] = create_random_point();
        if(i < nClusters){
            points[i].cluster = i;
        }
    }

    centroids = (Centroid*)malloc(sizeof(Centroid)*nClusters);
    for(int i = 0; i < nClusters; i++){
        centroids[i] = create_random_centroid();
    }
}

// Initialize random data
// Points will be placed in circular clusters 
void init_clustered_data(){
    float diameter = 500.0/sqrt(nClusters);

    centroids = (Centroid*)malloc(sizeof(Centroid)*nClusters);
    for(int i = 0; i < nClusters; i++){
        centroids[i] = create_random_centroid();
    }

    points = (Point*)malloc(sizeof(Point)*nPoints);
    for(int i = 0; i < nPoints; i++){
        points[i] = create_random_point();
        if(i < nClusters){
            points[i].cluster = i;
        }
    }

    for(int i = 0; i < nPoints; i++){
        int c = points[i].cluster;
        points[i].x = centroids[c].x + ((float)rand() / (float)RAND_MAX) * diameter - (diameter/2);
        points[i].y = centroids[c].y + ((float)rand() / (float)RAND_MAX) * diameter - (diameter/2);
        points[i].cluster = rand() % nClusters;
    }

    for(int i = 0; i < nClusters; i++){
        centroids[i] = create_random_centroid();
    }
}


// Print all points and centroids to standard output
void print_data(){
    for(int i = 0; i < nPoints; i++){
        printf("%f\t%f\t%d\t\n", points[i].x, points[i].y, points[i].cluster);
    }
    printf("\n\n");
    for(int i = 0; i < nClusters; i++){
        printf("%f\t%f\t%d\t\n", centroids[i].x, centroids[i].y, i);
    }
}

// Print all points and centroids to a file
// File name will be based on input argument
// Can be used to print result after each iteration
void print_data_to_file(int i){
    char filename[15];
    sprintf(filename, "%04d.dat", i);
    FILE* f = fopen(filename, "w+");

    for(int i = 0; i < nPoints; i++){
        fprintf(f, "%f\t%f\t%d\t\n", points[i].x, points[i].y, points[i].cluster);
    }
    fprintf(f,"\n\n");
    for(int i = 0; i < nClusters; i++){
        fprintf(f,"%f\t%f\t%d\t\n", centroids[i].x, centroids[i].y, i);
    }

    fclose(f);
}


// Computing distance between point and centroid
float distance(Point a, Centroid b){
    float dx = a.x - b.x;
    float dy = a.y - b.y;

    return sqrt(dx*dx + dy*dy);
}

// Cuda variables
Point *input_points;
Centroid *input_centroids;
int *cuda_updated;
int *cuda_nClusters;
int *cuda_nPoints;
int *cuda_nthreads;


__global__ void device_compute_centroids1(Point *d_points, Centroid *d_centroids, 
    int *d_nClusters, int *d_nPoints, int *d_nthreads);

__global__ void device_compute_centroids2(Point *d_points, Centroid *d_centroids, 
    int *d_nClusters, int *d_nPoints);

__global__ void device_reassign_points(Point *d_points, Centroid *d_centroids, 
    int *d_updated, int *d_nClusters, int *d_nPoints, int *d_nthreads);

int main(int argc, char** argv) {
    srand(0);
    parse_args(argc, argv);

    // Create random data, either function can be used.
    //init_clustered_data();
    init_data();

    int nThreadsPerBlock = 64;

    dim3 numBlocks, threadsPerBlock;
    numBlocks.x = nPoints / nThreadsPerBlock;
    threadsPerBlock.x = nThreadsPerBlock;

    // 1. Allocate buffers for the points and clusters
    cudaMalloc((void**) &input_points, sizeof(Point) * nPoints);
    cudaMalloc((void**) &input_centroids, sizeof(Centroid) * nClusters);
    cudaMalloc((void**) &cuda_updated, sizeof(int));
    cudaMalloc((void**) &cuda_nClusters, sizeof(int));
    cudaMalloc((void**) &cuda_nPoints, sizeof(int));
    cudaMalloc((void**) &cuda_nthreads, sizeof(int));

    cudaMemcpy(cuda_nClusters, &nClusters, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_nPoints, &nPoints, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_nthreads, &nThreadsPerBlock, sizeof(int), cudaMemcpyHostToDevice);

    // Iterate until no points are updated
    int updated = 1;
    
    cudaMemcpy(input_points, points, sizeof(Point) * nPoints, cudaMemcpyHostToDevice);
    cudaMemcpy(input_centroids, centroids, sizeof(Centroid) * nClusters, cudaMemcpyHostToDevice);

    size_t smem_size = sizeof(Point) * nThreadsPerBlock + sizeof(Centroid) * nClusters;

    // Because the 'device_compute_centroids1' function does not work correctly,
    // this variable is set to 0 to go for more serial appreach that works, 
    // but is slower (maybe even than a serial approach running on the host, 
    // because of the data movement between host and device. This of course, 
    // depends on the amount of Centroids.)
    int use_function1 = 0;

    while(updated) {
        updated = 0;

        if (use_function1) {
            // Transfer the points and clusters to device
            cudaMemcpy(input_points, points, sizeof(Point) * nPoints, cudaMemcpyHostToDevice);
            size_t size = sizeof(Centroid) * nThreadsPerBlock + sizeof(Point) * nPoints;
            // Reset centroid positions
            device_compute_centroids1<<<nClusters, threadsPerBlock, size>>>(input_points, 
                input_centroids, cuda_nClusters, cuda_nPoints, cuda_nthreads);
            // Transfer data to host
            cudaMemcpy(centroids, input_centroids, sizeof(Centroid) * nClusters, 
                cudaMemcpyDeviceToHost);
        } else {
            // Transfer the points and clusters to device
            cudaMemcpy(input_points, points, sizeof(Point) * nPoints, cudaMemcpyHostToDevice);
            // Reset centroid positions
            device_compute_centroids2<<<nClusters, 1>>>(input_points, input_centroids, 
                cuda_nClusters, cuda_nPoints);
            // Transfer data to host
            cudaMemcpy(centroids, input_centroids, sizeof(Centroid) * nClusters, 
                cudaMemcpyDeviceToHost);
        }

        // Because this function involes MATH rand(), it cannot be called from the kernel
        // By using the cuRand(), random numbers could have been generated on the device as well,
        // but it results in different results on host and device, and is thus hard to debug.
        for(int i = 0; i < nClusters; i++) {
            // If a centroid lost all its points, we give it a random position
            // (to avoid dividing by 0)
            if(centroids[i].nPoints == 0) {
                centroids[i] = create_random_centroid();
            }
            else {
                centroids[i].x /= centroids[i].nPoints;
                centroids[i].y /= centroids[i].nPoints;
            }
        }

        // Transfer the Centroids to the device, 
        // the Points is unchanged from the last time this function was called
        cudaMemcpy(input_centroids, centroids, sizeof(Centroid) * nClusters, 
            cudaMemcpyHostToDevice);
        // Reassign points
        device_reassign_points<<<numBlocks, threadsPerBlock, smem_size>>>(input_points, 
            input_centroids, cuda_updated, cuda_nClusters, cuda_nPoints, cuda_nthreads);
        // Transfer the points back to the host, the Centroids remains unchanged
        cudaMemcpy(points, input_points, sizeof(Point) * nPoints, cudaMemcpyDeviceToHost);
        cudaMemcpy(&updated, cuda_updated, sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaFree(input_points);
    cudaFree(input_centroids);
    cudaFree(cuda_updated);
    cudaFree(cuda_nClusters);
    cudaFree(cuda_nPoints);
    cudaFree(cuda_nthreads);

    print_data();
}

// A quite naive approach, without any use of reduction for the Points
// Some kind of reduction is tried implemented in the _next_ function, but the results
// is not completely satisfactory, thus this function is included as well.
// Note: set the variable 'use_function1' to 0 to call this function
__global__ void device_compute_centroids2(Point *d_points, Centroid *d_centroids, 
    int *d_nClusters, int *d_nPoints) 
{
    // extern __shared__ Centroid shared[];
    __shared__ Centroid s_centroid[1];

    int gid = blockIdx.x * blockDim.x; // + threadIdx.x
    
    int tid = threadIdx.x;

    // Reset centroid to 0
    s_centroid[tid] = (Centroid) {0.0, 0.0, 0};

    // Compute new centroids positions
    for (int i = 0; i < *d_nPoints; i++) {
        if (gid == d_points[i].cluster) {
            s_centroid[tid].x += d_points[i].x;
            s_centroid[tid].y += d_points[i].y;
            s_centroid[tid].nPoints++;
        }
    }
    // Transfer computed centroid back to global memory
    d_centroids[gid] = s_centroid[tid];
}

// This function is called for each of the Clusters, 64 threads run in parallel
// reducing all the points connected to the Cluster (identified by the block id)
// It does not seam to produce a 100% correct output, but sometimes look all right
// Note: set the variable 'use_function1' to 1 to call this function
__global__ void device_compute_centroids1(Point *d_points, Centroid *d_centroids, 
    int *d_nClusters, int *d_nPoints, int *d_nthreads) 
{
    extern __shared__ Centroid s_compute_centroids[];
    Centroid *s_centroid = s_compute_centroids;
    Point *s_points = (Point *) &s_centroid[*d_nthreads];

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    // Reset centroid to 0
    s_centroid[tid] = (Centroid) {0.0, 0.0, 0};

    int length = *d_nPoints / *d_nthreads;
    int start = tid * length;
    int end = start + length;

    if (tid == 0 && length < 0) {
        for (int i = 0; i < *d_nPoints; i++) {
            s_points[i] = d_points[i];
        }
    } else {
        for (int i = start; i < end; i++) {
            s_points[i] = d_points[i];
        }
    }
    __syncthreads();

    // If the length is small (i.e. there is fewer points than 64),
    // thread 0 does all the work
    if (tid == 0 && length < 0) {
        for (int i = 0; i < *d_nPoints; i++) {
            int cluster_id = d_points[i].cluster;

            if (cluster_id == bid) {
                s_centroid[0].x += d_points[i].x;
                s_centroid[0].y += d_points[i].y;
                s_centroid[0].nPoints++;
            }
        }
    } else {
        // A kind of reduction, where each thread runs through it's dedicated part
        // of the Points array and checks to see if the points is connected to the 
        // corresponding block. If it is, each thread updates it's own centroid
        // in shared memory
        for (int i = start; i < end; i++) {
            int cluster_id = d_points[i].cluster;

            if (cluster_id == bid) {
                s_centroid[tid].x += d_points[i].x;
                s_centroid[tid].y += d_points[i].y;
                s_centroid[tid].nPoints += 1;
            }
        }
        __syncthreads();

        // Thread 0 goes over the centroids in shared memory, and sums them together
        if (tid == 0) {
            for (int i = 0; i < 64; i++) {
                s_centroid[0].x += s_centroid[i].x;
                s_centroid[0].y += s_centroid[i].y;
                s_centroid[0].nPoints += s_centroid[i].nPoints;
            }
        }
    }

    // Transfer computed centroid back to global memory
    if (tid == 0) {
        d_centroids[bid] = s_centroid[0];
    }
}

// This function is runs with one thread for every point, calculating what
// centroid the respective point belongs to
__global__ void device_reassign_points(Point *d_points, Centroid *d_centroids, 
    int *d_updated, int *d_nClusters, int *d_nPoints, int *d_nthreads) 
{
    // The shared memory consists of 64 points (1 for each thread per block), and 'd_nClusters' Centroids
    extern __shared__ Point s_reassign_points[];
    Point *s_points = s_reassign_points;
    Centroid *s_centroids = (Centroid *) &s_points[*d_nthreads];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Make sure that updated is 0
    if (tid == 0) {
        *d_updated = 0;
    }

    // Init shared Points
    s_points[tid] = d_points[gid];

    // Init shared Centroids
    int length = *d_nClusters / *d_nthreads;
    int start = tid * length;
    int end = start + length;

    // If number of clusters is small, let thread 0 do all the work.
    // For some reason this initialization does not work in parallel, so thread 0 does all the work.
    if (tid == 0) {
        for (int i = 0; i < *d_nClusters; i++) {
            s_centroids[i] = d_centroids[i];
        }
    } 
    else if (length > 1 && 0) {
        for (int i = start; i < end; i++) {
            s_centroids[i] = d_centroids[i];
        }
    }
    __syncthreads();

    //Reassign points to closest centroid
    float bestDistance = DBL_MAX;
    int bestCluster = -1;

    for(int i = 0; i < *d_nClusters; i++) {
        Point a = s_points[tid];
        Centroid b = s_centroids[i];
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        float d = sqrt(dx*dx + dy*dy);

        if(d < bestDistance) {
            bestDistance = d;
            bestCluster = i;
        }
    }

    // If one point got reassigned to a new cluster, we have to do another iteration
    if(bestCluster != s_points[tid].cluster) {
        *d_updated = 1;
    }

    // Update the points in the global memory
    d_points[gid].cluster = bestCluster;
}
