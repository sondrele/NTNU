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

Point *input_points;
Centroid *input_centroids;
int *cuda_updated;
int *cuda_nClusters;
int *cuda_nPoints;

__global__ void device_reset_centroid_position(Centroid *input_centroids, Point *input_points, int *cuda_nPoints) {
    // extern __shared__ Point sdata[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // int tid = threadIdx.x;
    // sdata[tid] = input_points[i];
    __syncthreads();

    input_centroids[i].x = 0.0;
    input_centroids[i].y = 0.0;
    input_centroids[i].nPoints = 0;

    // for (int s = 1; s < blockDim.x; s *= 2) {
    //     if (tid % (2 * s) == 0) {
    //         sdata[tid].x += sdata[tid + s].x;
    //         sdata[tid].y += sdata[tid + s].y;
    //         sdata[tid].nPoints += sdata[tid + s].nPoints;
    //     }
    //     __syncthreads();
    // }

    for(int j = 0; j < *cuda_nPoints; j++) {
        if (i == input_points[j].cluster) {
            input_centroids[i].x += input_points[j].x;
            input_centroids[i].y += input_points[j].y;
            input_centroids[i].nPoints += 1;
        }
    }
}

__global__ void device_reset_centroid_pos(Centroid *input_centroids) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    input_centroids[i].x = 0.0;
    input_centroids[i].y = 0.0;
    input_centroids[i].nPoints = 0;
}

__global__ void device_reassign_points(Point *d_points, Centroid *d_centroids, int *d_updated, int *d_nClusters, int *d_nPoints);

__global__ void device_compute_centroid_positions(Point *d_points, Centroid *d_centroids, int *d_nClusters, int *d_nPoints);

int main(int argc, char** argv) {
    srand(0);
    parse_args(argc, argv);

    // Create random data, either function can be used.
    //init_clustered_data();
    init_data();

    dim3 numBlocks, threadsPerBlock;
    numBlocks.x = nPoints / 64;
    threadsPerBlock.x = 64;

    // 1. Allocate buffers for the points and clusters
    cudaMalloc((void**) &input_points, sizeof(Point) * nPoints);
    cudaMalloc((void**) &input_centroids, sizeof(Centroid) * nClusters);
    cudaMalloc((void**) &cuda_updated, sizeof(int));
    cudaMalloc((void**) &cuda_nClusters, sizeof(int));
    cudaMalloc((void**) &cuda_nPoints, sizeof(int));

    cudaMemcpy(cuda_nClusters, &nClusters, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_nPoints, &nPoints, sizeof(int), cudaMemcpyHostToDevice);

    // Iterate until no points are updated
    int updated = 1;
    
    cudaMemcpy(input_points, points, sizeof(Point) * nPoints, cudaMemcpyHostToDevice);
    cudaMemcpy(input_centroids, centroids, sizeof(Centroid) * nClusters, cudaMemcpyHostToDevice);

    size_t smem_size = sizeof(Point) * 64 + sizeof(Centroid) * nClusters;

    while(updated) {
        updated = 0;

        if (nClusters < 0) {
            for (int i = 0; i < nClusters; i++) {
                centroids[i].x = 0.0;
                centroids[i].y = 0.0;
                centroids[i].nPoints= 0;
            }
            // Transfer the points and clusters to device
            cudaMemcpy(input_points, points, sizeof(Point) * nPoints, cudaMemcpyHostToDevice);
            cudaMemcpy(input_centroids, centroids, sizeof(Centroid) * nClusters, cudaMemcpyHostToDevice);
            // Reset centroid positions
            device_compute_centroid_positions<<<nClusters, threadsPerBlock, smem_size>>>(input_points, input_centroids, cuda_nClusters, cuda_nPoints);
            // Transfer data to host
            cudaMemcpy(centroids, input_centroids, sizeof(Centroid) * nClusters, cudaMemcpyDeviceToHost);
        } else {
            // Reset centroid positions
            for(int i = 0; i < nClusters; i++){
                centroids[i].x = 0.0;
                centroids[i].y = 0.0;
                centroids[i].nPoints= 0;
            }

            // Compute new centroids positions
            for(int i = 0; i < nPoints; i++){
                int c = points[i].cluster;
                centroids[c].x += points[i].x;
                centroids[c].y += points[i].y;
                centroids[c].nPoints++;
            }
        }

        // Because this function involes MATH rand(), it cannot be called from the kernel
        // By using the cuda_rand a random number could have been achieved, but it will
        // result in different result when running the host vs the cuda coda.
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

        // Transfer the Centroids to the device, the Points is unchanged from the last time this function was called
        cudaMemcpy(input_centroids, centroids, sizeof(Centroid) * nClusters, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_updated, &updated, sizeof(int), cudaMemcpyHostToDevice);

        // Reassign points
        device_reassign_points<<<numBlocks, threadsPerBlock, smem_size>>>(input_points, input_centroids, cuda_updated, cuda_nClusters, cuda_nPoints);
        
        // Transfer the points back to the host, the Centroids remains unchanged
        cudaMemcpy(points, input_points, sizeof(Point) * nPoints, cudaMemcpyDeviceToHost);
        cudaMemcpy(&updated, cuda_updated, sizeof(int), cudaMemcpyDeviceToHost);

    }

    cudaFree(input_points);
    cudaFree(input_centroids);
    cudaFree(cuda_updated);
    cudaFree(cuda_nClusters);
    cudaFree(cuda_nPoints);

    print_data();
}

__global__ void device_compute_centroid_positions(Point *d_points, Centroid *d_centroids, int *d_nClusters, int *d_nPoints) {
    extern __shared__ Point s[];
    Point *s_points = s;
    Centroid *s_centroids = (Centroid *) &s_points[64];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Init shared Points
    s_points[tid] = d_points[i];

    // Init shared Centroids
    int length = *d_nClusters / 64;

    // If number of clusters is small, let thread 0 do all the work, otherwise share between the 64 threads
    if (i == 0 && length < 1) {
        for (int j = 0; j < *d_nClusters; j++) {
            s_centroids[j] = (Centroid) {0.0, 0.0, 0};
        }
    } else if (length > 1) {
        int start = tid * length;
        int end = start + length;
        for (int j = start; j < end; j++) {
            s_centroids[j] = (Centroid) {0.0, 0.0, 0};
        }
    }
    __syncthreads();
    
    // if (i == 0) {
    //     printf("x:%f y:%f c:%d\n", d_centroids[0].x, d_centroids[0].y, d_centroids[0].nPoints);
    //     printf("x:%f y:%f c:%d\n", s_centroids[0].x, s_centroids[0].y, s_centroids[0].nPoints);
    // }

    int c = s_points[tid].cluster;
    atomicAdd(&s_centroids[c].x, s_points[tid].x);
    atomicAdd(&s_centroids[c].y, s_points[tid].y);
    atomicAdd(&s_centroids[c].nPoints, 1);
    __syncthreads();

    // // Reduce the Centroids
    // if (i < *d_nClusters) {
    //     atomicAdd(&d_centroids[i].x, s_centroids[i].x);
    //     atomicAdd(&d_centroids[i].y, s_centroids[i].y);
    //     atomicAdd(&d_centroids[i].nPoints, s_centroids[i].nPoints);
    // }
    if (tid == 0) {
        for(int x = 0; x < *d_nClusters; x++){
            atomicAdd(&d_centroids[x].x, s_centroids[x].x);
            atomicAdd(&d_centroids[x].y, s_centroids[x].y);
            atomicAdd(&d_centroids[x].nPoints, s_centroids[x].nPoints);
        }
    }
}

__global__ void device_reassign_points(Point *d_points, Centroid *d_centroids, int *d_updated, int *d_nClusters, int *d_nPoints) {
    // The shared memory consists of 64 points (1 for each thread per block), and 'd_nClusters' Centroids
    extern __shared__ Point s[];
    Point *s_points = s;
    Centroid *s_centroids = (Centroid *) &s_points[64];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Init shared Points
    s_points[tid] = d_points[i];

    // Init shared Centroids
    int length = *d_nClusters / 64;

    // If number of clusters is small, let thread 0 do all the work, otherwise share between the 64 threads
    if (tid == 0 && length < 1) {
        for (int b = 0; b < *d_nClusters; b++) {
            s_centroids[b] = d_centroids[b];
        }
    } else if (length > 1) {
        int start = tid * length;
        int end = start + length;
        for (int b = start; b < end; b++) {
            s_centroids[b] = d_centroids[b];
        }
    }
    __syncthreads();


    //Reassign points to closest centroid
    float bestDistance = DBL_MAX;
    int bestCluster = -1;

    for(int j = 0; j < *d_nClusters; j++) {
        Point a = s_points[tid];
        Centroid b = s_centroids[j];
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        float d = sqrt(dx*dx + dy*dy);

        if(d < bestDistance) {
            bestDistance = d;
            bestCluster = j;
        }
    }

    // If one point got reassigned to a new cluster, we have to do another iteration
    if(bestCluster != s_points[tid].cluster) {
        *d_updated = 1;
    }

    // Update the points in the global memory
    d_points[i].cluster = bestCluster;
}
