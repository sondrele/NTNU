#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <omp.h>

// Type for points
typedef struct{
    double x;    // x coordinate
    double y;    // y coordinate
    int cluster; // cluster this point belongs to
} Point;

// Type for centroids
typedef struct{
    double x;    // x coordinate
    double y;    // y coordinate
    int nPoints; // number of points in this cluster
} Centroid;

// Global variables
int nPoints;   // Number of points
int nClusters; // Number of clusters/centroids
int nThreads;  // Number of threads to use

Point* points;       // Array containig all points
Centroid* centroids; // Array containing all centroids


// Reading command line arguments
void parse_args(int argc, char** argv){
    if(argc != 4){
        printf("Useage: kmeans nThreads nClusters nPoints\n");
        exit(-1);
    }
    nThreads = atoi(argv[1]);
    nClusters = atoi(argv[2]);
    nPoints = atoi(argv[3]);
}


// Create random point
Point create_random_point(){
    Point p;
    p.x = ((double)rand() / (double)RAND_MAX) * 1000.0 - 500.0;
    p.y = ((double)rand() / (double)RAND_MAX) * 1000.0 - 500.0;
    p.cluster = rand() % nClusters;
    return p;
}


// Create random centroid
Centroid create_random_centroid(){
    Centroid p;
    p.x = ((double)rand() / (double)RAND_MAX) * 1000.0 - 500.0;
    p.y = ((double)rand() / (double)RAND_MAX) * 1000.0 - 500.0;
    p.nPoints = 0;
    return p;
}


// Initialize random data
// Points will be uniformly distributed
void init_data(){
    points = malloc(sizeof(Point)*nPoints);
    for(int i = 0; i < nPoints; i++){
        points[i] = create_random_point();
        if(i < nClusters){
            points[i].cluster = i;
        }
    }

    centroids = malloc(sizeof(Centroid)*nClusters);
    for(int i = 0; i < nClusters; i++){
        centroids[i] = create_random_centroid();
    }
}

// Initialize random data
// Points will be placed in circular clusters 
void init_clustered_data(){
    double diameter = 500.0/sqrt(nClusters);

    centroids = malloc(sizeof(Centroid)*nClusters);
    for(int i = 0; i < nClusters; i++){
        centroids[i] = create_random_centroid();
    }

    points = malloc(sizeof(Point)*nPoints);
    for(int i = 0; i < nPoints; i++){
        points[i] = create_random_point();
        if(i < nClusters){
            points[i].cluster = i;
        }
    }

    for(int i = 0; i < nPoints; i++){
        int c = points[i].cluster;
        points[i].x = centroids[c].x + ((double)rand() / (double)RAND_MAX) * diameter - (diameter/2);
        points[i].y = centroids[c].y + ((double)rand() / (double)RAND_MAX) * diameter - (diameter/2);
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
double distance(Point a, Centroid b){
    double dx = a.x - b.x;
    double dy = a.y - b.y;

    return sqrt(dx*dx + dy*dy);
}

int main(int argc, char** argv){
    parse_args(argc, argv);

    // Create random data, either function can be used.
    //init_clustered_data();
    init_data();

    omp_set_num_threads(nThreads);

    omp_lock_t writelock;
    omp_init_lock(&writelock);

    // Iterate until no points are updated
    int updated = 1;
    while (updated) {
        updated = 0;

        #pragma omp parallel for
        for (int i = 0; i < nClusters; i++) {
            centroids[i].x = 0.0;
            centroids[i].y = 0.0;
            centroids[i].nPoints= 0;
        }

        // Compute new centroids positions
        for (int i = 0; i < nPoints; i++) {
            int c = points[i].cluster;
            centroids[c].x += points[i].x;
            centroids[c].y += points[i].y;
            centroids[c].nPoints++;
        }

        for (int i = 0; i < nClusters; i++) {
            // If a centroid lost all its points, we give it a random position
            // (to avoid dividing by 0)
            if (centroids[i].nPoints == 0) {
                centroids[i] = create_random_centroid();
            } else {
                centroids[i].x /= centroids[i].nPoints;
                centroids[i].y /= centroids[i].nPoints;
            }
        }


        double bestDistance = DBL_MAX;
        int bestCluster = -1;
        double d;
        //Reassign points to closest centroid
        #pragma omp parallel for \
        private(bestDistance, bestCluster, d)
        for (int i = 0; i < nPoints; i++) {
            bestDistance = DBL_MAX;
            bestCluster = -1;
            for (int j = 0; j < nClusters; j++) {
                d = distance(points[i], centroids[j]);
                if (d < bestDistance) {
                    bestDistance = d;
                    bestCluster = j;
                }
            }

            // If one point got reassigned to a new cluster, we have to do another iteration
            if (bestCluster != points[i].cluster) {
                updated = 1;
            }
            points[i].cluster = bestCluster;
        }
    }

    print_data();
}
