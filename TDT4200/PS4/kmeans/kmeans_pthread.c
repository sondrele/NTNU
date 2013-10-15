#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <pthread.h>


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

typedef struct {
    int i;
    int length;
} index_t;

index_t *ci, *pi;

void init_indexes() {
    ci = malloc(sizeof(index_t) * nThreads);

    if (nThreads > nClusters) {
        // TODO: fix 
    } else {
        int length = nClusters / nThreads;
        for (int i = 0; i < nThreads; i++) {
            ci[i].i = i * length;
            ci[i].length = i * length + length;
        }
    }

    pi = malloc(sizeof(index_t) * nThreads);
    if (nThreads > nPoints) {

    } else {
        int length = nPoints / nThreads;
        for (int i = 0; i < nThreads; i++) {
            pi[i].i = i * length;
            pi[i].length = i * length + length;
        }
    }
}

// Creates a barrier
int thread_counter = 0;
pthread_mutex_t mutex_barrier,
    mutex_centroid;
pthread_cond_t barrier_cond;

typedef struct {
    // private
    int id;
    int bestCluster;
    double bestDistance;
    double d;

    // shared
    int *updated;
} rp_args;

void thread_barrier () {
    pthread_mutex_lock(&mutex_barrier);

    thread_counter += 1;
    if (thread_counter == nThreads) {
        thread_counter = 0;
        pthread_cond_broadcast(&barrier_cond);
    } else {
        while (pthread_cond_wait(&barrier_cond, &mutex_barrier) != 0);
    }

    pthread_mutex_unlock(&mutex_barrier);
}

void* reset_centroids(void *id) {
    int tid = *(int*) id;
    // printf("%d: i=%d, l=%d\n", tid, ci[tid].i, ci[tid].length);
    for (int i = ci[tid].i; i < ci[tid].length; i++) {
        centroids[i].x = 0.0;
        centroids[i].y = 0.0;
        centroids[i].nPoints= 0;
    }

    return NULL;
}

void* compute_centroids(void *args) {
    rp_args *a = (rp_args *) args;

    for (int i = pi[a->id].i; i < pi[a->id].length; i++) {
        pthread_mutex_lock(&mutex_centroid);
        // Compute new centroids positions
        int c = points[i].cluster;
        centroids[c].x += points[i].x;
        centroids[c].y += points[i].y;
        centroids[c].nPoints++;
        pthread_mutex_unlock(&mutex_centroid);
    }

}

void* mediate_centroids(void *id) {
    int tid = *(int*) id;
    // printf("tid=%d\n", tid);
    for (int i = ci[tid].i; i < ci[tid].length; i++) {
        if(centroids[i].nPoints == 0){
            centroids[i] = create_random_centroid();
        } else {
            centroids[i].x /= centroids[i].nPoints;
            centroids[i].y /= centroids[i].nPoints;
        }
    }
    
    return NULL;
}

void* reassign_points(void *args) {
    rp_args *a = (rp_args *) args;
    for (int i = pi[a->id].i; i < pi[a->id].length; i++) {
        // printf("%d: i=%d, l=%d\n", id, i, pi[a->id].length); 
        a->bestDistance = DBL_MAX;
        a->bestCluster = -1;

        for (int j = 0; j < nClusters; j++) {
            a->d = distance(points[i], centroids[j]);
            if(a->d < a->bestDistance){
                a->bestDistance = a->d;
                a->bestCluster = j;
            }
        }

        // If one point got reassigned to a new cluster, we have to do another iteration
        if (a->bestCluster != points[i].cluster) {
            *a->updated = 1;
        }
        points[i].cluster = a->bestCluster;
    }

    return NULL;
}

void* test_barrier(void *args) {
    int tid = *(int *) args;

    printf("hello from: %d\n", tid);

    thread_barrier();

    printf("hello again: %d\n", tid);

    return NULL;
}

void* run_kmeans(void *args) {
    rp_args *a = (rp_args *) args;

    reset_centroids(&a->id);

    thread_barrier();

    compute_centroids(a);

    thread_barrier();

    mediate_centroids(&a->id);

    thread_barrier();

    reassign_points(a);

    return NULL;
}

int main(int argc, char** argv){
    srand(0);
    parse_args(argc, argv);

    // Create random data, either function can be used.
    //init_clustered_data();
    init_data();

    init_indexes();
    rp_args *args = malloc(sizeof(rp_args) * nThreads);

    // Threads
    pthread_t threads[nThreads];
    void *exit_status;
    pthread_attr_t attr;

    // Mutexes
    pthread_mutex_init(&mutex_barrier, NULL);
    pthread_mutex_init(&mutex_centroid, NULL);
    
    // Explicitly create joinable threads
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    // Iterate until no points are updated
    int updated = 1;
    while (updated) {
        updated = 0;

        // Reset centroid positions
        for (int i = 0; i < nThreads; i++) {
            args[i].id = i;
            args[i].bestCluster = -1;
            args[i].bestDistance = DBL_MAX;
            args[i].updated = &updated;

            pthread_create(&threads[i], &attr, run_kmeans, (void *) (args + i));
        }

        for (int i = 0; i < nThreads; i++) {
            pthread_join(threads[i], &exit_status);
        }
    }

    print_data();

    pthread_mutex_destroy(&mutex_barrier);
    pthread_exit(NULL);


    return 0;
}
