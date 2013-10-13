#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

float random(){
    return ((float)rand())/((float)RAND_MAX);
}


float* create_random_matrix(int m, int n){
    float* A = ( float*)malloc(sizeof( float)*m*n);

    for(int i = 0; i < m*n; i++){
        A[i] = random();
    }

    return A;
}


void print_matrix( float* A, int m, int n){

    int max_size = 10;
    if(m > max_size || n > max_size){
        printf("WARNING: matrix too large, only printing part of it\n");
        m = max_size;
        n = max_size;
    }

    for(int y = 0; y < m; y++){
        for(int x = 0; x < n; x++){
            printf("%.4f  ", A[y*n + x]);
        }
        printf("\n");
    }
    printf("\n");
}

typedef struct {
    float *A;
    float *B;
    float *C;
    float alpha;
    float beta;
    int x_min;
    int x_max;
    int y_min;
    int y_max;
    int n;
    int k;
} sub_matrix_args;

sub_matrix_args* create_args(float *A, float *B, float *C, float alpha, float beta,
        int x_min, int x_max, int y_min, int y_max, int n, int k) {
    sub_matrix_args *args = malloc(sizeof(sub_matrix_args));
    args->A = A;
    args->B = B;
    args->C = C;
    args->alpha = alpha;
    args->beta = beta;
    args->x_min = x_min;
    args->x_max = x_max;
    args->y_min = y_min;
    args->y_max = y_max;
    args->n = n;
    args->k = k;

    return args;
}

void* calculate_sub_matrix(void *args) {
    // int counter = 0;
    // pthread_mutex_t mutex;
    // pthread_cond_t cond_var;

    sub_matrix_args *a = (sub_matrix_args *) args;

    for(int x = a->x_min; x < a->x_max; x++){
        for(int y = a->y_min; y < a->y_max; y++){
            a->C[y * a->n + x] *= a->beta;
            for(int z = 0; z < a->k; z++){
                a->C[y * a->n + x] += a->alpha * a->A[y * a->k + z] * a-> B[z * a->n + x];
            }
        }
    }

    // pthread_mutex_lock(&mutex);
    // counter++;
    // if(counter == 4){
    //     counter = 0;
    //     pthread_cond_broadcast(&cond_var);
    // } else {
    //     while(pthread_cond_wait(&cond_var, &mutex) != 0);
    // }
    // pthread_mutex_unlock(&mutex);
}

int main(int argc, char** argv){

    // Number of threads to use
    int nThreads = 1;

    // Matrix sizes
    int m = 2;
    int n = 2;
    int k = 2;

    // Reading command line arguments
    if(argc != 5){
        printf("useage: gemm nThreads m n k\n");
        exit(-1);
    }
    else{
        nThreads = atoi(argv[1]);
        m = atoi(argv[2]);
        n = atoi(argv[3]);
        k = atoi(argv[4]);
    }

    pthread_t threads[nThreads];

    // Initializing matrices
    float alpha = -2;
    float beta = 1;

    float* A = create_random_matrix(m,k);
    float* B = create_random_matrix(k,n);
    float* C = create_random_matrix(m,n);



    for (int t = 0; t < nThreads; t++) {
        int x_min = t * n / nThreads;
        int x_max = x_min + n / nThreads;
        sub_matrix_args *args = create_args(A, B, C, alpha, beta, x_min, x_max, 0, m, n, k);
        pthread_create(&threads[t], NULL, calculate_sub_matrix, (void *) args);
    }

    

    // Printing result
    print_matrix(A, m,k);
    print_matrix(B, k,n);
    print_matrix(C, m,n);

}
