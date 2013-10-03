#include <complex.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

void simd_complex_mult_short(complex float *a, complex float *b, complex float *r);
void simd_complex_mult(complex float *, complex float *, complex float *, complex float *);

void transpose(complex float *M, int k, int n, complex float *R) {
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            R[i * n + j] = M[j * k + i];
        }
    }
}

void gemm(complex float *A,
        complex float *B,
        complex float *C,
        int m,
        int n,
        int k,
        complex float alpha,
        complex float beta) {

    complex float alphas[2],
                  betas[2];

    alphas[0] = alpha;
    alphas[1] = alpha;

    betas[0] = beta;
    betas[1] = beta;

    complex float *a_res = malloc(sizeof(complex float) * 2);
    complex float *b_res = malloc(sizeof(complex float) * 2);

    complex float *T = malloc(sizeof(complex float) * n * k);
    transpose(B, k, n, T);

    for (int x = 0; x < n; x += 2) {
        for (int y = 0; y < m; y++) {
            simd_complex_mult_short((C + y*n + x), betas, (C + y*n + x));

            for (int z = 0; z < k; z++) {
                // simd_complex_mult(alphas, (A + y*k + z), (T + x*n + z * 2), b_res);
                // C[y*n + x] += b_res[0];
                // C[y*n + x + 1] += b_res[1];

                simd_complex_mult(alphas, (A + y*k + z), (B + z*n + x), b_res);
                C[y*n + x] += b_res[0];
                simd_complex_mult(alphas, (A + y*k + z), (B + z*n + x + 1), b_res);
                C[y*n + x + 1] += b_res[0];
            }
        }
    }

    free(a_res);
    free(b_res);
    free(T);
}

void simd_complex_mult_short(complex float *a,
                       complex float *b,
                       complex float *r) {
    __m128 a_reg,
           b_reg,
           t_reg1,
           t_reg2,
           r_reg;

    b_reg = _mm_loadu_ps((float *) a); 
    a_reg = _mm_loadu_ps((float *) b);

    t_reg1 = _mm_moveldup_ps(b_reg);
    t_reg2 = t_reg1 * a_reg;

    a_reg = _mm_shuffle_ps(a_reg, a_reg, 0xb1);
    t_reg1 = _mm_movehdup_ps(b_reg);
    t_reg1 = t_reg1 * a_reg;

    r_reg = _mm_addsub_ps(t_reg2, t_reg1);
    _mm_storeu_ps((float *) r, r_reg);
}

void simd_complex_mult(complex float *a, complex float *b, complex float *c, complex float *r) {
    __m128 a_reg,
           b_reg,
           c_reg,
           t_reg1,
           t_reg2,
           r_reg;

    a_reg = _mm_loadu_ps((float *) b);
    b_reg = _mm_loadu_ps((float *) a); 
    c_reg = _mm_loadu_ps((float *) c);

    t_reg1 = _mm_moveldup_ps(b_reg);
    t_reg2 = t_reg1 * a_reg;

    a_reg = _mm_shuffle_ps(a_reg, a_reg, 0xb1);
    t_reg1 = _mm_movehdup_ps(b_reg);
    t_reg1 = t_reg1 * a_reg;

    r_reg = _mm_addsub_ps(t_reg2, t_reg1);

    t_reg1 = _mm_moveldup_ps(r_reg);
    t_reg2 = t_reg1 * c_reg;

    c_reg = _mm_shuffle_ps(c_reg, c_reg, 0xb1);
    t_reg1 = _mm_movehdup_ps(r_reg);
    t_reg1 = t_reg1 * c_reg;

    r_reg = _mm_addsub_ps(t_reg2, t_reg1);
    _mm_storeu_ps((float *) r, r_reg);
}
