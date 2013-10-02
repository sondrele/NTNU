#include <complex.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

void simd_complex_mult(complex float *, complex float *, complex float *);

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

    complex float *a_res = malloc(sizeof(complex float));
    complex float *b_res = malloc(sizeof(complex float));
    complex float *c_res = malloc(sizeof(complex float));

    for (int x = 0; x < n; x++) {
        for (int y = 0; y < m; y++) {
            C[y*n + x] *= beta;
            // simd_complex_mult(betas, (C + y*n + x), c_res);
            // C[y*n + x] = c_res[0];
            // C[y*n + x + 1] = c_res[1];

            for (int z = 0; z < k; z++) {
                simd_complex_mult(alphas, (A + y*k + z), a_res);
                simd_complex_mult(a_res, (B + z*n + x), b_res);
                C[y*n + x] += b_res[0];
                // TODO: z += 2, indekser riktig og putt inn _to_ riktige elementer fra B! - altså ikke x  + width
            }
        }
    }

    free(a_res);
    free(b_res);
    free(c_res);
}

void simd_complex_mult(complex float *a,
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
