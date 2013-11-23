#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "tables.h"

#define NUM_BLOCKS                  64 * 64
#define THREADS_PER_BLOCK           64
#define NUM_CUBES                   (NUM_BLOCKS * THREADS_PER_BLOCK)

// OGL vertex buffer object
GLuint vbo;
struct cudaGraphicsResource *vbo_resource;

// Size of voxel grid
const int dim_x = 64;
const int dim_y = 64;
const int dim_z = 64;

float sim_time = 0.0;


// CUDA buffers
float *volume;
float4 *vertices;
// float4 *output_vertices;

uint *edge_table;
uint *tri_table;
uint *num_verts_table;

// Get triangles kernel
__global__ void get_triangles(float4 *out, float *volume,
    uint *tri_table,
    uint *num_verts_table) //Some of the tables might be unnecessary
{
    float iso = 0.5;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int x = threadIdx.x;
    int y = blockIdx.x % 64;
    int z = blockIdx.x / 64;
    uint index = 0;
    index =  (uint) (volume[(x  ) + (y  )*64 + (z  )*64*64] < iso);
    index += (uint) (volume[(x+1) + (y  )*64 + (z  )*64*64] < iso) * 2;
    index += (uint) (volume[(x+1) + (y+1)*64 + (z  )*64*64] < iso) * 4;
    index += (uint) (volume[(x  ) + (y+1)*64 + (z  )*64*64] < iso) * 8;
    index += (uint) (volume[(x  ) + (y  )*64 + (z+1)*64*64] < iso) * 16;
    index += (uint) (volume[(x+1) + (y  )*64 + (z+1)*64*64] < iso) * 32;
    index += (uint) (volume[(x+1) + (y+1)*64 + (z+1)*64*64] < iso) * 64;
    index += (uint) (volume[(x  ) + (y+1)*64 + (z+1)*64*64] < iso) * 128;

    // index >= 0 && index < 256
    uint *cube_verts = (tri_table + index * 16);
    int num_verts    = num_verts_table[index];
    int verts_index  = gid * 15;

    for (int i = 0; i < num_verts; i++) {
        uint cur_vert = cube_verts[i];

        float x_len = x / 64.0;
        float y_len = y / 64.0;
        float z_len = z / 64.0;
        float full_len = 1.0 / 64.0;
        float half_len = full_len / 2.0;

        float4 cur_vox;
        switch (cur_vert) {
            case  0: cur_vox = (float4) {x_len+half_len, y_len,          z_len,          1.0}; break;
            case  1: cur_vox = (float4) {x_len+full_len, y_len+half_len, z_len,          1.0}; break;
            case  2: cur_vox = (float4) {x_len+half_len, y_len+full_len, z_len,          1.0}; break;
            case  3: cur_vox = (float4) {x_len,          y_len+half_len, z_len,          1.0}; break;
            case  4: cur_vox = (float4) {x_len+half_len, y_len,          z_len+full_len, 1.0}; break;
            case  5: cur_vox = (float4) {x_len+full_len, y_len+half_len, z_len+full_len, 1.0}; break;
            case  6: cur_vox = (float4) {x_len+half_len, y_len+full_len, z_len+full_len, 1.0}; break;
            case  7: cur_vox = (float4) {x_len,          y_len+half_len, z_len+full_len, 1.0}; break;
            case  8: cur_vox = (float4) {x_len,          y_len,          z_len+half_len, 1.0}; break;
            case  9: cur_vox = (float4) {x_len+full_len, y_len,          z_len+half_len, 1.0}; break;
            case 10: cur_vox = (float4) {x_len+full_len, y_len+full_len, z_len+half_len, 1.0}; break;
            case 11: cur_vox = (float4) {x_len,          y_len+full_len, z_len+half_len, 1.0}; break;
        }

        out[verts_index + i] = cur_vox;
    }

    for (int i = num_verts; i < 15; i++) {
        out[verts_index + i] = (float4) {0.0, 0.0, 0.0, 0.0};
    } 
}

// Set up and call get_triangles kernel
void call_get_triangles() {

    // CUDA taking over vertices buffer from OGL
    size_t num_bytes; 
    cudaGraphicsMapResources(1, &vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer(
        (void **)&vertices, &num_bytes, vbo_resource);

    // Insert call to get_triangles kernel here
    get_triangles<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>
        (vertices, volume, tri_table, num_verts_table);
    // cudaMemcpy(output_vertices, vertices, 
    //     sizeof(float4) * NUM_BLOCKS, cudaMemcpyDeviceToHost);

    // CUDA giving back vertices buffer to OGL
    cudaGraphicsUnmapResources(1, &vbo_resource, 0);
}

// Fill_volume kernel
__global__ void fill_volume(float *volume, float t) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = ((float) threadIdx.x)     / 64; 
    float y = (float) (blockIdx.x % 64) / 64;
    float z = (float) (blockIdx.x / 64) / 64;

    float dx = x - 0.5;
    float dy = y - 0.5;
    float dz = z - 0.5;
    float v1 = sqrt((5 + 3.5 * sin(0.1 * t)) * dx * dx +
        (5 + 2 * sin(t + 3.14)) * dy * dy +
        (5 + 2 * sin(t * 0.5)) * dz * dz);
    float v2 = sqrt(dx * dx) + sqrt(dy * dy) + sqrt(dz * dz);
    float f = abs(cos(0.01 * t));

    volume[gid] = f * v2 + (1 - f) * v1;
}

// Set up and call fill_volume kernel
void call_fill_volume() {
    fill_volume<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>
        (volume, sim_time);
}


// Creating vertex buffer in OpenGL
void init_vertex_buffer() {
    // vbo == vertexBuffer
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, dim_x*dim_y*dim_z*15*4*
        sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&vbo_resource, vbo,
        cudaGraphicsMapFlagsWriteDiscard);
}

// The display function is called at each iteration of the
// OGL main loop. It calls the kernels, and draws the result
void display() {
    sim_time += 0.1;

    // Call kernels
    call_fill_volume();
    call_get_triangles();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //Rotate camera
    glTranslatef(0.5,0.5,0.5);
    glRotatef(2*sim_time, 0.0, 0.0, 1.0);
    glTranslatef(-0.5,-0.5,-0.5);

    //Draw wireframe
    glTranslatef(0.5,0.5,0.5);
    glColor3f(0.0, 0.0, 0.0);
    glutWireCube(1);
    glTranslatef(-0.5,-0.5,-0.5);

    // Render vbo as buffer of points
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(0.7, 0.1, 0.3);
    glDrawArrays(GL_TRIANGLES, 0, dim_x*dim_y*dim_z*15);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();
}

void init_GL(int *argc, char **argv) {

    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(512, 512);
    glutCreateWindow("CUDA Marching Cubes");
    glutDisplayFunc(display);

    glewInit();

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    GLfloat diffuse[] = {1.0,1.0,1.0,1.0};
    GLfloat ambient[] = {0.0,0.0,0.0,1.0};
    GLfloat specular[] = {1.0,1.0,1.0,1.0};
    GLfloat pos[] = {1.0,1.0,0.0,1.0};

    glLightfv(GL_LIGHT0, GL_POSITION, pos);
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, specular);

    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, 512, 512);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, 1, 0.1, 10.0);
    gluLookAt(1.5,1.5,1.5,0.5,0.5,0.5,0,0,1);
}



int main(int argc, char **argv) {
    
    // Setting up OpenGL
    init_GL(&argc, argv);

    // Setting up OpenGL on CUDA device 0
    cudaGLSetGLDevice(0);

    // Creating vertices buffer in OGL/CUDA
    init_vertex_buffer();

    // Allocate memory for volume
    cudaMalloc((void**) &volume, sizeof(float) *  NUM_CUBES);

    // Allocate memory and transfer tables
    // cudaMalloc((void**) &edge_table, sizeof(uint) * 256);
    // cudaMemcpy(edge_table, &edgeTable, 
    //     sizeof(uint) * 256, cudaMemcpyHostToDevice);

    cudaMalloc((void**) &tri_table, sizeof(uint) * 256 * 16);
    cudaMemcpy(tri_table, &triTable, 
        sizeof(uint) * 256 * 16, cudaMemcpyHostToDevice);

    cudaMalloc((void**) &num_verts_table, sizeof(uint) * 256);
    cudaMemcpy(num_verts_table, &numVertsTable, 
        sizeof(uint) * 256, cudaMemcpyHostToDevice);

    glutMainLoop();
}
