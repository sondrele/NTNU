#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "tables.h"

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

uint *edge_table;
uint *tri_table;
uint *num_verts_table;

// Fill_volume kernel
__global__ void fill_volume(float *volume, float t) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    float dx = x - 0.5;
    float dy = y - 0.5;
    float dz = z - 0.5;
    float v1 = sqrt((5+3.5*sin(0.1*t))*dx*dx +
    (5+2*sin(t+3.14))*dy*dy +
    (5+2*sin(t*0.5))*dz*dz);
    float v2 = sqrt(dx*dx) + sqrt(dy*dy) +
    sqrt(dz*dz);
    float f = abs(cos(0.01*t));
    volume[id] = f*v2 + (1-f)*v1;
}

// Get triangles kernel
__global__ void get_triangles(float4 *out, float *volume,
    uint *edge_table,
    uint *tri_table,
    uint *num_verts_table) //Some of the tables might be unnecessary
{

}

// Set up and call get_triangles kernel
void call_get_triangles() {

    // CUDA taking over vertices buffer from OGL
    size_t num_bytes; 
    cudaGraphicsMapResources(1, &vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&vertices, &num_bytes, vbo_resource);

    // Insert call to get_triangles kernel here

    // CUDA giving back vertices buffer to OGL
    cudaGraphicsUnmapResources(1, &vbo_resource, 0);
}

// Set up and call fill_volume kernel
void call_fill_volume() {
}


// Creating vertex buffer in OpenGL
void init_vertex_buffer() {
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, dim_x*dim_y*dim_z*15*4*sizeof(float), 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);	
  cudaGraphicsGLRegisterBuffer(&vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);
}

// The display function is called at each iteration of the
// OGL main loop. It calls the kernels, and draws the result
void display() {
    sim_time+= 0.1;

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
    cudaMalloc((void**) &volume, sizeof(float) * nPoints);
    cudaMalloc((void**) &vertices, sizeof(float4) * nPoints);

    // Allocate memory and transfer tables


    glutMainLoop();
}
