#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <GL/glew.h>

#include <GL/glut.h>
#include <GL/glx.h>

#include "clutil.h"
typedef unsigned int uint;
#include "tables.h"

#define NUM_BLOCKS                  64 * 64
#define THREADS_PER_BLOCK           64
#define NUM_CUBES                   (NUM_BLOCKS * THREADS_PER_BLOCK)

// OGL vertex buffer object
GLuint vbo;

// For OCL error codes
cl_int err;

// Various OCL variables
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;

// OCL kernels
cl_kernel fill_volume_kernel;
cl_kernel get_triangles_kernel;

// OCL Memory buffers
cl_mem volume;
cl_mem vertices;
cl_mem edge_table;
cl_mem tri_table;
cl_mem num_verts_table;
cl_mem sim_time_kernel;

// Size of voxel grid
const int dim_x = 64;
const int dim_y = 64;
const int dim_z = 64;

float sim_time = 0.0;

#define WG_SIZE 1
size_t gws[WG_SIZE] = {64 * 64};

// Set up and call fill_volume kernel
void fill_volume() {
    // Update sim_time for kernel
    err = clEnqueueWriteBuffer(queue, sim_time_kernel, CL_FALSE, 0, 
        sizeof(cl_float), &sim_time, 0, NULL, NULL);

    err = clSetKernelArg(fill_volume_kernel, 0, sizeof(volume), (void*)&volume);
    err = clSetKernelArg(fill_volume_kernel, 1, sizeof(cl_float), (void*)&sim_time_kernel);
    clError("Error setting arguments", err);

    // size_t gws = {64, 64, 1};
    size_t lws = 64;
    clEnqueueNDRangeKernel(queue, fill_volume_kernel, WG_SIZE, NULL, 
        gws, &lws, 0, NULL, NULL);
    clError("Error starting kernel", err);
}

// Set up and call get_triangles kernel
void get_triangles(){

    // OCL taking over vertices buffer from OGL
    glFlush();
    err = clEnqueueAcquireGLObjects(queue, 1, &vertices, 0,0,0);

	//Set up an call kernel here
    err = clSetKernelArg(get_triangles_kernel, 0, sizeof(volume), (void*)&volume);
    err = clSetKernelArg(get_triangles_kernel, 1, sizeof(vertices), (void*)&vertices);
    err = clSetKernelArg(get_triangles_kernel, 2, sizeof(tri_table), (void*)&tri_table);
    err = clSetKernelArg(get_triangles_kernel, 3, sizeof(num_verts_table), (void*)&num_verts_table);
    clError("Error setting arguments", err);
    
    // size_t gws = {64, 64, 1};
    size_t lws = 64;
    err = clEnqueueNDRangeKernel(queue, get_triangles_kernel, WG_SIZE, NULL, 
        gws, &lws, 0, NULL, NULL);
    clError("Error starting kernel", err);
    
    // OCL giving vertices buffer back to OGL
    err = clEnqueueReleaseGLObjects(queue, 1, &vertices, 0,0,0);
    err = clFlush(queue);
}

void init_vertex_buffer(){
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, dim_x*dim_y*dim_z*15*4*sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // Creating OCL buffer from OGL vbo
    vertices = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, vbo, &err);
	clError("Error creating vertices buffer",err);
}


// The display function is called at each iteration of the
// OGL main loop. It calls the kernels, and draws the result
void display(){
    sim_time+= 0.1;

    // Call kernels
    fill_volume();
    get_triangles();

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

    // Render vbo as buffer of vertices 
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(0.7, 0.1, 0.3);
    glDrawArrays(GL_TRIANGLES, 0, dim_x*dim_y*dim_z*15);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();
}

// Set up OpenGL
void init_GL(int *argc, char **argv){

    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(512, 512);
    glutCreateWindow("OpenCL Marching Cubes");
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

    glColorMaterial ( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE ) ;
    glEnable ( GL_COLOR_MATERIAL ) ;

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, 512, 512);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, 1, 0.1, 10.0);
    gluLookAt(1.5,1.5,1.5,0.5,0.5,0.5,0,0,1);
}

// Set up OpenCL
void init_CL(){
    err = clGetPlatformIDs(1, &platform, NULL);
    clError("Couldn't get platform ID", err);
    printPlatformInfo(platform);


    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    clError("Couldn't get device ID", err);
    printDeviceInfo(device);

    cl_context_properties properties[] = {
        CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
        CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        0
    };
    context = clCreateContext(properties, 1, &device, NULL, NULL, &err);
    clError("Couldn't get context", err);

    queue = clCreateCommandQueue(context, device, 0, &err);
    clError("Couldn't create command queue", err);
}



int main(int argc, char** argv) {

    // Setting up OpenGL
    init_GL(&argc, argv);

    // Setting up OpenCL
    init_CL();

    // Creating vertices buffer in OGL
    init_vertex_buffer();
    
    // Build kernels
    fill_volume_kernel = buildKernel("mc.cl", "fill_volume", context, device);
    get_triangles_kernel = buildKernel("mc.cl", "get_triangles", context, device);

    // Allocate memory for volume
    volume = clCreateBuffer(context, CL_MEM_READ_WRITE, 
        sizeof(cl_float) * NUM_CUBES, NULL, &err);

    // Allocate memory and transfer tables
    tri_table = clCreateBuffer(context, CL_MEM_READ_WRITE, 
        sizeof(cl_uint) * 256 * 16, NULL, &err);
    err = clEnqueueWriteBuffer(queue, tri_table, CL_TRUE, 0, 
        sizeof(cl_uint) * 256 * 16, triTable, 0, NULL, NULL);

    num_verts_table = clCreateBuffer(context, CL_MEM_READ_WRITE, 
        sizeof(cl_uint) * 256, NULL, &err);
    err = clEnqueueWriteBuffer(queue, num_verts_table, CL_TRUE, 0, 
        sizeof(cl_uint) * 256, numVertsTable, 0, NULL, NULL);

    sim_time_kernel = clCreateBuffer(context, CL_MEM_READ_WRITE, 
        sizeof(cl_float), NULL, &err);

    glutMainLoop();

    clReleaseMemObject(volume);
    clReleaseMemObject(tri_table);
    clReleaseMemObject(num_verts_table);
    clReleaseKernel(fill_volume_kernel);
    clReleaseKernel(get_triangles_kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
