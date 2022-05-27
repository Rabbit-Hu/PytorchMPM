#ifndef _VOX_CUH
#define _VOX_CUH

// CUDA function declarations
void avg_voxelize(int b, int c, int n, int r, int r2, int r3, const int *coords,
                  const double *feat, int *ind, int *cnt, double *out);
void avg_voxelize_grad(int b, int c, int n, int s, const int *idx,
                       const int *cnt, const double *grad_y, double *grad_x);


void mpm_point2voxel(
    double *__restrict__ xyz, //n x 3
    double *__restrict__ feature, //n x c 
    int gx, int gy, int gz, // dx, dy, dz
    double dx,
    double *__restrict__ voxel,
    int *__restrict__ batch_index,
    int d,
    int c,
    int dim);

void mpm_point2voxel_grad(
    double *__restrict__ xyz, //n x 3
    double *__restrict__ feature, //n x c 
    int gx, int gy, int gz,
    double dx,
    double *__restrict__ voxel,
    double *__restrict__ voxel_grad,
    double *__restrict__ x_grad,
    double *__restrict__ feature_grad,
    int *__restrict__ batch_index,
    int d,
    int c,
    int dim);
#endif