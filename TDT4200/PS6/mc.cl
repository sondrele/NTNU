
__kernel void fill_volume(__global float* volume, __global float *t) {
    int gid = get_global_id(0);
    float x = ((float) get_local_id(0))      / 64; 
    float y = ((float) get_group_id(1)) / 64;
    float z = ((float) get_group_id(2)) / 64;

    float dx = x - 0.5;
    float dy = y - 0.5;
    float dz = z - 0.5;
    float v1 = sqrt((5 + 3.5 * sin(0.1 * t[0])) * dx * dx +
        (5 + 2 * sin(t[0] + 3.14)) * dy * dy +
        (5 + 2 * sin(t[0] * 0.5)) * dz * dz);
    float v2 = sqrt(dx * dx) + sqrt(dy * dy) + sqrt(dz * dz);
    float f = fabs(cos(0.01 * t[0]));

    volume[gid] = f * v2 + (1 - f) * v1;
}


__kernel void get_triangles(__global float* volume,
    __global float4* out, __global uint* tri_table,
    __global uint* num_verts_table) //Some of the tables might be unnecessary
{
    float iso = 0.5;

    int gid = get_global_id(0);
    int x = get_local_id(0);
    int y = get_group_id(1);
    int z = get_group_id(2);
    
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
    // uint *cube_verts = &tri_table[index * 16];
    int num_verts    = num_verts_table[index];
    int verts_index  = gid * 15;

    for (int i = 0; i < num_verts; i++) {
        uint cur_vert = tri_table[index * 16 + i];

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
