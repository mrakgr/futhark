import sys
import numpy as np
import ctypes as ct
import pyopencl as cl
import pyopencl.array
import time
import argparse
FUT_BLOCK_DIM = "16"
cl_group_size = np.int32(512)
synchronous = False
fut_opencl_src = """typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long int64_t;
typedef uchar uint8_t;
typedef ushort uint16_t;
typedef uint uint32_t;
typedef ulong uint64_t;
static inline int8_t add8(int8_t x, int8_t y)
{
    return x + y;
}
static inline int16_t add16(int16_t x, int16_t y)
{
    return x + y;
}
static inline int32_t add32(int32_t x, int32_t y)
{
    return x + y;
}
static inline int64_t add64(int64_t x, int64_t y)
{
    return x + y;
}
static inline int8_t sub8(int8_t x, int8_t y)
{
    return x - y;
}
static inline int16_t sub16(int16_t x, int16_t y)
{
    return x - y;
}
static inline int32_t sub32(int32_t x, int32_t y)
{
    return x - y;
}
static inline int64_t sub64(int64_t x, int64_t y)
{
    return x - y;
}
static inline int8_t mul8(int8_t x, int8_t y)
{
    return x * y;
}
static inline int16_t mul16(int16_t x, int16_t y)
{
    return x * y;
}
static inline int32_t mul32(int32_t x, int32_t y)
{
    return x * y;
}
static inline int64_t mul64(int64_t x, int64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline char ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline char ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline char ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline char ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline char ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline char ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline char ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline char ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline char slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline char slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline char slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline char slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline char sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline char sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline char sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline char sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int8_t sext_i8_i8(int8_t x)
{
    return x;
}
static inline int16_t sext_i8_i16(int8_t x)
{
    return x;
}
static inline int32_t sext_i8_i32(int8_t x)
{
    return x;
}
static inline int64_t sext_i8_i64(int8_t x)
{
    return x;
}
static inline int8_t sext_i16_i8(int16_t x)
{
    return x;
}
static inline int16_t sext_i16_i16(int16_t x)
{
    return x;
}
static inline int32_t sext_i16_i32(int16_t x)
{
    return x;
}
static inline int64_t sext_i16_i64(int16_t x)
{
    return x;
}
static inline int8_t sext_i32_i8(int32_t x)
{
    return x;
}
static inline int16_t sext_i32_i16(int32_t x)
{
    return x;
}
static inline int32_t sext_i32_i32(int32_t x)
{
    return x;
}
static inline int64_t sext_i32_i64(int32_t x)
{
    return x;
}
static inline int8_t sext_i64_i8(int64_t x)
{
    return x;
}
static inline int16_t sext_i64_i16(int64_t x)
{
    return x;
}
static inline int32_t sext_i64_i32(int64_t x)
{
    return x;
}
static inline int64_t sext_i64_i64(int64_t x)
{
    return x;
}
static inline uint8_t zext_i8_i8(uint8_t x)
{
    return x;
}
static inline uint16_t zext_i8_i16(uint8_t x)
{
    return x;
}
static inline uint32_t zext_i8_i32(uint8_t x)
{
    return x;
}
static inline uint64_t zext_i8_i64(uint8_t x)
{
    return x;
}
static inline uint8_t zext_i16_i8(uint16_t x)
{
    return x;
}
static inline uint16_t zext_i16_i16(uint16_t x)
{
    return x;
}
static inline uint32_t zext_i16_i32(uint16_t x)
{
    return x;
}
static inline uint64_t zext_i16_i64(uint16_t x)
{
    return x;
}
static inline uint8_t zext_i32_i8(uint32_t x)
{
    return x;
}
static inline uint16_t zext_i32_i16(uint32_t x)
{
    return x;
}
static inline uint32_t zext_i32_i32(uint32_t x)
{
    return x;
}
static inline uint64_t zext_i32_i64(uint32_t x)
{
    return x;
}
static inline uint8_t zext_i64_i8(uint64_t x)
{
    return x;
}
static inline uint16_t zext_i64_i16(uint64_t x)
{
    return x;
}
static inline uint32_t zext_i64_i32(uint64_t x)
{
    return x;
}
static inline uint64_t zext_i64_i64(uint64_t x)
{
    return x;
}
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline char cmplt32(float x, float y)
{
    return x < y;
}
static inline char cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return x;
}
__kernel void map_kernel_1274(int32_t m_659, __global unsigned char *mem_1215)
{
    const uint global_thread_index_1274 = get_global_id(0);
    
    if (global_thread_index_1274 >= m_659)
        return;
    
    int32_t i_1275;
    
    // compute thread index
    {
        i_1275 = global_thread_index_1274;
    }
    // read kernel parameters
    { }
    // write kernel result
    {
        *(__global int32_t *) &mem_1215[i_1275 * 4] = 255;
    }
}
__kernel void map_kernel_1278(int32_t n_658, int32_t m_659, __global
                              unsigned char *mem_1215, __global
                              unsigned char *mem_1218)
{
    const uint global_thread_index_1278 = get_global_id(0);
    
    if (global_thread_index_1278 >= n_658 * m_659)
        return;
    
    int32_t i_1279;
    int32_t j_1280;
    int32_t input_1281;
    
    // compute thread index
    {
        i_1279 = squot32(global_thread_index_1278, m_659);
        j_1280 = global_thread_index_1278 - squot32(global_thread_index_1278,
                                                    m_659) * m_659;
    }
    // read kernel parameters
    {
        input_1281 = *(__global int32_t *) &mem_1215[j_1280 * 4];
    }
    // write kernel result
    {
        *(__global int32_t *) &mem_1218[(i_1279 * m_659 + j_1280) * 4] =
            input_1281;
    }
}
__kernel void map_kernel_1107(__global unsigned char *mem_1224, __global
                              unsigned char *history_mem_1220, int32_t m_680,
                              int32_t nesting_size_1105, __global
                              unsigned char *res_mem_1226, __global
                              unsigned char *mem_1222, int32_t n_679, __global
                              unsigned char *mem_1230)
{
    const uint kernel_thread_index_1107 = get_global_id(0);
    
    if (kernel_thread_index_1107 >= n_679 * m_680)
        return;
    
    int32_t i_1108;
    int32_t i_1109;
    int32_t age_1110;
    
    // compute thread index
    {
        i_1108 = squot32(kernel_thread_index_1107, m_680);
        i_1109 = kernel_thread_index_1107 - squot32(kernel_thread_index_1107,
                                                    m_680) * m_680;
    }
    // read kernel parameters
    {
        age_1110 = *(__global int32_t *) &history_mem_1220[(i_1108 * m_680 +
                                                            i_1109) * 4];
    }
    
    char cond_1111 = age_1110 == 0;
    char cond_1112 = slt32(age_1110, 127);
    int32_t res_1113;
    
    if (cond_1112) {
        res_1113 = age_1110;
    } else {
        res_1113 = 127;
    }
    
    int8_t y_1114 = sext_i32_i8(res_1113);
    int8_t res_1115 = 127 + y_1114;
    
    if (cond_1111) {
        *(__global int8_t *) &mem_1222[kernel_thread_index_1107] = 0;
        *(__global int8_t *) &mem_1222[nesting_size_1105 +
                                       kernel_thread_index_1107] = 0;
        *(__global int8_t *) &mem_1222[2 * nesting_size_1105 +
                                       kernel_thread_index_1107] = 0;
        for (int i_1286 = 0; i_1286 < 3; i_1286++) {
            *(__global int8_t *) &res_mem_1226[nesting_size_1105 * i_1286 +
                                               kernel_thread_index_1107] =
                *(__global int8_t *) &mem_1222[nesting_size_1105 * i_1286 +
                                               kernel_thread_index_1107];
        }
    } else {
        *(__global int8_t *) &mem_1224[kernel_thread_index_1107] = -1;
        *(__global int8_t *) &mem_1224[nesting_size_1105 +
                                       kernel_thread_index_1107] = res_1115;
        *(__global int8_t *) &mem_1224[2 * nesting_size_1105 +
                                       kernel_thread_index_1107] = res_1115;
        for (int i_1287 = 0; i_1287 < 3; i_1287++) {
            *(__global int8_t *) &res_mem_1226[nesting_size_1105 * i_1287 +
                                               kernel_thread_index_1107] =
                *(__global int8_t *) &mem_1224[nesting_size_1105 * i_1287 +
                                               kernel_thread_index_1107];
        }
    }
    // write kernel result
    {
        for (int i_1288 = 0; i_1288 < 3; i_1288++) {
            *(__global int8_t *) &mem_1230[3 * (m_680 * i_1108) + (m_680 *
                                                                   i_1288 +
                                                                   i_1109)] =
                *(__global int8_t *) &res_mem_1226[nesting_size_1105 * i_1288 +
                                                   kernel_thread_index_1107];
        }
    }
}
__kernel void fut_kernel_map_transpose_i8(__global int8_t *odata,
                                          uint odata_offset, __global
                                          int8_t *idata, uint idata_offset,
                                          uint width, uint height,
                                          uint total_size, __local
                                          int8_t *block)
{
    uint x_index;
    uint y_index;
    uint our_array_offset;
    
    // Adjust the input and output arrays with the basic offset.

    odata += odata_offset / sizeof(int8_t);
    idata += idata_offset / sizeof(int8_t);
    // Adjust the input and output arrays for the third dimension.

    our_array_offset = get_global_id(2) * width * height;
    odata += our_array_offset;
    idata += our_array_offset;
    // read the matrix tile into shared memory

    x_index = get_global_id(0);
    y_index = get_global_id(1);
    
    uint index_in = y_index * width + x_index;
    
    if ((x_index < width && y_index < height) && index_in < total_size)
        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =
            idata[index_in];
    barrier(CLK_LOCAL_MEM_FENCE);
    // Write the transposed matrix tile to global memory.

    x_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(0);
    y_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(1);
    
    uint index_out = y_index * height + x_index;
    
    if ((x_index < height && y_index < width) && index_out < total_size)
        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +
                                 get_local_id(1)];
}
__kernel void map_kernel_1168(int32_t n_699, __global unsigned char *mem_1241,
                              __global unsigned char *mem_1243)
{
    const uint kernel_thread_index_1168 = get_global_id(0);
    
    if (kernel_thread_index_1168 >= n_699)
        return;
    
    int32_t i_1169;
    
    // compute thread index
    {
        i_1169 = kernel_thread_index_1168;
    }
    // read kernel parameters
    { }
    
    int32_t x_1171 = i_1169 - 1;
    int32_t res_1172 = smod32(x_1171, n_699);
    int32_t x_1173 = i_1169 + 1;
    int32_t res_1174 = smod32(x_1173, n_699);
    
    // write kernel result
    {
        *(__global int32_t *) &mem_1241[i_1169 * 4] = res_1174;
        *(__global int32_t *) &mem_1243[i_1169 * 4] = res_1172;
    }
}
__kernel void map_kernel_1132(int32_t m_700, __global unsigned char *mem_1241,
                              __global unsigned char *world_mem_1245, __global
                              unsigned char *mem_1243, int32_t n_699, __global
                              unsigned char *mem_1250)
{
    const uint kernel_thread_index_1132 = get_global_id(0);
    
    if (kernel_thread_index_1132 >= n_699 * m_700)
        return;
    
    int32_t i_1133;
    int32_t i_1134;
    int32_t res_1136;
    int32_t res_1137;
    
    // compute thread index
    {
        i_1133 = squot32(kernel_thread_index_1132, m_700);
        i_1134 = kernel_thread_index_1132 - squot32(kernel_thread_index_1132,
                                                    m_700) * m_700;
    }
    // read kernel parameters
    {
        res_1136 = *(__global int32_t *) &mem_1241[i_1133 * 4];
        res_1137 = *(__global int32_t *) &mem_1243[i_1133 * 4];
    }
    
    int32_t x_1139 = i_1134 + 1;
    int32_t res_1140 = smod32(x_1139, m_700);
    int32_t x_1141 = i_1134 - 1;
    int32_t res_1142 = smod32(x_1141, m_700);
    char arg_1143 = *(__global char *) &world_mem_1245[res_1137 * m_700 +
                                                       res_1142];
    int32_t res_1144;
    
    if (arg_1143) {
        res_1144 = 1;
    } else {
        res_1144 = 0;
    }
    
    char arg_1145 = *(__global char *) &world_mem_1245[res_1137 * m_700 +
                                                       i_1134];
    int32_t res_1146;
    
    if (arg_1145) {
        res_1146 = 1;
    } else {
        res_1146 = 0;
    }
    
    int32_t x_1147 = res_1144 + res_1146;
    char arg_1148 = *(__global char *) &world_mem_1245[res_1137 * m_700 +
                                                       res_1140];
    int32_t res_1149;
    
    if (arg_1148) {
        res_1149 = 1;
    } else {
        res_1149 = 0;
    }
    
    int32_t x_1150 = x_1147 + res_1149;
    char arg_1151 = *(__global char *) &world_mem_1245[i_1133 * m_700 +
                                                       res_1142];
    int32_t res_1152;
    
    if (arg_1151) {
        res_1152 = 1;
    } else {
        res_1152 = 0;
    }
    
    int32_t x_1153 = x_1150 + res_1152;
    char arg_1154 = *(__global char *) &world_mem_1245[i_1133 * m_700 +
                                                       res_1140];
    int32_t res_1155;
    
    if (arg_1154) {
        res_1155 = 1;
    } else {
        res_1155 = 0;
    }
    
    int32_t x_1156 = x_1153 + res_1155;
    char arg_1157 = *(__global char *) &world_mem_1245[res_1136 * m_700 +
                                                       res_1142];
    int32_t res_1158;
    
    if (arg_1157) {
        res_1158 = 1;
    } else {
        res_1158 = 0;
    }
    
    int32_t x_1159 = x_1156 + res_1158;
    char arg_1160 = *(__global char *) &world_mem_1245[res_1136 * m_700 +
                                                       i_1134];
    int32_t res_1161;
    
    if (arg_1160) {
        res_1161 = 1;
    } else {
        res_1161 = 0;
    }
    
    int32_t x_1162 = x_1159 + res_1161;
    char arg_1163 = *(__global char *) &world_mem_1245[res_1136 * m_700 +
                                                       res_1140];
    int32_t res_1164;
    
    if (arg_1163) {
        res_1164 = 1;
    } else {
        res_1164 = 0;
    }
    
    int32_t res_1165 = x_1162 + res_1164;
    
    // write kernel result
    {
        *(__global int32_t *) &mem_1250[(i_1133 * m_700 + i_1134) * 4] =
            res_1165;
    }
}
__kernel void map_kernel_1186(int32_t m_700, __global
                              unsigned char *world_mem_1245, __global
                              unsigned char *mem_1250, int32_t n_699, __global
                              unsigned char *mem_1253)
{
    const uint kernel_thread_index_1186 = get_global_id(0);
    
    if (kernel_thread_index_1186 >= n_699 * m_700)
        return;
    
    int32_t i_1187;
    int32_t i_1188;
    int32_t neighbors_1189;
    char alive_1190;
    
    // compute thread index
    {
        i_1187 = squot32(kernel_thread_index_1186, m_700);
        i_1188 = kernel_thread_index_1186 - squot32(kernel_thread_index_1186,
                                                    m_700) * m_700;
    }
    // read kernel parameters
    {
        neighbors_1189 = *(__global int32_t *) &mem_1250[(i_1187 * m_700 +
                                                          i_1188) * 4];
        alive_1190 = *(__global char *) &world_mem_1245[i_1187 * m_700 +
                                                        i_1188];
    }
    
    char cond_1191 = slt32(neighbors_1189, 2);
    char cond_1192 = neighbors_1189 == 3;
    char y_1193 = slt32(neighbors_1189, 4);
    char cond_1194 = alive_1190 && y_1193;
    char x_1195 = !cond_1192;
    char y_1196 = x_1195 && cond_1194;
    char res_1197 = cond_1192 || y_1196;
    char x_1198 = !cond_1191;
    char y_1199 = x_1198 && res_1197;
    
    // write kernel result
    {
        *(__global char *) &mem_1253[i_1187 * m_700 + i_1188] = y_1199;
    }
}
__kernel void map_kernel_1204(int32_t m_700, __global unsigned char *mem_1253,
                              int32_t n_699, __global
                              unsigned char *history_mem_1247, __global
                              unsigned char *mem_1256)
{
    const uint kernel_thread_index_1204 = get_global_id(0);
    
    if (kernel_thread_index_1204 >= n_699 * m_700)
        return;
    
    int32_t i_1205;
    int32_t i_1206;
    int32_t x_1207;
    char alive_1208;
    
    // compute thread index
    {
        i_1205 = squot32(kernel_thread_index_1204, m_700);
        i_1206 = kernel_thread_index_1204 - squot32(kernel_thread_index_1204,
                                                    m_700) * m_700;
    }
    // read kernel parameters
    {
        x_1207 = *(__global int32_t *) &history_mem_1247[(i_1205 * m_700 +
                                                          i_1206) * 4];
        alive_1208 = *(__global char *) &mem_1253[i_1205 * m_700 + i_1206];
    }
    
    int32_t res_1209 = x_1207 + 1;
    int32_t res_1210;
    
    if (alive_1208) {
        res_1210 = 0;
    } else {
        res_1210 = res_1209;
    }
    // write kernel result
    {
        *(__global int32_t *) &mem_1256[(i_1205 * m_700 + i_1206) * 4] =
            res_1210;
    }
}
"""
# Hacky parser/reader for values written in Futhark syntax.  Used for
# reading stdin when compiling standalone programs with the Python
# code generator.

lookahead_buffer = []

def reset_lookahead():
    global lookahead_buffer
    lookahead_buffer = []

def get_char(f):
    global lookahead_buffer
    if len(lookahead_buffer) == 0:
        return f.read(1)
    else:
        c = lookahead_buffer[0]
        lookahead_buffer = lookahead_buffer[1:]
        return c

def unget_char(f, c):
    global lookahead_buffer
    lookahead_buffer = [c] + lookahead_buffer

def peek_char(f):
    c = get_char(f)
    if c:
        unget_char(f, c)
    return c

def skip_spaces(f):
    c = get_char(f)
    while c != None:
        if c.isspace():
            c = get_char(f)
        elif c == '-':
          # May be line comment.
          if peek_char(f) == '-':
            # Yes, line comment. Skip to end of line.
            while (c != '\n' and c != None):
              c = get_char(f)
          else:
            break
        else:
          break
    if c:
        unget_char(f, c)

def parse_specific_char(f, expected):
    got = get_char(f)
    if got != expected:
        unget_char(f, got)
        raise ValueError
    return True

def parse_specific_string(f, s):
    for c in s:
        parse_specific_char(f, c)
    return True

def optional(p, *args):
    try:
        return p(*args)
    except ValueError:
        return None

def sepBy(p, sep, *args):
    elems = []
    x = optional(p, *args)
    if x != None:
        elems += [x]
        while optional(sep, *args) != None:
            x = p(*args)
            elems += [x]
    return elems

def parse_int(f):
    s = ''
    c = get_char(f)
    while c != None:
        if c.isdigit():
            s += c
            c = get_char(f)
        else:
            unget_char(f, c)
            break
    optional(read_int_trailer, f)
    return s

def parse_int_signed(f):
    s = ''
    c = get_char(f)

    if c == '-' and peek_char(f).isdigit():
      s = c + parse_int(f)
    else:
      unget_char(f, c)
      s = parse_int(f)

    return s

def read_int_trailer(f):
  parse_specific_char(f, 'i')
  while peek_char(f).isdigit():
    get_char(f)

def read_comma(f):
    skip_spaces(f)
    parse_specific_char(f, ',')
    return ','

def read_int(f):
    skip_spaces(f)
    return int(parse_int_signed(f))

def read_char(f):
    skip_spaces(f)
    parse_specific_char(f, '\'')
    c = get_char(f)
    parse_specific_char(f, '\'')
    return c

def read_double(f):
    skip_spaces(f)
    c = get_char(f)
    if (c == '-'):
      sign = '-'
    else:
      unget_char(f,c)
      sign = ''
    bef = optional(parse_int, f)
    if bef == None:
        bef = '0'
        parse_specific_char(f, '.')
        aft = parse_int(f)
    elif optional(parse_specific_char, f, '.'):
        aft = parse_int(f)
    else:
        aft = '0'
    if (optional(parse_specific_char, f, 'E') or
        optional(parse_specific_char, f, 'e')):
        expt = parse_int_signed(f)
    else:
        expt = '0'
    optional(read_float_trailer, f)
    return float(sign + bef + '.' + aft + 'E' + expt)

def read_float(f):
    return read_double(f)

def read_float_trailer(f):
  parse_specific_char(f, 'f')
  while peek_char(f).isdigit():
    get_char(f)

def read_bool(f):
    skip_spaces(f)
    if peek_char(f) == 'T':
        parse_specific_string(f, 'True')
        return True
    elif peek_char(f) == 'F':
        parse_specific_string(f, 'False')
        return False
    else:
        raise ValueError

def read_array_elems(f, elem_reader):
    skip_spaces(f)
    parse_specific_char(f, '[')
    xs = sepBy(elem_reader, read_comma, f)
    skip_spaces(f)
    parse_specific_char(f, ']')
    return xs

def read_array_helper(f, elem_reader, rank):
    def nested_row_reader(_):
        return read_array_helper(f, elem_reader, rank-1)
    if rank == 1:
        row_reader = elem_reader
    else:
        row_reader = nested_row_reader
    return read_array_elems(f, row_reader)

def expected_array_dims(l, rank):
  if rank > 1:
      n = len(l)
      if n == 0:
          elem = []
      else:
          elem = l[0]
      return [n] + expected_array_dims(elem, rank-1)
  else:
      return [len(l)]

def verify_array_dims(l, dims):
    if dims[0] != len(l):
        raise ValueError
    if len(dims) > 1:
        for x in l:
            verify_array_dims(x, dims[1:])

def read_double_signed(f):

    skip_spaces(f)
    c = get_char(f)

    if c == '-' and peek_char(f).isdigit():
      v = -1 * read_double(f)
    else:
      unget_char(f, c)
      v = read_double(f)

    return v

def read_array(f, elem_reader, rank, bt):
    elems = read_array_helper(f, elem_reader, rank)
    dims = expected_array_dims(elems, rank)
    verify_array_dims(elems, dims)
    return np.array(elems, dtype=bt)
# Scalar functions.

import numpy as np

def signed(x):
  if type(x) == np.uint8:
    return np.int8(x)
  elif type(x) == np.uint16:
    return np.int16(x)
  elif type(x) == np.uint32:
    return np.int32(x)
  else:
    return np.int64(x)

def unsigned(x):
  if type(x) == np.int8:
    return np.uint8(x)
  elif type(x) == np.int16:
    return np.uint16(x)
  elif type(x) == np.int32:
    return np.uint32(x)
  else:
    return np.uint64(x)

def shlN(x,y):
  return x << y

def ashrN(x,y):
  return x >> y

def sdivN(x,y):
  return x / y

def smodN(x,y):
  return x % y

def udivN(x,y):
  return signed(unsigned(x) / unsigned(y))

def umodN(x,y):
  return signed(unsigned(x) % unsigned(y))

def squotN(x,y):
  return np.int32(float(x) / float(y))

def sremN(x,y):
  return np.fmod(x,y)

def powN(x,y):
  return x ** y

def fpowN(x,y):
  return x ** y

def sleN(x,y):
  return x <= y

def sltN(x,y):
  return x < y

def uleN(x,y):
  return unsigned(x) <= unsigned(y)

def ultN(x,y):
  return unsigned(x) < unsigned(y)

def lshr8(x,y):
  return np.int8(np.uint8(x) >> np.uint8(y))

def lshr16(x,y):
  return np.int16(np.uint16(x) >> np.uint16(y))

def lshr32(x,y):
  return np.int32(np.uint32(x) >> np.uint32(y))

def lshr64(x,y):
  return np.int64(np.uint64(x) >> np.uint64(y))

def sext_T_i8(x):
  return np.int8(x)

def sext_T_i16(x):
  return np.int16(x)

def sext_T_i32(x):
  return np.int32(x)

def sext_T_i64(x):
  return np.int32(x)

def zext_i8_i8(x):
  return np.int8(np.uint8(x))

def zext_i8_i16(x):
  return np.int16(np.uint8(x))

def zext_i8_i32(x):
  return np.int32(np.uint8(x))

def zext_i8_i64(x):
  return np.int64(np.uint8(x))

def zext_i16_i8(x):
  return np.int8(np.uint16(x))

def zext_i16_i16(x):
  return np.int16(np.uint16(x))

def zext_i16_i32(x):
  return np.int32(np.uint16(x))

def zext_i16_i64(x):
  return np.int64(np.uint16(x))

def zext_i32_i8(x):
  return np.int8(np.uint32(x))

def zext_i32_i16(x):
  return np.int16(np.uint32(x))

def zext_i32_i32(x):
  return np.int32(np.uint32(x))

def zext_i32_i64(x):
  return np.int64(np.uint32(x))

def zext_i64_i8(x):
  return np.int8(np.uint64(x))

def zext_i64_i16(x):
  return np.int16(np.uint64(x))

def zext_i64_i32(x):
  return np.int32(np.uint64(x))

def zext_i64_i64(x):
  return np.int64(np.uint64(x))

shl8 = shl16 = shl32 = shl64 = shlN
ashr8 = ashr16 = ashr32 = ashr64 = ashrN
sdiv8 = sdiv16 = sdiv32 = sdiv64 = sdivN
smod8 = smod16 = smod32 = smod64 = smodN
udiv8 = udiv16 = udiv32 = udiv64 = udivN
umod8 = umod16 = umod32 = umod64 = umodN
squot8 = squot16 = squot32 = squot64 = squotN
srem8 = srem16 = srem32 = srem64 = sremN
pow8 = pow16 = pow32 = pow64 = powN
fpow32 = fpow64 = fpowN
sle8 = sle16 = sle32 = sle64 = sleN
slt8 = slt16 = slt32 = slt64 = sltN
ule8 = ule16 = ule32 = ule64 = uleN
ult8 = ult16 = ult32 = ult64 = ultN
sext_i8_i8 = sext_i16_i8 = sext_i32_i8 = sext_i64_i8 = sext_T_i8
sext_i8_i16 = sext_i16_i16 = sext_i32_i16 = sext_i64_i16 = sext_T_i16
sext_i8_i32 = sext_i16_i32 = sext_i32_i32 = sext_i64_i32 = sext_T_i32
sext_i8_i64 = sext_i16_i64 = sext_i32_i64 = sext_i64_i64 = sext_T_i64

def ssignum(x):
  return np.sign(x)

def usignum(x):
  if x < 0:
    return ssignum(-x)
  else:
    return ssignum(x)

def sitofp_T_f32(x):
  return np.float32(x)
sitofp_i8_f32 = sitofp_i16_f32 = sitofp_i32_f32 = sitofp_i64_f32 = sitofp_T_f32

def sitofp_T_f64(x):
  return np.float64(x)
sitofp_i8_f64 = sitofp_i16_f64 = sitofp_i32_f64 = sitofp_i64_f64 = sitofp_T_f64

def uitofp_T_f32(x):
  return np.float32(unsigned(x))
uitofp_i8_f32 = uitofp_i16_f32 = uitofp_i32_f32 = uitofp_i64_f32 = uitofp_T_f32

def uitofp_T_f64(x):
  return np.float64(unsigned(x))
uitofp_i8_f64 = uitofp_i16_f64 = uitofp_i32_f64 = uitofp_i64_f64 = uitofp_T_f64

def fptosi_T_i8(x):
  return np.int8(np.trunc(x))
fptosi_f32_i8 = fptosi_f64_i8 = fptosi_T_i8

def fptosi_T_i16(x):
  return np.int16(np.trunc(x))
fptosi_f32_i16 = fptosi_f64_i16 = fptosi_T_i16

def fptosi_T_i32(x):
  return np.int32(np.trunc(x))
fptosi_f32_i32 = fptosi_f64_i32 = fptosi_T_i32

def fptosi_T_i64(x):
  return np.int64(np.trunc(x))
fptosi_f32_i64 = fptosi_f64_i64 = fptosi_T_i64

def fptoui_T_i8(x):
  return np.uint8(np.trunc(x))
fptoui_f32_i8 = fptoui_f64_i8 = fptoui_T_i8

def fptoui_T_i16(x):
  return np.uint16(np.trunc(x))
fptoui_f32_i16 = fptoui_f64_i16 = fptoui_T_i16

def fptoui_T_i32(x):
  return np.uint32(np.trunc(x))
fptoui_f32_i32 = fptoui_f64_i32 = fptoui_T_i32

def fptoui_T_i64(x):
  return np.uint64(np.trunc(x))
fptoui_f32_i64 = fptoui_f64_i64 = fptoui_T_i64

def fpconv_f32_f64(x):
  return np.float64(x)

def fpconv_f64_f32(x):
  return np.float32(x)

def futhark_log64(x):
  return np.float64(np.log(x))

def futhark_sqrt64(x):
  return np.sqrt(x)

def futhark_exp64(x):
  return np.exp(x)

def futhark_cos64(x):
  return np.cos(x)

def futhark_sin64(x):
  return np.sin(x)

def futhark_atan2_64(x, y):
  return np.arctan2(x, y)

def futhark_isnan64(x):
  return np.isnan(x)

def futhark_isinf64(x):
  return np.isinf(x)

def futhark_log32(x):
  return np.float32(np.log(x))

def futhark_sqrt32(x):
  return np.float32(np.sqrt(x))

def futhark_exp32(x):
  return np.exp(x)

def futhark_cos32(x):
  return np.cos(x)

def futhark_sin32(x):
  return np.sin(x)

def futhark_atan2_32(x, y):
  return np.arctan2(x, y)

def futhark_isnan32(x):
  return np.isnan(x)

def futhark_isinf32(x):
  return np.isinf(x)
class life:
  def __init__(self):
    self.ctx = cl.create_some_context(interactive=False)

    self.queue = cl.CommandQueue(self.ctx)

     # XXX: Assuming just a single device here.

    platform_name = self.ctx.get_info(cl.context_info.DEVICES)[0].platform.name

    device_type = self.ctx.get_info(cl.context_info.DEVICES)[0].type

    lockstep_width = 1

    if ((platform_name == "NVIDIA CUDA") and (device_type == cl.device_type.GPU)):
      lockstep_width = np.int32(32)
    if ((platform_name == "AMD Accelerated Parallel Processing") and (device_type == cl.device_type.GPU)):
      lockstep_width = np.int32(64)

    if (len(fut_opencl_src) >= 0):

      program = cl.Program(self.ctx, fut_opencl_src).build(["-DFUT_BLOCK_DIM={}".format(FUT_BLOCK_DIM), "-DLOCKSTEP_WIDTH={}".format(lockstep_width)])

    

    self.map_kernel_1274_var = program.map_kernel_1274
    self.map_kernel_1278_var = program.map_kernel_1278
    self.map_kernel_1107_var = program.map_kernel_1107
    self.fut_kernel_map_transpose_i8_var = program.fut_kernel_map_transpose_i8
    self.map_kernel_1168_var = program.map_kernel_1168
    self.map_kernel_1132_var = program.map_kernel_1132
    self.map_kernel_1186_var = program.map_kernel_1186
    self.map_kernel_1204_var = program.map_kernel_1204
  def futhark_init(self, world_mem_size_1212, world_mem_1213, n_658, m_659):
    bytes_1214 = (np.int32(4) * m_659)
    mem_1215 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1214) if (bytes_1214 > np.int32(0)) else np.int32(1)))
    group_size_1276 = np.int32(512)
    num_groups_1277 = squot32(((m_659 + group_size_1276) - np.int32(1)),
                              group_size_1276)
    if ((np.int32(1) * (num_groups_1277 * group_size_1276)) != np.int32(0)):
      self.map_kernel_1274_var.set_args(np.int32(m_659), mem_1215)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1274_var,
                                 (long((num_groups_1277 * group_size_1276)),),
                                 (long(group_size_1276),))
      if synchronous:
        self.queue.finish()
    x_1217 = (np.int32(4) * n_658)
    bytes_1216 = (x_1217 * m_659)
    mem_1218 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1216) if (bytes_1216 > np.int32(0)) else np.int32(1)))
    group_size_1282 = np.int32(512)
    num_groups_1283 = squot32((((n_658 * m_659) + group_size_1282) - np.int32(1)),
                              group_size_1282)
    if ((np.int32(1) * (num_groups_1283 * group_size_1282)) != np.int32(0)):
      self.map_kernel_1278_var.set_args(np.int32(n_658), np.int32(m_659),
                                        mem_1215, mem_1218)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1278_var,
                                 (long((num_groups_1283 * group_size_1282)),),
                                 (long(group_size_1282),))
      if synchronous:
        self.queue.finish()
    out_mem_1270 = world_mem_1213
    out_memsize_1271 = world_mem_size_1212
    out_mem_1272 = mem_1218
    out_memsize_1273 = bytes_1216
    return (out_memsize_1271, out_mem_1270, out_memsize_1273, out_mem_1272)
  def futhark_render_frame(self, history_mem_size_1219, history_mem_1220, n_679,
                           m_680):
    nesting_size_1105 = (m_680 * n_679)
    x_1229 = (n_679 * np.int32(3))
    bytes_1227 = (x_1229 * m_680)
    mem_1230 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1227) if (bytes_1227 > np.int32(0)) else np.int32(1)))
    total_size_1267 = (nesting_size_1105 * np.int32(3))
    mem_1224 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(total_size_1267) if (total_size_1267 > np.int32(0)) else np.int32(1)))
    total_size_1268 = (nesting_size_1105 * np.int32(3))
    res_mem_1226 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                             long(long(total_size_1268) if (total_size_1268 > np.int32(0)) else np.int32(1)))
    total_size_1269 = (nesting_size_1105 * np.int32(3))
    mem_1222 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(total_size_1269) if (total_size_1269 > np.int32(0)) else np.int32(1)))
    group_size_1289 = np.int32(512)
    num_groups_1290 = squot32((((n_679 * m_680) + group_size_1289) - np.int32(1)),
                              group_size_1289)
    if ((np.int32(1) * (num_groups_1290 * group_size_1289)) != np.int32(0)):
      self.map_kernel_1107_var.set_args(mem_1224, history_mem_1220,
                                        np.int32(m_680),
                                        np.int32(nesting_size_1105),
                                        res_mem_1226, mem_1222, np.int32(n_679),
                                        mem_1230)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1107_var,
                                 (long((num_groups_1290 * group_size_1289)),),
                                 (long(group_size_1289),))
      if synchronous:
        self.queue.finish()
    x_1233 = (n_679 * m_680)
    bytes_1231 = (x_1233 * np.int32(3))
    mem_1234 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1231) if (bytes_1231 > np.int32(0)) else np.int32(1)))
    if ((((np.int32(1) * (m_680 + srem32((np.int32(16) - srem32(m_680,
                                                                np.int32(16))),
                                         np.int32(16)))) * (np.int32(3) + srem32((np.int32(16) - srem32(np.int32(3),
                                                                                                        np.int32(16))),
                                                                                 np.int32(16)))) * n_679) != np.int32(0)):
      self.fut_kernel_map_transpose_i8_var.set_args(mem_1234,
                                                    np.int32(np.int32(0)),
                                                    mem_1230,
                                                    np.int32(np.int32(0)),
                                                    np.int32(m_680),
                                                    np.int32(np.int32(3)),
                                                    np.int32(((n_679 * m_680) * np.int32(3))),
                                                    cl.LocalMemory(long((((np.int32(16) + np.int32(1)) * np.int32(16)) * np.int32(1)))))
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.fut_kernel_map_transpose_i8_var,
                                 (long((m_680 + srem32((np.int32(16) - srem32(m_680,
                                                                              np.int32(16))),
                                                       np.int32(16)))),
                                  long((np.int32(3) + srem32((np.int32(16) - srem32(np.int32(3),
                                                                                    np.int32(16))),
                                                             np.int32(16)))),
                                  long(n_679)), (long(np.int32(16)),
                                                 long(np.int32(16)),
                                                 long(np.int32(1))))
      if synchronous:
        self.queue.finish()
    out_mem_1284 = mem_1234
    out_memsize_1285 = bytes_1231
    return (out_memsize_1285, out_mem_1284)
  def futhark_main(self):
    scalar_out_1291 = np.int32(2)
    return scalar_out_1291
  def futhark_steps(self, world_mem_size_1236, history_mem_size_1238,
                    world_mem_1237, history_mem_1239, n_699, m_700, steps_703):
    bytes_1240 = (np.int32(4) * n_699)
    mem_1241 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1240) if (bytes_1240 > np.int32(0)) else np.int32(1)))
    mem_1243 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1240) if (bytes_1240 > np.int32(0)) else np.int32(1)))
    group_size_1296 = np.int32(512)
    num_groups_1297 = squot32(((n_699 + group_size_1296) - np.int32(1)),
                              group_size_1296)
    if ((np.int32(1) * (num_groups_1297 * group_size_1296)) != np.int32(0)):
      self.map_kernel_1168_var.set_args(np.int32(n_699), mem_1241, mem_1243)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1168_var,
                                 (long((num_groups_1297 * group_size_1296)),),
                                 (long(group_size_1296),))
      if synchronous:
        self.queue.finish()
    nesting_size_1130 = (m_700 * n_699)
    bytes_1248 = (bytes_1240 * m_700)
    mem_1250 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1248) if (bytes_1248 > np.int32(0)) else np.int32(1)))
    bytes_1251 = (n_699 * m_700)
    double_buffer_mem_1263 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                       long(long(bytes_1251) if (bytes_1251 > np.int32(0)) else np.int32(1)))
    double_buffer_mem_1264 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                       long(long(bytes_1248) if (bytes_1248 > np.int32(0)) else np.int32(1)))
    mem_1253 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1251) if (bytes_1251 > np.int32(0)) else np.int32(1)))
    mem_1256 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1248) if (bytes_1248 > np.int32(0)) else np.int32(1)))
    world_mem_size_1244 = world_mem_size_1236
    history_mem_size_1246 = history_mem_size_1238
    world_mem_1245 = world_mem_1237
    history_mem_1247 = history_mem_1239
    i_706 = np.int32(0)
    one_1311 = np.int32(1)
    for counter_1310 in range(steps_703):
      group_size_1304 = np.int32(512)
      num_groups_1305 = squot32((((n_699 * m_700) + group_size_1304) - np.int32(1)),
                                group_size_1304)
      if ((np.int32(1) * (num_groups_1305 * group_size_1304)) != np.int32(0)):
        self.map_kernel_1132_var.set_args(np.int32(m_700), mem_1241,
                                          world_mem_1245, mem_1243,
                                          np.int32(n_699), mem_1250)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1132_var,
                                   (long((num_groups_1305 * group_size_1304)),),
                                   (long(group_size_1304),))
        if synchronous:
          self.queue.finish()
      group_size_1306 = np.int32(512)
      num_groups_1307 = squot32((((n_699 * m_700) + group_size_1306) - np.int32(1)),
                                group_size_1306)
      if ((np.int32(1) * (num_groups_1307 * group_size_1306)) != np.int32(0)):
        self.map_kernel_1186_var.set_args(np.int32(m_700), world_mem_1245,
                                          mem_1250, np.int32(n_699), mem_1253)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1186_var,
                                   (long((num_groups_1307 * group_size_1306)),),
                                   (long(group_size_1306),))
        if synchronous:
          self.queue.finish()
      group_size_1308 = np.int32(512)
      num_groups_1309 = squot32((((n_699 * m_700) + group_size_1308) - np.int32(1)),
                                group_size_1308)
      if ((np.int32(1) * (num_groups_1309 * group_size_1308)) != np.int32(0)):
        self.map_kernel_1204_var.set_args(np.int32(m_700), mem_1253,
                                          np.int32(n_699), history_mem_1247,
                                          mem_1256)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1204_var,
                                   (long((num_groups_1309 * group_size_1308)),),
                                   (long(group_size_1308),))
        if synchronous:
          self.queue.finish()
      if (((n_699 * m_700) * np.int32(1)) != np.int32(0)):
        cl.enqueue_copy(self.queue, double_buffer_mem_1263, mem_1253,
                        dest_offset=long(np.int32(0)),
                        src_offset=long(np.int32(0)),
                        byte_count=long(((n_699 * m_700) * np.int32(1))))
      if synchronous:
        self.queue.finish()
      if (((n_699 * m_700) * np.int32(4)) != np.int32(0)):
        cl.enqueue_copy(self.queue, double_buffer_mem_1264, mem_1256,
                        dest_offset=long(np.int32(0)),
                        src_offset=long(np.int32(0)),
                        byte_count=long(((n_699 * m_700) * np.int32(4))))
      if synchronous:
        self.queue.finish()
      world_mem_size_tmp_1298 = bytes_1251
      history_mem_size_tmp_1299 = bytes_1248
      world_mem_tmp_1300 = double_buffer_mem_1263
      history_mem_tmp_1301 = double_buffer_mem_1264
      world_mem_size_1244 = world_mem_size_tmp_1298
      history_mem_size_1246 = history_mem_size_tmp_1299
      world_mem_1245 = world_mem_tmp_1300
      history_mem_1247 = history_mem_tmp_1301
      i_706 += one_1311
    world_mem_1258 = world_mem_1245
    world_mem_size_1257 = world_mem_size_1244
    history_mem_1260 = history_mem_1247
    history_mem_size_1259 = history_mem_size_1246
    out_mem_1292 = world_mem_1258
    out_memsize_1293 = world_mem_size_1257
    out_mem_1294 = history_mem_1260
    out_memsize_1295 = history_mem_size_1259
    return (out_memsize_1293, out_mem_1292, out_memsize_1295, out_mem_1294)
  def init(self, world_mem_1213_ext):
    n_658 = np.int32(world_mem_1213_ext.shape[np.int32(0)])
    m_659 = np.int32(world_mem_1213_ext.shape[np.int32(1)])
    world_mem_size_1212 = np.int32(world_mem_1213_ext.nbytes)
    if (type(world_mem_1213_ext) == cl.array.Array):
      world_mem_1213 = world_mem_1213_ext.data
    else:
      world_mem_1213 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                 long(long(world_mem_size_1212) if (world_mem_size_1212 > np.int32(0)) else np.int32(1)))
      if (world_mem_size_1212 != np.int32(0)):
        cl.enqueue_copy(self.queue, world_mem_1213, world_mem_1213_ext,
                        is_blocking=synchronous)
    (out_memsize_1271, out_mem_1270, out_memsize_1273,
     out_mem_1272) = self.futhark_init(world_mem_size_1212, world_mem_1213,
                                       n_658, m_659)
    return (cl.array.Array(self.queue, (n_658, m_659), ct.c_bool,
                           data=out_mem_1270), cl.array.Array(self.queue,
                                                              (n_658, m_659),
                                                              ct.c_int32,
                                                              data=out_mem_1272))
  def render_frame(self, history_mem_1220_ext):
    n_679 = np.int32(history_mem_1220_ext.shape[np.int32(0)])
    m_680 = np.int32(history_mem_1220_ext.shape[np.int32(1)])
    history_mem_size_1219 = np.int32(history_mem_1220_ext.nbytes)
    if (type(history_mem_1220_ext) == cl.array.Array):
      history_mem_1220 = history_mem_1220_ext.data
    else:
      history_mem_1220 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                   long(long(history_mem_size_1219) if (history_mem_size_1219 > np.int32(0)) else np.int32(1)))
      if (history_mem_size_1219 != np.int32(0)):
        cl.enqueue_copy(self.queue, history_mem_1220, history_mem_1220_ext,
                        is_blocking=synchronous)
    (out_memsize_1285,
     out_mem_1284) = self.futhark_render_frame(history_mem_size_1219,
                                               history_mem_1220, n_679, m_680)
    return cl.array.Array(self.queue, (n_679, m_680, np.int32(3)), ct.c_int8,
                          data=out_mem_1284)
  def main(self):
    scalar_out_1291 = self.futhark_main()
    return scalar_out_1291
  def steps(self, world_mem_1237_ext, history_mem_1239_ext, steps_703_ext):
    n_699 = np.int32(world_mem_1237_ext.shape[np.int32(0)])
    m_700 = np.int32(world_mem_1237_ext.shape[np.int32(1)])
    world_mem_size_1236 = np.int32(world_mem_1237_ext.nbytes)
    if (type(world_mem_1237_ext) == cl.array.Array):
      world_mem_1237 = world_mem_1237_ext.data
    else:
      world_mem_1237 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                 long(long(world_mem_size_1236) if (world_mem_size_1236 > np.int32(0)) else np.int32(1)))
      if (world_mem_size_1236 != np.int32(0)):
        cl.enqueue_copy(self.queue, world_mem_1237, world_mem_1237_ext,
                        is_blocking=synchronous)
    n_699 = np.int32(history_mem_1239_ext.shape[np.int32(0)])
    m_700 = np.int32(history_mem_1239_ext.shape[np.int32(1)])
    history_mem_size_1238 = np.int32(history_mem_1239_ext.nbytes)
    if (type(history_mem_1239_ext) == cl.array.Array):
      history_mem_1239 = history_mem_1239_ext.data
    else:
      history_mem_1239 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                   long(long(history_mem_size_1238) if (history_mem_size_1238 > np.int32(0)) else np.int32(1)))
      if (history_mem_size_1238 != np.int32(0)):
        cl.enqueue_copy(self.queue, history_mem_1239, history_mem_1239_ext,
                        is_blocking=synchronous)
    steps_703 = np.int32(steps_703_ext)
    (out_memsize_1293, out_mem_1292, out_memsize_1295,
     out_mem_1294) = self.futhark_steps(world_mem_size_1236,
                                        history_mem_size_1238, world_mem_1237,
                                        history_mem_1239, n_699, m_700,
                                        steps_703)
    return (cl.array.Array(self.queue, (n_699, m_700), ct.c_bool,
                           data=out_mem_1292), cl.array.Array(self.queue,
                                                              (n_699, m_700),
                                                              ct.c_int32,
                                                              data=out_mem_1294))