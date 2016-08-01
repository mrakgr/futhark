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
__kernel void map_kernel_1491(int32_t rows_1201, __global
                              unsigned char *mem_1621, int32_t cols_1202,
                              __global unsigned char *mem_1624, __global
                              unsigned char *mem_1627, __global
                              unsigned char *mem_1630)
{
    const uint kernel_thread_index_1491 = get_global_id(0);
    
    if (kernel_thread_index_1491 >= rows_1201 * cols_1202)
        return;
    
    int32_t i_1492;
    int32_t i_1493;
    
    // compute thread index
    {
        i_1492 = squot32(kernel_thread_index_1491, cols_1202);
        i_1493 = kernel_thread_index_1491 - squot32(kernel_thread_index_1491,
                                                    cols_1202) * cols_1202;
    }
    // read kernel parameters
    { }
    
    int8_t tofloat_arg_1495 = *(__global int8_t *) &mem_1621[i_1492 *
                                                             cols_1202 +
                                                             i_1493];
    float x_1496 = uitofp_i8_f32(tofloat_arg_1495);
    float res_1497 = x_1496 / 255.0F;
    int8_t tofloat_arg_1498 = *(__global int8_t *) &mem_1621[rows_1201 *
                                                             cols_1202 +
                                                             i_1492 *
                                                             cols_1202 +
                                                             i_1493];
    float x_1499 = uitofp_i8_f32(tofloat_arg_1498);
    float res_1500 = x_1499 / 255.0F;
    int8_t tofloat_arg_1501 = *(__global int8_t *) &mem_1621[2 * (rows_1201 *
                                                                  cols_1202) +
                                                             i_1492 *
                                                             cols_1202 +
                                                             i_1493];
    float x_1502 = uitofp_i8_f32(tofloat_arg_1501);
    float res_1503 = x_1502 / 255.0F;
    
    // write kernel result
    {
        *(__global float *) &mem_1624[(i_1492 * cols_1202 + i_1493) * 4] =
            res_1497;
        *(__global float *) &mem_1627[(i_1492 * cols_1202 + i_1493) * 4] =
            res_1500;
        *(__global float *) &mem_1630[(i_1492 * cols_1202 + i_1493) * 4] =
            res_1503;
    }
}
__kernel void map_kernel_1587(int32_t y_1224, int32_t rows_1201, __global
                              unsigned char *mem_1632, __global
                              unsigned char *mem_1634, __global
                              unsigned char *mem_1636)
{
    const uint kernel_thread_index_1587 = get_global_id(0);
    
    if (kernel_thread_index_1587 >= rows_1201)
        return;
    
    int32_t i_1588;
    
    // compute thread index
    {
        i_1588 = kernel_thread_index_1587;
    }
    // read kernel parameters
    { }
    
    char x_1590 = slt32(0, i_1588);
    char y_1591 = slt32(i_1588, y_1224);
    char x_1592 = x_1590 && y_1591;
    int32_t i_1593 = i_1588 - 1;
    int32_t i_1594 = i_1588 + 1;
    
    // write kernel result
    {
        *(__global int32_t *) &mem_1632[i_1588 * 4] = i_1593;
        *(__global int32_t *) &mem_1634[i_1588 * 4] = i_1594;
        *(__global char *) &mem_1636[i_1588] = x_1592;
    }
}
__kernel void map_kernel_1511(__global unsigned char *mem_1636, __global
                              unsigned char *mem_1632, __global
                              unsigned char *gs_mem_1640, int32_t rows_1201,
                              int32_t y_1225, int32_t cols_1202, __global
                              unsigned char *rs_mem_1638, __global
                              unsigned char *mem_1634, __global
                              unsigned char *bs_mem_1642, __global
                              unsigned char *mem_1645, __global
                              unsigned char *mem_1648, __global
                              unsigned char *mem_1651)
{
    const uint kernel_thread_index_1511 = get_global_id(0);
    
    if (kernel_thread_index_1511 >= rows_1201 * cols_1202)
        return;
    
    int32_t i_1512;
    int32_t i_1513;
    int32_t i_1515;
    int32_t i_1516;
    char x_1517;
    
    // compute thread index
    {
        i_1512 = squot32(kernel_thread_index_1511, cols_1202);
        i_1513 = kernel_thread_index_1511 - squot32(kernel_thread_index_1511,
                                                    cols_1202) * cols_1202;
    }
    // read kernel parameters
    {
        i_1515 = *(__global int32_t *) &mem_1632[i_1512 * 4];
        i_1516 = *(__global int32_t *) &mem_1634[i_1512 * 4];
        x_1517 = *(__global char *) &mem_1636[i_1512];
    }
    
    char y_1519 = slt32(0, i_1513);
    char x_1520 = x_1517 && y_1519;
    char y_1521 = slt32(i_1513, y_1225);
    char cond_1522 = x_1520 && y_1521;
    int32_t i_1523 = i_1513 - 1;
    int32_t i_1524 = i_1513 + 1;
    float res_1544;
    
    if (cond_1522) {
        float x_1525 = *(__global float *) &rs_mem_1638[(i_1515 * cols_1202 +
                                                         i_1523) * 4];
        float y_1526 = *(__global float *) &rs_mem_1638[(i_1515 * cols_1202 +
                                                         i_1513) * 4];
        float x_1527 = x_1525 + y_1526;
        float y_1528 = *(__global float *) &rs_mem_1638[(i_1515 * cols_1202 +
                                                         i_1524) * 4];
        float x_1529 = x_1527 + y_1528;
        float y_1530 = *(__global float *) &rs_mem_1638[(i_1512 * cols_1202 +
                                                         i_1523) * 4];
        float x_1531 = x_1529 + y_1530;
        float y_1532 = *(__global float *) &rs_mem_1638[(i_1512 * cols_1202 +
                                                         i_1513) * 4];
        float x_1533 = x_1531 + y_1532;
        float y_1534 = *(__global float *) &rs_mem_1638[(i_1512 * cols_1202 +
                                                         i_1524) * 4];
        float x_1535 = x_1533 + y_1534;
        float y_1536 = *(__global float *) &rs_mem_1638[(i_1516 * cols_1202 +
                                                         i_1523) * 4];
        float x_1537 = x_1535 + y_1536;
        float y_1538 = *(__global float *) &rs_mem_1638[(i_1516 * cols_1202 +
                                                         i_1513) * 4];
        float x_1539 = x_1537 + y_1538;
        float y_1540 = *(__global float *) &rs_mem_1638[(i_1516 * cols_1202 +
                                                         i_1524) * 4];
        float res_1541 = x_1539 + y_1540;
        float res_1542 = res_1541 / 9.0F;
        
        res_1544 = res_1542;
    } else {
        float res_1543 = *(__global float *) &rs_mem_1638[(i_1512 * cols_1202 +
                                                           i_1513) * 4];
        
        res_1544 = res_1543;
    }
    
    float res_1564;
    
    if (cond_1522) {
        float x_1545 = *(__global float *) &gs_mem_1640[(i_1515 * cols_1202 +
                                                         i_1523) * 4];
        float y_1546 = *(__global float *) &gs_mem_1640[(i_1515 * cols_1202 +
                                                         i_1513) * 4];
        float x_1547 = x_1545 + y_1546;
        float y_1548 = *(__global float *) &gs_mem_1640[(i_1515 * cols_1202 +
                                                         i_1524) * 4];
        float x_1549 = x_1547 + y_1548;
        float y_1550 = *(__global float *) &gs_mem_1640[(i_1512 * cols_1202 +
                                                         i_1523) * 4];
        float x_1551 = x_1549 + y_1550;
        float y_1552 = *(__global float *) &gs_mem_1640[(i_1512 * cols_1202 +
                                                         i_1513) * 4];
        float x_1553 = x_1551 + y_1552;
        float y_1554 = *(__global float *) &gs_mem_1640[(i_1512 * cols_1202 +
                                                         i_1524) * 4];
        float x_1555 = x_1553 + y_1554;
        float y_1556 = *(__global float *) &gs_mem_1640[(i_1516 * cols_1202 +
                                                         i_1523) * 4];
        float x_1557 = x_1555 + y_1556;
        float y_1558 = *(__global float *) &gs_mem_1640[(i_1516 * cols_1202 +
                                                         i_1513) * 4];
        float x_1559 = x_1557 + y_1558;
        float y_1560 = *(__global float *) &gs_mem_1640[(i_1516 * cols_1202 +
                                                         i_1524) * 4];
        float res_1561 = x_1559 + y_1560;
        float res_1562 = res_1561 / 9.0F;
        
        res_1564 = res_1562;
    } else {
        float res_1563 = *(__global float *) &gs_mem_1640[(i_1512 * cols_1202 +
                                                           i_1513) * 4];
        
        res_1564 = res_1563;
    }
    
    float res_1584;
    
    if (cond_1522) {
        float x_1565 = *(__global float *) &bs_mem_1642[(i_1515 * cols_1202 +
                                                         i_1523) * 4];
        float y_1566 = *(__global float *) &bs_mem_1642[(i_1515 * cols_1202 +
                                                         i_1513) * 4];
        float x_1567 = x_1565 + y_1566;
        float y_1568 = *(__global float *) &bs_mem_1642[(i_1515 * cols_1202 +
                                                         i_1524) * 4];
        float x_1569 = x_1567 + y_1568;
        float y_1570 = *(__global float *) &bs_mem_1642[(i_1512 * cols_1202 +
                                                         i_1523) * 4];
        float x_1571 = x_1569 + y_1570;
        float y_1572 = *(__global float *) &bs_mem_1642[(i_1512 * cols_1202 +
                                                         i_1513) * 4];
        float x_1573 = x_1571 + y_1572;
        float y_1574 = *(__global float *) &bs_mem_1642[(i_1512 * cols_1202 +
                                                         i_1524) * 4];
        float x_1575 = x_1573 + y_1574;
        float y_1576 = *(__global float *) &bs_mem_1642[(i_1516 * cols_1202 +
                                                         i_1523) * 4];
        float x_1577 = x_1575 + y_1576;
        float y_1578 = *(__global float *) &bs_mem_1642[(i_1516 * cols_1202 +
                                                         i_1513) * 4];
        float x_1579 = x_1577 + y_1578;
        float y_1580 = *(__global float *) &bs_mem_1642[(i_1516 * cols_1202 +
                                                         i_1524) * 4];
        float res_1581 = x_1579 + y_1580;
        float res_1582 = res_1581 / 9.0F;
        
        res_1584 = res_1582;
    } else {
        float res_1583 = *(__global float *) &bs_mem_1642[(i_1512 * cols_1202 +
                                                           i_1513) * 4];
        
        res_1584 = res_1583;
    }
    // write kernel result
    {
        *(__global float *) &mem_1645[(i_1512 * cols_1202 + i_1513) * 4] =
            res_1584;
        *(__global float *) &mem_1648[(i_1512 * cols_1202 + i_1513) * 4] =
            res_1564;
        *(__global float *) &mem_1651[(i_1512 * cols_1202 + i_1513) * 4] =
            res_1544;
    }
}
__kernel void map_kernel_1599(int32_t rows_1201, __global
                              unsigned char *bs_mem_1657,
                              int32_t nesting_size_1489, __global
                              unsigned char *rs_mem_1653, int32_t cols_1202,
                              __global unsigned char *gs_mem_1655, __global
                              unsigned char *mem_1659, __global
                              unsigned char *mem_1663)
{
    const uint kernel_thread_index_1599 = get_global_id(0);
    
    if (kernel_thread_index_1599 >= rows_1201 * cols_1202)
        return;
    
    int32_t i_1600;
    int32_t i_1601;
    float r_1602;
    float g_1603;
    float b_1604;
    
    // compute thread index
    {
        i_1600 = squot32(kernel_thread_index_1599, cols_1202);
        i_1601 = kernel_thread_index_1599 - squot32(kernel_thread_index_1599,
                                                    cols_1202) * cols_1202;
    }
    // read kernel parameters
    {
        r_1602 = *(__global float *) &rs_mem_1653[(i_1600 * cols_1202 +
                                                   i_1601) * 4];
        g_1603 = *(__global float *) &gs_mem_1655[(i_1600 * cols_1202 +
                                                   i_1601) * 4];
        b_1604 = *(__global float *) &bs_mem_1657[(i_1600 * cols_1202 +
                                                   i_1601) * 4];
    }
    
    float trunc_arg_1605 = r_1602 * 255.0F;
    int8_t arr_elem_1606 = fptoui_f32_i8(trunc_arg_1605);
    float trunc_arg_1607 = g_1603 * 255.0F;
    int8_t arr_elem_1608 = fptoui_f32_i8(trunc_arg_1607);
    float trunc_arg_1609 = b_1604 * 255.0F;
    int8_t arr_elem_1610 = fptoui_f32_i8(trunc_arg_1609);
    
    *(__global int8_t *) &mem_1659[kernel_thread_index_1599] = arr_elem_1606;
    *(__global int8_t *) &mem_1659[nesting_size_1489 +
                                   kernel_thread_index_1599] = arr_elem_1608;
    *(__global int8_t *) &mem_1659[2 * nesting_size_1489 +
                                   kernel_thread_index_1599] = arr_elem_1610;
    // write kernel result
    {
        for (int i_1690 = 0; i_1690 < 3; i_1690++) {
            *(__global int8_t *) &mem_1663[3 * (cols_1202 * i_1600) +
                                           (cols_1202 * i_1690 + i_1601)] =
                *(__global int8_t *) &mem_1659[nesting_size_1489 * i_1690 +
                                               kernel_thread_index_1599];
        }
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
class blur:
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

    

    self.fut_kernel_map_transpose_i8_var = program.fut_kernel_map_transpose_i8
    self.map_kernel_1491_var = program.map_kernel_1491
    self.map_kernel_1587_var = program.map_kernel_1587
    self.map_kernel_1511_var = program.map_kernel_1511
    self.map_kernel_1599_var = program.map_kernel_1599
  def futhark_main(self, image_mem_size_1616, image_mem_1617, rows_1201,
                   cols_1202, iterations_1203):
    nesting_size_1489 = (cols_1202 * rows_1201)
    x_1620 = (np.int32(3) * rows_1201)
    bytes_1618 = (x_1620 * cols_1202)
    mem_1621 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1618) if (bytes_1618 > np.int32(0)) else np.int32(1)))
    if ((((np.int32(1) * (np.int32(3) + srem32((np.int32(16) - srem32(np.int32(3),
                                                                      np.int32(16))),
                                               np.int32(16)))) * ((rows_1201 * cols_1202) + srem32((np.int32(16) - srem32((rows_1201 * cols_1202),
                                                                                                                          np.int32(16))),
                                                                                                   np.int32(16)))) * np.int32(1)) != np.int32(0)):
      self.fut_kernel_map_transpose_i8_var.set_args(mem_1621,
                                                    np.int32(np.int32(0)),
                                                    image_mem_1617,
                                                    np.int32(np.int32(0)),
                                                    np.int32(np.int32(3)),
                                                    np.int32((rows_1201 * cols_1202)),
                                                    np.int32(((np.int32(3) * rows_1201) * cols_1202)),
                                                    cl.LocalMemory(long((((np.int32(16) + np.int32(1)) * np.int32(16)) * np.int32(1)))))
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.fut_kernel_map_transpose_i8_var,
                                 (long((np.int32(3) + srem32((np.int32(16) - srem32(np.int32(3),
                                                                                    np.int32(16))),
                                                             np.int32(16)))),
                                  long(((rows_1201 * cols_1202) + srem32((np.int32(16) - srem32((rows_1201 * cols_1202),
                                                                                                np.int32(16))),
                                                                         np.int32(16)))),
                                  long(np.int32(1))), (long(np.int32(16)),
                                                       long(np.int32(16)),
                                                       long(np.int32(1))))
      if synchronous:
        self.queue.finish()
    x_1623 = (np.int32(4) * rows_1201)
    bytes_1622 = (x_1623 * cols_1202)
    mem_1624 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1622) if (bytes_1622 > np.int32(0)) else np.int32(1)))
    mem_1627 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1622) if (bytes_1622 > np.int32(0)) else np.int32(1)))
    mem_1630 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1622) if (bytes_1622 > np.int32(0)) else np.int32(1)))
    group_size_1678 = np.int32(512)
    num_groups_1679 = squot32((((rows_1201 * cols_1202) + group_size_1678) - np.int32(1)),
                              group_size_1678)
    if ((np.int32(1) * (num_groups_1679 * group_size_1678)) != np.int32(0)):
      self.map_kernel_1491_var.set_args(np.int32(rows_1201), mem_1621,
                                        np.int32(cols_1202), mem_1624, mem_1627,
                                        mem_1630)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1491_var,
                                 (long((num_groups_1679 * group_size_1678)),),
                                 (long(group_size_1678),))
      if synchronous:
        self.queue.finish()
    y_1224 = (rows_1201 - np.int32(1))
    y_1225 = (cols_1202 - np.int32(1))
    mem_1632 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(x_1623) if (x_1623 > np.int32(0)) else np.int32(1)))
    mem_1634 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(x_1623) if (x_1623 > np.int32(0)) else np.int32(1)))
    mem_1636 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(rows_1201) if (rows_1201 > np.int32(0)) else np.int32(1)))
    group_size_1680 = np.int32(512)
    num_groups_1681 = squot32(((rows_1201 + group_size_1680) - np.int32(1)),
                              group_size_1680)
    if ((np.int32(1) * (num_groups_1681 * group_size_1680)) != np.int32(0)):
      self.map_kernel_1587_var.set_args(np.int32(y_1224), np.int32(rows_1201),
                                        mem_1632, mem_1634, mem_1636)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1587_var,
                                 (long((num_groups_1681 * group_size_1680)),),
                                 (long(group_size_1680),))
      if synchronous:
        self.queue.finish()
    double_buffer_mem_1669 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                       long(long(bytes_1622) if (bytes_1622 > np.int32(0)) else np.int32(1)))
    double_buffer_mem_1670 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                       long(long(bytes_1622) if (bytes_1622 > np.int32(0)) else np.int32(1)))
    double_buffer_mem_1671 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                       long(long(bytes_1622) if (bytes_1622 > np.int32(0)) else np.int32(1)))
    mem_1645 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1622) if (bytes_1622 > np.int32(0)) else np.int32(1)))
    mem_1648 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1622) if (bytes_1622 > np.int32(0)) else np.int32(1)))
    mem_1651 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1622) if (bytes_1622 > np.int32(0)) else np.int32(1)))
    rs_mem_1638 = mem_1624
    gs_mem_1640 = mem_1627
    bs_mem_1642 = mem_1630
    i_1229 = np.int32(0)
    one_1694 = np.int32(1)
    for counter_1693 in range(iterations_1203):
      group_size_1688 = np.int32(512)
      num_groups_1689 = squot32((((rows_1201 * cols_1202) + group_size_1688) - np.int32(1)),
                                group_size_1688)
      if ((np.int32(1) * (num_groups_1689 * group_size_1688)) != np.int32(0)):
        self.map_kernel_1511_var.set_args(mem_1636, mem_1632, gs_mem_1640,
                                          np.int32(rows_1201), np.int32(y_1225),
                                          np.int32(cols_1202), rs_mem_1638,
                                          mem_1634, bs_mem_1642, mem_1645,
                                          mem_1648, mem_1651)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1511_var,
                                   (long((num_groups_1689 * group_size_1688)),),
                                   (long(group_size_1688),))
        if synchronous:
          self.queue.finish()
      if (((rows_1201 * cols_1202) * np.int32(4)) != np.int32(0)):
        cl.enqueue_copy(self.queue, double_buffer_mem_1669, mem_1651,
                        dest_offset=long(np.int32(0)),
                        src_offset=long(np.int32(0)),
                        byte_count=long(((rows_1201 * cols_1202) * np.int32(4))))
      if synchronous:
        self.queue.finish()
      if (((rows_1201 * cols_1202) * np.int32(4)) != np.int32(0)):
        cl.enqueue_copy(self.queue, double_buffer_mem_1670, mem_1648,
                        dest_offset=long(np.int32(0)),
                        src_offset=long(np.int32(0)),
                        byte_count=long(((rows_1201 * cols_1202) * np.int32(4))))
      if synchronous:
        self.queue.finish()
      if (((rows_1201 * cols_1202) * np.int32(4)) != np.int32(0)):
        cl.enqueue_copy(self.queue, double_buffer_mem_1671, mem_1645,
                        dest_offset=long(np.int32(0)),
                        src_offset=long(np.int32(0)),
                        byte_count=long(((rows_1201 * cols_1202) * np.int32(4))))
      if synchronous:
        self.queue.finish()
      rs_mem_tmp_1682 = double_buffer_mem_1669
      gs_mem_tmp_1683 = double_buffer_mem_1670
      bs_mem_tmp_1684 = double_buffer_mem_1671
      rs_mem_1638 = rs_mem_tmp_1682
      gs_mem_1640 = gs_mem_tmp_1683
      bs_mem_1642 = bs_mem_tmp_1684
      i_1229 += one_1694
    rs_mem_1653 = rs_mem_1638
    gs_mem_1655 = gs_mem_1640
    bs_mem_1657 = bs_mem_1642
    x_1662 = (rows_1201 * np.int32(3))
    bytes_1660 = (x_1662 * cols_1202)
    mem_1663 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1660) if (bytes_1660 > np.int32(0)) else np.int32(1)))
    total_size_1675 = (nesting_size_1489 * np.int32(3))
    mem_1659 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(total_size_1675) if (total_size_1675 > np.int32(0)) else np.int32(1)))
    group_size_1691 = np.int32(512)
    num_groups_1692 = squot32((((rows_1201 * cols_1202) + group_size_1691) - np.int32(1)),
                              group_size_1691)
    if ((np.int32(1) * (num_groups_1692 * group_size_1691)) != np.int32(0)):
      self.map_kernel_1599_var.set_args(np.int32(rows_1201), bs_mem_1657,
                                        np.int32(nesting_size_1489),
                                        rs_mem_1653, np.int32(cols_1202),
                                        gs_mem_1655, mem_1659, mem_1663)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1599_var,
                                 (long((num_groups_1692 * group_size_1691)),),
                                 (long(group_size_1691),))
      if synchronous:
        self.queue.finish()
    x_1666 = (rows_1201 * cols_1202)
    bytes_1664 = (x_1666 * np.int32(3))
    mem_1667 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1664) if (bytes_1664 > np.int32(0)) else np.int32(1)))
    if ((((np.int32(1) * (cols_1202 + srem32((np.int32(16) - srem32(cols_1202,
                                                                    np.int32(16))),
                                             np.int32(16)))) * (np.int32(3) + srem32((np.int32(16) - srem32(np.int32(3),
                                                                                                            np.int32(16))),
                                                                                     np.int32(16)))) * rows_1201) != np.int32(0)):
      self.fut_kernel_map_transpose_i8_var.set_args(mem_1667,
                                                    np.int32(np.int32(0)),
                                                    mem_1663,
                                                    np.int32(np.int32(0)),
                                                    np.int32(cols_1202),
                                                    np.int32(np.int32(3)),
                                                    np.int32(((rows_1201 * cols_1202) * np.int32(3))),
                                                    cl.LocalMemory(long((((np.int32(16) + np.int32(1)) * np.int32(16)) * np.int32(1)))))
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.fut_kernel_map_transpose_i8_var,
                                 (long((cols_1202 + srem32((np.int32(16) - srem32(cols_1202,
                                                                                  np.int32(16))),
                                                           np.int32(16)))),
                                  long((np.int32(3) + srem32((np.int32(16) - srem32(np.int32(3),
                                                                                    np.int32(16))),
                                                             np.int32(16)))),
                                  long(rows_1201)), (long(np.int32(16)),
                                                     long(np.int32(16)),
                                                     long(np.int32(1))))
      if synchronous:
        self.queue.finish()
    out_mem_1676 = mem_1667
    out_memsize_1677 = bytes_1664
    return (out_memsize_1677, out_mem_1676)
  def main(self, iterations_1203_ext, image_mem_1617_ext):
    iterations_1203 = np.int32(iterations_1203_ext)
    rows_1201 = np.int32(image_mem_1617_ext.shape[np.int32(0)])
    cols_1202 = np.int32(image_mem_1617_ext.shape[np.int32(1)])
    assert (np.int32(3) == image_mem_1617_ext.shape[np.int32(2)]), 'shape dimension is incorrect for the constant dimension'
    image_mem_size_1616 = np.int32(image_mem_1617_ext.nbytes)
    if (type(image_mem_1617_ext) == cl.array.Array):
      image_mem_1617 = image_mem_1617_ext.data
    else:
      image_mem_1617 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                 long(long(image_mem_size_1616) if (image_mem_size_1616 > np.int32(0)) else np.int32(1)))
      if (image_mem_size_1616 != np.int32(0)):
        cl.enqueue_copy(self.queue, image_mem_1617, image_mem_1617_ext,
                        is_blocking=synchronous)
    (out_memsize_1677, out_mem_1676) = self.futhark_main(image_mem_size_1616,
                                                         image_mem_1617,
                                                         rows_1201, cols_1202,
                                                         iterations_1203)
    return cl.array.Array(self.queue, (rows_1201, cols_1202, np.int32(3)),
                          ct.c_int8, data=out_mem_1676)