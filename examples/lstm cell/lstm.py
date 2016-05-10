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
__kernel void map_kernel_5064(int32_t m_4191, __global unsigned char *mem_4977)
{
    const uint global_thread_index_5064 = get_global_id(0);
    
    if (global_thread_index_5064 >= m_4191)
        return;
    
    int32_t i_5065;
    
    // compute thread index
    {
        i_5065 = global_thread_index_5064;
    }
    // read kernel parameters
    { }
    // write kernel result
    {
        *(__global float *) &mem_4977[i_5065 * 4] = 0.0F;
    }
}
__kernel void map_kernel_5068(int32_t n_4193, __global unsigned char *mem_4977,
                              int32_t m_4191, __global unsigned char *mem_4980)
{
    const uint global_thread_index_5068 = get_global_id(0);
    
    if (global_thread_index_5068 >= n_4193 * m_4191)
        return;
    
    int32_t i_5069;
    int32_t j_5070;
    float input_5071;
    
    // compute thread index
    {
        i_5069 = squot32(global_thread_index_5068, m_4191);
        j_5070 = global_thread_index_5068 - squot32(global_thread_index_5068,
                                                    m_4191) * m_4191;
    }
    // read kernel parameters
    {
        input_5071 = *(__global float *) &mem_4977[j_5070 * 4];
    }
    // write kernel result
    {
        *(__global float *) &mem_4980[(i_5069 * m_4191 + j_5070) * 4] =
            input_5071;
    }
}
__kernel void map_kernel_5077(int32_t n_4193, __global unsigned char *mem_4982)
{
    const uint global_thread_index_5077 = get_global_id(0);
    
    if (global_thread_index_5077 >= n_4193)
        return;
    
    int32_t i_5078;
    
    // compute thread index
    {
        i_5078 = global_thread_index_5077;
    }
    // read kernel parameters
    { }
    // write kernel result
    {
        *(__global float *) &mem_4982[i_5078 * 4] = 0.0F;
    }
}
__kernel void map_kernel_5081(int32_t n_4193, __global unsigned char *mem_4982,
                              int32_t m_4191, __global unsigned char *mem_4985)
{
    const uint global_thread_index_5081 = get_global_id(0);
    
    if (global_thread_index_5081 >= m_4191 * n_4193)
        return;
    
    int32_t i_5082;
    int32_t j_5083;
    float input_5084;
    
    // compute thread index
    {
        i_5082 = squot32(global_thread_index_5081, n_4193);
        j_5083 = global_thread_index_5081 - squot32(global_thread_index_5081,
                                                    n_4193) * n_4193;
    }
    // read kernel parameters
    {
        input_5084 = *(__global float *) &mem_4982[j_5083 * 4];
    }
    // write kernel result
    {
        *(__global float *) &mem_4985[(i_5082 * n_4193 + j_5083) * 4] =
            input_5084;
    }
}
__kernel void map_kernel_4631(__global unsigned char *mem_4980, int32_t n_4193,
                              __global unsigned char *mem_4985, int32_t m_4191,
                              __global unsigned char *mem_4988)
{
    const uint kernel_thread_index_4631 = get_global_id(0);
    
    if (kernel_thread_index_4631 >= n_4193 * m_4191)
        return;
    
    int32_t i_4632;
    int32_t i_4633;
    float x_4634;
    float y_4635;
    
    // compute thread index
    {
        i_4632 = squot32(kernel_thread_index_4631, m_4191);
        i_4633 = kernel_thread_index_4631 - squot32(kernel_thread_index_4631,
                                                    m_4191) * m_4191;
    }
    // read kernel parameters
    {
        x_4634 = *(__global float *) &mem_4980[(i_4632 * m_4191 + i_4633) * 4];
        y_4635 = *(__global float *) &mem_4985[(i_4632 * n_4193 + i_4633) * 4];
    }
    
    float res_4636 = x_4634 + y_4635;
    
    // write kernel result
    {
        *(__global float *) &mem_4988[(i_4632 * m_4191 + i_4633) * 4] =
            res_4636;
    }
}
__kernel void map_kernel_4654(__global unsigned char *mem_4988,
                              int32_t size_4253, __global
                              unsigned char *b_bi_mem_4951, int32_t m_4191,
                              __global unsigned char *mem_4991)
{
    const uint kernel_thread_index_4654 = get_global_id(0);
    
    if (kernel_thread_index_4654 >= m_4191 * size_4253)
        return;
    
    int32_t i_4655;
    int32_t i_4656;
    float x_4657;
    float y_4658;
    
    // compute thread index
    {
        i_4655 = squot32(kernel_thread_index_4654, size_4253);
        i_4656 = kernel_thread_index_4654 - squot32(kernel_thread_index_4654,
                                                    size_4253) * size_4253;
    }
    // read kernel parameters
    {
        x_4657 = *(__global float *) &mem_4988[(i_4655 * m_4191 + i_4656) * 4];
        y_4658 = *(__global float *) &b_bi_mem_4951[i_4655 * 4];
    }
    
    float res_4659 = x_4657 + y_4658;
    
    // write kernel result
    {
        *(__global float *) &mem_4991[(i_4655 * size_4253 + i_4656) * 4] =
            res_4659;
    }
}
__kernel void map_kernel_4671(int32_t n_4193, int32_t size_4253, __global
                              unsigned char *mem_4991, int32_t m_4191, __global
                              unsigned char *mem_4994)
{
    const uint kernel_thread_index_4671 = get_global_id(0);
    
    if (kernel_thread_index_4671 >= n_4193 * m_4191)
        return;
    
    int32_t i_4672;
    int32_t i_4673;
    float not_curried_4674;
    
    // compute thread index
    {
        i_4672 = squot32(kernel_thread_index_4671, m_4191);
        i_4673 = kernel_thread_index_4671 - squot32(kernel_thread_index_4671,
                                                    m_4191) * m_4191;
    }
    // read kernel parameters
    {
        not_curried_4674 = *(__global float *) &mem_4991[(i_4672 * size_4253 +
                                                          i_4673) * 4];
    }
    
    float arg_4675 = 0.0F - not_curried_4674;
    float res_4676 = fpow32(2.718280076980591F, arg_4675);
    float y_4677 = 1.0F + res_4676;
    float res_4678 = 1.0F / y_4677;
    
    // write kernel result
    {
        *(__global float *) &mem_4994[(i_4672 * m_4191 + i_4673) * 4] =
            res_4678;
    }
}
__kernel void map_kernel_5096(int32_t n_4193, __global unsigned char *mem_4977,
                              int32_t m_4191, __global unsigned char *mem_4997)
{
    const uint global_thread_index_5096 = get_global_id(0);
    
    if (global_thread_index_5096 >= n_4193 * m_4191)
        return;
    
    int32_t i_5097;
    int32_t j_5098;
    float input_5099;
    
    // compute thread index
    {
        i_5097 = squot32(global_thread_index_5096, m_4191);
        j_5098 = global_thread_index_5096 - squot32(global_thread_index_5096,
                                                    m_4191) * m_4191;
    }
    // read kernel parameters
    {
        input_5099 = *(__global float *) &mem_4977[j_5098 * 4];
    }
    // write kernel result
    {
        *(__global float *) &mem_4997[(i_5097 * m_4191 + j_5098) * 4] =
            input_5099;
    }
}
__kernel void map_kernel_5105(int32_t n_4193, __global unsigned char *mem_4982,
                              int32_t m_4191, __global unsigned char *mem_5000)
{
    const uint global_thread_index_5105 = get_global_id(0);
    
    if (global_thread_index_5105 >= m_4191 * n_4193)
        return;
    
    int32_t i_5106;
    int32_t j_5107;
    float input_5108;
    
    // compute thread index
    {
        i_5106 = squot32(global_thread_index_5105, n_4193);
        j_5107 = global_thread_index_5105 - squot32(global_thread_index_5105,
                                                    n_4193) * n_4193;
    }
    // read kernel parameters
    {
        input_5108 = *(__global float *) &mem_4982[j_5107 * 4];
    }
    // write kernel result
    {
        *(__global float *) &mem_5000[(i_5106 * n_4193 + j_5107) * 4] =
            input_5108;
    }
}
__kernel void map_kernel_4691(__global unsigned char *mem_5000, int32_t n_4193,
                              __global unsigned char *mem_4997, int32_t m_4191,
                              __global unsigned char *mem_5003)
{
    const uint kernel_thread_index_4691 = get_global_id(0);
    
    if (kernel_thread_index_4691 >= n_4193 * m_4191)
        return;
    
    int32_t i_4692;
    int32_t i_4693;
    float x_4694;
    float y_4695;
    
    // compute thread index
    {
        i_4692 = squot32(kernel_thread_index_4691, m_4191);
        i_4693 = kernel_thread_index_4691 - squot32(kernel_thread_index_4691,
                                                    m_4191) * m_4191;
    }
    // read kernel parameters
    {
        x_4694 = *(__global float *) &mem_4997[(i_4692 * m_4191 + i_4693) * 4];
        y_4695 = *(__global float *) &mem_5000[(i_4692 * n_4193 + i_4693) * 4];
    }
    
    float res_4696 = x_4694 + y_4695;
    
    // write kernel result
    {
        *(__global float *) &mem_5003[(i_4692 * m_4191 + i_4693) * 4] =
            res_4696;
    }
}
__kernel void map_kernel_4714(int32_t size_4253, __global
                              unsigned char *b_ig_mem_4957, __global
                              unsigned char *mem_5003, int32_t m_4191, __global
                              unsigned char *mem_5006)
{
    const uint kernel_thread_index_4714 = get_global_id(0);
    
    if (kernel_thread_index_4714 >= m_4191 * size_4253)
        return;
    
    int32_t i_4715;
    int32_t i_4716;
    float x_4717;
    float y_4718;
    
    // compute thread index
    {
        i_4715 = squot32(kernel_thread_index_4714, size_4253);
        i_4716 = kernel_thread_index_4714 - squot32(kernel_thread_index_4714,
                                                    size_4253) * size_4253;
    }
    // read kernel parameters
    {
        x_4717 = *(__global float *) &mem_5003[(i_4715 * m_4191 + i_4716) * 4];
        y_4718 = *(__global float *) &b_ig_mem_4957[i_4715 * 4];
    }
    
    float res_4719 = x_4717 + y_4718;
    
    // write kernel result
    {
        *(__global float *) &mem_5006[(i_4715 * size_4253 + i_4716) * 4] =
            res_4719;
    }
}
__kernel void map_kernel_4731(int32_t n_4193, int32_t size_4253, __global
                              unsigned char *mem_5006, int32_t m_4191, __global
                              unsigned char *mem_5009)
{
    const uint kernel_thread_index_4731 = get_global_id(0);
    
    if (kernel_thread_index_4731 >= n_4193 * m_4191)
        return;
    
    int32_t i_4732;
    int32_t i_4733;
    float not_curried_4734;
    
    // compute thread index
    {
        i_4732 = squot32(kernel_thread_index_4731, m_4191);
        i_4733 = kernel_thread_index_4731 - squot32(kernel_thread_index_4731,
                                                    m_4191) * m_4191;
    }
    // read kernel parameters
    {
        not_curried_4734 = *(__global float *) &mem_5006[(i_4732 * size_4253 +
                                                          i_4733) * 4];
    }
    
    float arg_4735 = 0.0F - not_curried_4734;
    float res_4736 = fpow32(2.718280076980591F, arg_4735);
    float y_4737 = 1.0F + res_4736;
    float res_4738 = 1.0F / y_4737;
    
    // write kernel result
    {
        *(__global float *) &mem_5009[(i_4732 * m_4191 + i_4733) * 4] =
            res_4738;
    }
}
__kernel void map_kernel_5120(int32_t n_4193, __global unsigned char *mem_4977,
                              int32_t m_4191, __global unsigned char *mem_5012)
{
    const uint global_thread_index_5120 = get_global_id(0);
    
    if (global_thread_index_5120 >= n_4193 * m_4191)
        return;
    
    int32_t i_5121;
    int32_t j_5122;
    float input_5123;
    
    // compute thread index
    {
        i_5121 = squot32(global_thread_index_5120, m_4191);
        j_5122 = global_thread_index_5120 - squot32(global_thread_index_5120,
                                                    m_4191) * m_4191;
    }
    // read kernel parameters
    {
        input_5123 = *(__global float *) &mem_4977[j_5122 * 4];
    }
    // write kernel result
    {
        *(__global float *) &mem_5012[(i_5121 * m_4191 + j_5122) * 4] =
            input_5123;
    }
}
__kernel void map_kernel_5129(int32_t n_4193, __global unsigned char *mem_4982,
                              int32_t m_4191, __global unsigned char *mem_5015)
{
    const uint global_thread_index_5129 = get_global_id(0);
    
    if (global_thread_index_5129 >= m_4191 * n_4193)
        return;
    
    int32_t i_5130;
    int32_t j_5131;
    float input_5132;
    
    // compute thread index
    {
        i_5130 = squot32(global_thread_index_5129, n_4193);
        j_5131 = global_thread_index_5129 - squot32(global_thread_index_5129,
                                                    n_4193) * n_4193;
    }
    // read kernel parameters
    {
        input_5132 = *(__global float *) &mem_4982[j_5131 * 4];
    }
    // write kernel result
    {
        *(__global float *) &mem_5015[(i_5130 * n_4193 + j_5131) * 4] =
            input_5132;
    }
}
__kernel void map_kernel_4751(__global unsigned char *mem_5012, int32_t n_4193,
                              __global unsigned char *mem_5015, int32_t m_4191,
                              __global unsigned char *mem_5018)
{
    const uint kernel_thread_index_4751 = get_global_id(0);
    
    if (kernel_thread_index_4751 >= n_4193 * m_4191)
        return;
    
    int32_t i_4752;
    int32_t i_4753;
    float x_4754;
    float y_4755;
    
    // compute thread index
    {
        i_4752 = squot32(kernel_thread_index_4751, m_4191);
        i_4753 = kernel_thread_index_4751 - squot32(kernel_thread_index_4751,
                                                    m_4191) * m_4191;
    }
    // read kernel parameters
    {
        x_4754 = *(__global float *) &mem_5012[(i_4752 * m_4191 + i_4753) * 4];
        y_4755 = *(__global float *) &mem_5015[(i_4752 * n_4193 + i_4753) * 4];
    }
    
    float res_4756 = x_4754 + y_4755;
    
    // write kernel result
    {
        *(__global float *) &mem_5018[(i_4752 * m_4191 + i_4753) * 4] =
            res_4756;
    }
}
__kernel void map_kernel_4774(int32_t size_4253, __global
                              unsigned char *mem_5018, __global
                              unsigned char *b_fg_mem_4963, int32_t m_4191,
                              __global unsigned char *mem_5021)
{
    const uint kernel_thread_index_4774 = get_global_id(0);
    
    if (kernel_thread_index_4774 >= m_4191 * size_4253)
        return;
    
    int32_t i_4775;
    int32_t i_4776;
    float x_4777;
    float y_4778;
    
    // compute thread index
    {
        i_4775 = squot32(kernel_thread_index_4774, size_4253);
        i_4776 = kernel_thread_index_4774 - squot32(kernel_thread_index_4774,
                                                    size_4253) * size_4253;
    }
    // read kernel parameters
    {
        x_4777 = *(__global float *) &mem_5018[(i_4775 * m_4191 + i_4776) * 4];
        y_4778 = *(__global float *) &b_fg_mem_4963[i_4775 * 4];
    }
    
    float res_4779 = x_4777 + y_4778;
    
    // write kernel result
    {
        *(__global float *) &mem_5021[(i_4775 * size_4253 + i_4776) * 4] =
            res_4779;
    }
}
__kernel void map_kernel_4791(int32_t n_4193, __global unsigned char *mem_5021,
                              int32_t size_4253, int32_t m_4191, __global
                              unsigned char *mem_5024)
{
    const uint kernel_thread_index_4791 = get_global_id(0);
    
    if (kernel_thread_index_4791 >= n_4193 * m_4191)
        return;
    
    int32_t i_4792;
    int32_t i_4793;
    float not_curried_4794;
    
    // compute thread index
    {
        i_4792 = squot32(kernel_thread_index_4791, m_4191);
        i_4793 = kernel_thread_index_4791 - squot32(kernel_thread_index_4791,
                                                    m_4191) * m_4191;
    }
    // read kernel parameters
    {
        not_curried_4794 = *(__global float *) &mem_5021[(i_4792 * size_4253 +
                                                          i_4793) * 4];
    }
    
    float arg_4795 = 0.0F - not_curried_4794;
    float res_4796 = fpow32(2.718280076980591F, arg_4795);
    float y_4797 = 1.0F + res_4796;
    float res_4798 = 1.0F / y_4797;
    
    // write kernel result
    {
        *(__global float *) &mem_5024[(i_4792 * m_4191 + i_4793) * 4] =
            res_4798;
    }
}
__kernel void map_kernel_5144(int32_t n_4193, __global unsigned char *mem_4977,
                              int32_t m_4191, __global unsigned char *mem_5027)
{
    const uint global_thread_index_5144 = get_global_id(0);
    
    if (global_thread_index_5144 >= n_4193 * m_4191)
        return;
    
    int32_t i_5145;
    int32_t j_5146;
    float input_5147;
    
    // compute thread index
    {
        i_5145 = squot32(global_thread_index_5144, m_4191);
        j_5146 = global_thread_index_5144 - squot32(global_thread_index_5144,
                                                    m_4191) * m_4191;
    }
    // read kernel parameters
    {
        input_5147 = *(__global float *) &mem_4977[j_5146 * 4];
    }
    // write kernel result
    {
        *(__global float *) &mem_5027[(i_5145 * m_4191 + j_5146) * 4] =
            input_5147;
    }
}
__kernel void map_kernel_5153(int32_t n_4193, __global unsigned char *mem_4982,
                              int32_t m_4191, __global unsigned char *mem_5030)
{
    const uint global_thread_index_5153 = get_global_id(0);
    
    if (global_thread_index_5153 >= m_4191 * n_4193)
        return;
    
    int32_t i_5154;
    int32_t j_5155;
    float input_5156;
    
    // compute thread index
    {
        i_5154 = squot32(global_thread_index_5153, n_4193);
        j_5155 = global_thread_index_5153 - squot32(global_thread_index_5153,
                                                    n_4193) * n_4193;
    }
    // read kernel parameters
    {
        input_5156 = *(__global float *) &mem_4982[j_5155 * 4];
    }
    // write kernel result
    {
        *(__global float *) &mem_5030[(i_5154 * n_4193 + j_5155) * 4] =
            input_5156;
    }
}
__kernel void map_kernel_4811(int32_t n_4193, __global unsigned char *mem_5030,
                              __global unsigned char *mem_5027, int32_t m_4191,
                              __global unsigned char *mem_5033)
{
    const uint kernel_thread_index_4811 = get_global_id(0);
    
    if (kernel_thread_index_4811 >= n_4193 * m_4191)
        return;
    
    int32_t i_4812;
    int32_t i_4813;
    float x_4814;
    float y_4815;
    
    // compute thread index
    {
        i_4812 = squot32(kernel_thread_index_4811, m_4191);
        i_4813 = kernel_thread_index_4811 - squot32(kernel_thread_index_4811,
                                                    m_4191) * m_4191;
    }
    // read kernel parameters
    {
        x_4814 = *(__global float *) &mem_5027[(i_4812 * m_4191 + i_4813) * 4];
        y_4815 = *(__global float *) &mem_5030[(i_4812 * n_4193 + i_4813) * 4];
    }
    
    float res_4816 = x_4814 + y_4815;
    
    // write kernel result
    {
        *(__global float *) &mem_5033[(i_4812 * m_4191 + i_4813) * 4] =
            res_4816;
    }
}
__kernel void map_kernel_4834(__global unsigned char *mem_5033, __global
                              unsigned char *b_og_mem_4969, int32_t size_4253,
                              int32_t m_4191, __global unsigned char *mem_5036)
{
    const uint kernel_thread_index_4834 = get_global_id(0);
    
    if (kernel_thread_index_4834 >= m_4191 * size_4253)
        return;
    
    int32_t i_4835;
    int32_t i_4836;
    float x_4837;
    float y_4838;
    
    // compute thread index
    {
        i_4835 = squot32(kernel_thread_index_4834, size_4253);
        i_4836 = kernel_thread_index_4834 - squot32(kernel_thread_index_4834,
                                                    size_4253) * size_4253;
    }
    // read kernel parameters
    {
        x_4837 = *(__global float *) &mem_5033[(i_4835 * m_4191 + i_4836) * 4];
        y_4838 = *(__global float *) &b_og_mem_4969[i_4835 * 4];
    }
    
    float res_4839 = x_4837 + y_4838;
    
    // write kernel result
    {
        *(__global float *) &mem_5036[(i_4835 * size_4253 + i_4836) * 4] =
            res_4839;
    }
}
__kernel void map_kernel_4851(__global unsigned char *mem_5036, int32_t n_4193,
                              int32_t size_4253, int32_t m_4191, __global
                              unsigned char *mem_5039)
{
    const uint kernel_thread_index_4851 = get_global_id(0);
    
    if (kernel_thread_index_4851 >= n_4193 * m_4191)
        return;
    
    int32_t i_4852;
    int32_t i_4853;
    float not_curried_4854;
    
    // compute thread index
    {
        i_4852 = squot32(kernel_thread_index_4851, m_4191);
        i_4853 = kernel_thread_index_4851 - squot32(kernel_thread_index_4851,
                                                    m_4191) * m_4191;
    }
    // read kernel parameters
    {
        not_curried_4854 = *(__global float *) &mem_5036[(i_4852 * size_4253 +
                                                          i_4853) * 4];
    }
    
    float arg_4855 = 0.0F - not_curried_4854;
    float res_4856 = fpow32(2.718280076980591F, arg_4855);
    float y_4857 = 1.0F + res_4856;
    float res_4858 = 1.0F / y_4857;
    
    // write kernel result
    {
        *(__global float *) &mem_5039[(i_4852 * m_4191 + i_4853) * 4] =
            res_4858;
    }
}
__kernel void map_kernel_4895(__global unsigned char *mem_5024, int32_t n_4193,
                              int32_t size_4253, __global
                              unsigned char *mem_5009, __global
                              unsigned char *mem_4994, __global
                              unsigned char *prev_cell_mem_4975, int32_t m_4191,
                              __global unsigned char *mem_5042)
{
    const uint kernel_thread_index_4895 = get_global_id(0);
    
    if (kernel_thread_index_4895 >= n_4193 * size_4253)
        return;
    
    int32_t i_4896;
    int32_t i_4897;
    float y_4898;
    float x_4899;
    float y_4900;
    float x_4901;
    
    // compute thread index
    {
        i_4896 = squot32(kernel_thread_index_4895, size_4253);
        i_4897 = kernel_thread_index_4895 - squot32(kernel_thread_index_4895,
                                                    size_4253) * size_4253;
    }
    // read kernel parameters
    {
        y_4898 = *(__global float *) &mem_5009[(i_4896 * m_4191 + i_4897) * 4];
        x_4899 = *(__global float *) &prev_cell_mem_4975[(i_4896 * m_4191 +
                                                          i_4897) * 4];
        y_4900 = *(__global float *) &mem_5024[(i_4896 * m_4191 + i_4897) * 4];
        x_4901 = *(__global float *) &mem_4994[(i_4896 * m_4191 + i_4897) * 4];
    }
    
    float res_4902 = x_4901 * y_4898;
    float res_4903 = x_4899 * y_4900;
    float res_4904 = res_4902 + res_4903;
    
    // write kernel result
    {
        *(__global float *) &mem_5042[(i_4896 * size_4253 + i_4897) * 4] =
            res_4904;
    }
}
__kernel void map_kernel_4871(int32_t n_4193, int32_t size_4253, __global
                              unsigned char *mem_5042, __global
                              unsigned char *mem_5039, int32_t m_4191, __global
                              unsigned char *mem_5045)
{
    const uint kernel_thread_index_4871 = get_global_id(0);
    
    if (kernel_thread_index_4871 >= n_4193 * m_4191)
        return;
    
    int32_t i_4872;
    int32_t i_4873;
    float not_curried_4874;
    float x_4875;
    
    // compute thread index
    {
        i_4872 = squot32(kernel_thread_index_4871, m_4191);
        i_4873 = kernel_thread_index_4871 - squot32(kernel_thread_index_4871,
                                                    m_4191) * m_4191;
    }
    // read kernel parameters
    {
        not_curried_4874 = *(__global float *) &mem_5042[(i_4872 * size_4253 +
                                                          i_4873) * 4];
        x_4875 = *(__global float *) &mem_5039[(i_4872 * m_4191 + i_4873) * 4];
    }
    
    float arg_4876 = 0.0F - not_curried_4874;
    float res_4877 = fpow32(2.718280076980591F, arg_4876);
    float y_4878 = 1.0F + res_4877;
    float res_4879 = 1.0F / y_4878;
    float res_4880 = x_4875 * res_4879;
    
    // write kernel result
    {
        *(__global float *) &mem_5045[(i_4872 * m_4191 + i_4873) * 4] =
            res_4880;
    }
}
__kernel void map_kernel_5174(int32_t m_4573, __global unsigned char *mem_5053)
{
    const uint global_thread_index_5174 = get_global_id(0);
    
    if (global_thread_index_5174 >= m_4573)
        return;
    
    int32_t i_5175;
    
    // compute thread index
    {
        i_5175 = global_thread_index_5174;
    }
    // read kernel parameters
    { }
    // write kernel result
    {
        *(__global float *) &mem_5053[i_5175 * 4] = 0.0F;
    }
}
__kernel void map_kernel_5178(__global unsigned char *mem_5053, int32_t m_4573,
                              int32_t n_4575, __global unsigned char *mem_5056)
{
    const uint global_thread_index_5178 = get_global_id(0);
    
    if (global_thread_index_5178 >= n_4575 * m_4573)
        return;
    
    int32_t i_5179;
    int32_t j_5180;
    float input_5181;
    
    // compute thread index
    {
        i_5179 = squot32(global_thread_index_5178, m_4573);
        j_5180 = global_thread_index_5178 - squot32(global_thread_index_5178,
                                                    m_4573) * m_4573;
    }
    // read kernel parameters
    {
        input_5181 = *(__global float *) &mem_5053[j_5180 * 4];
    }
    // write kernel result
    {
        *(__global float *) &mem_5056[(i_5179 * m_4573 + j_5180) * 4] =
            input_5181;
    }
}
__kernel void map_kernel_4940(__global unsigned char *mem_5056, __global
                              unsigned char *b_bi_mem_5049, int32_t m_4573,
                              __global unsigned char *mem_5059)
{
    const uint kernel_thread_index_4940 = get_global_id(0);
    
    if (kernel_thread_index_4940 >= m_4573 * m_4573)
        return;
    
    int32_t i_4941;
    int32_t i_4942;
    float x_4943;
    float y_4944;
    
    // compute thread index
    {
        i_4941 = squot32(kernel_thread_index_4940, m_4573);
        i_4942 = kernel_thread_index_4940 - squot32(kernel_thread_index_4940,
                                                    m_4573) * m_4573;
    }
    // read kernel parameters
    {
        x_4943 = *(__global float *) &mem_5056[(i_4941 * m_4573 + i_4942) * 4];
        y_4944 = *(__global float *) &b_bi_mem_5049[i_4941 * 4];
    }
    
    float res_4945 = x_4943 + y_4944;
    
    // write kernel result
    {
        *(__global float *) &mem_5059[(i_4941 * m_4573 + i_4942) * 4] =
            res_4945;
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
class lstm:
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

    

    self.map_kernel_5064_var = program.map_kernel_5064
    self.map_kernel_5068_var = program.map_kernel_5068
    self.map_kernel_5077_var = program.map_kernel_5077
    self.map_kernel_5081_var = program.map_kernel_5081
    self.map_kernel_4631_var = program.map_kernel_4631
    self.map_kernel_4654_var = program.map_kernel_4654
    self.map_kernel_4671_var = program.map_kernel_4671
    self.map_kernel_5096_var = program.map_kernel_5096
    self.map_kernel_5105_var = program.map_kernel_5105
    self.map_kernel_4691_var = program.map_kernel_4691
    self.map_kernel_4714_var = program.map_kernel_4714
    self.map_kernel_4731_var = program.map_kernel_4731
    self.map_kernel_5120_var = program.map_kernel_5120
    self.map_kernel_5129_var = program.map_kernel_5129
    self.map_kernel_4751_var = program.map_kernel_4751
    self.map_kernel_4774_var = program.map_kernel_4774
    self.map_kernel_4791_var = program.map_kernel_4791
    self.map_kernel_5144_var = program.map_kernel_5144
    self.map_kernel_5153_var = program.map_kernel_5153
    self.map_kernel_4811_var = program.map_kernel_4811
    self.map_kernel_4834_var = program.map_kernel_4834
    self.map_kernel_4851_var = program.map_kernel_4851
    self.map_kernel_4895_var = program.map_kernel_4895
    self.map_kernel_4871_var = program.map_kernel_4871
    self.map_kernel_5174_var = program.map_kernel_5174
    self.map_kernel_5178_var = program.map_kernel_5178
    self.map_kernel_4940_var = program.map_kernel_4940
  def futhark_main(self, W_bi_mem_size_4946, U_bi_mem_size_4948,
                   b_bi_mem_size_4950, W_ig_mem_size_4952, U_ig_mem_size_4954,
                   b_ig_mem_size_4956, W_fg_mem_size_4958, U_fg_mem_size_4960,
                   b_fg_mem_size_4962, W_og_mem_size_4964, U_og_mem_size_4966,
                   b_og_mem_size_4968, input_mem_size_4970,
                   prev_output_mem_size_4972, prev_cell_mem_size_4974,
                   W_bi_mem_4947, U_bi_mem_4949, b_bi_mem_4951, W_ig_mem_4953,
                   U_ig_mem_4955, b_ig_mem_4957, W_fg_mem_4959, U_fg_mem_4961,
                   b_fg_mem_4963, W_og_mem_4965, U_og_mem_4967, b_og_mem_4969,
                   input_mem_4971, prev_output_mem_4973, prev_cell_mem_4975,
                   m_4191, o_4192, n_4193):
    bytes_4976 = (np.int32(4) * m_4191)
    mem_4977 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4976) if (bytes_4976 > np.int32(0)) else np.int32(1)))
    group_size_5066 = np.int32(512)
    num_groups_5067 = squot32(((m_4191 + group_size_5066) - np.int32(1)),
                              group_size_5066)
    if ((np.int32(1) * (num_groups_5067 * group_size_5066)) != np.int32(0)):
      self.map_kernel_5064_var.set_args(np.int32(m_4191), mem_4977)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5064_var,
                                 (long((num_groups_5067 * group_size_5066)),),
                                 (long(group_size_5066),))
      if synchronous:
        self.queue.finish()
    x_4979 = (np.int32(4) * n_4193)
    bytes_4978 = (x_4979 * m_4191)
    mem_4980 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4978) if (bytes_4978 > np.int32(0)) else np.int32(1)))
    group_size_5072 = np.int32(512)
    num_groups_5073 = squot32((((n_4193 * m_4191) + group_size_5072) - np.int32(1)),
                              group_size_5072)
    if ((np.int32(1) * (num_groups_5073 * group_size_5072)) != np.int32(0)):
      self.map_kernel_5068_var.set_args(np.int32(n_4193), mem_4977,
                                        np.int32(m_4191), mem_4980)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5068_var,
                                 (long((num_groups_5073 * group_size_5072)),),
                                 (long(group_size_5072),))
      if synchronous:
        self.queue.finish()
    i_4212 = np.int32(0)
    one_5196 = np.int32(1)
    for counter_5195 in range(m_4191):
      y_4213 = slt32(i_4212, n_4193)
      assert y_4213, 'lstm.fut:12:16-12:16'
      j_4216 = np.int32(0)
      one_5194 = np.int32(1)
      for counter_5193 in range(n_4193):
        res_4217 = np.float32(0.0)
        k_4218 = np.int32(0)
        one_5192 = np.int32(1)
        for counter_5191 in range(o_4192):
          read_res_5189 = np.empty(np.int32(1), dtype=ct.c_float)
          cl.enqueue_copy(self.queue, read_res_5189, W_bi_mem_4947,
                          device_offset=long((((i_4212 * o_4192) + k_4218) * np.int32(4))),
                          is_blocking=True)
          x_4219 = read_res_5189[np.int32(0)]
          read_res_5190 = np.empty(np.int32(1), dtype=ct.c_float)
          cl.enqueue_copy(self.queue, read_res_5190, input_mem_4971,
                          device_offset=long((((k_4218 * n_4193) + j_4216) * np.int32(4))),
                          is_blocking=True)
          y_4220 = read_res_5190[np.int32(0)]
          y_4221 = (x_4219 * y_4220)
          res_4222 = (res_4217 + y_4221)
          res_tmp_5076 = res_4222
          res_4217 = res_tmp_5076
          k_4218 += one_5192
        res_4223 = res_4217
        y_4224 = slt32(j_4216, m_4191)
        assert y_4224, 'lstm.fut:12:16-12:16'
        cl.enqueue_copy(self.queue, mem_4980, np.array(res_4223,
                                                       dtype=ct.c_float),
                        device_offset=long((((i_4212 * m_4191) + j_4216) * np.int32(4))),
                        is_blocking=synchronous)
        j_4216 += one_5194
      i_4212 += one_5196
    mem_4982 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(x_4979) if (x_4979 > np.int32(0)) else np.int32(1)))
    group_size_5079 = np.int32(512)
    num_groups_5080 = squot32(((n_4193 + group_size_5079) - np.int32(1)),
                              group_size_5079)
    if ((np.int32(1) * (num_groups_5080 * group_size_5079)) != np.int32(0)):
      self.map_kernel_5077_var.set_args(np.int32(n_4193), mem_4982)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5077_var,
                                 (long((num_groups_5080 * group_size_5079)),),
                                 (long(group_size_5079),))
      if synchronous:
        self.queue.finish()
    bytes_4983 = (bytes_4976 * n_4193)
    mem_4985 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4983) if (bytes_4983 > np.int32(0)) else np.int32(1)))
    group_size_5085 = np.int32(512)
    num_groups_5086 = squot32((((m_4191 * n_4193) + group_size_5085) - np.int32(1)),
                              group_size_5085)
    if ((np.int32(1) * (num_groups_5086 * group_size_5085)) != np.int32(0)):
      self.map_kernel_5081_var.set_args(np.int32(n_4193), mem_4982,
                                        np.int32(m_4191), mem_4985)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5081_var,
                                 (long((num_groups_5086 * group_size_5085)),),
                                 (long(group_size_5085),))
      if synchronous:
        self.queue.finish()
    i_4232 = np.int32(0)
    one_5204 = np.int32(1)
    for counter_5203 in range(n_4193):
      y_4233 = slt32(i_4232, m_4191)
      assert y_4233, 'lstm.fut:12:16-12:16'
      j_4236 = np.int32(0)
      one_5202 = np.int32(1)
      for counter_5201 in range(m_4191):
        res_4237 = np.float32(0.0)
        k_4238 = np.int32(0)
        one_5200 = np.int32(1)
        for counter_5199 in range(n_4193):
          read_res_5197 = np.empty(np.int32(1), dtype=ct.c_float)
          cl.enqueue_copy(self.queue, read_res_5197, U_bi_mem_4949,
                          device_offset=long((((i_4232 * n_4193) + k_4238) * np.int32(4))),
                          is_blocking=True)
          x_4239 = read_res_5197[np.int32(0)]
          read_res_5198 = np.empty(np.int32(1), dtype=ct.c_float)
          cl.enqueue_copy(self.queue, read_res_5198, prev_output_mem_4973,
                          device_offset=long((((k_4238 * m_4191) + j_4236) * np.int32(4))),
                          is_blocking=True)
          y_4240 = read_res_5198[np.int32(0)]
          y_4241 = (x_4239 * y_4240)
          res_4242 = (res_4237 + y_4241)
          res_tmp_5089 = res_4242
          res_4237 = res_tmp_5089
          k_4238 += one_5200
        res_4243 = res_4237
        y_4244 = slt32(j_4236, n_4193)
        assert y_4244, 'lstm.fut:12:16-12:16'
        cl.enqueue_copy(self.queue, mem_4985, np.array(res_4243,
                                                       dtype=ct.c_float),
                        device_offset=long((((i_4232 * n_4193) + j_4236) * np.int32(4))),
                        is_blocking=synchronous)
        j_4236 += one_5202
      i_4232 += one_5204
    zip_cmp_4249 = (n_4193 == m_4191)
    assert zip_cmp_4249, 'lstm.fut:18:9-18:9'
    cond_4252 = (n_4193 == np.int32(0))
    if cond_4252:
      size_4253 = np.int32(0)
    else:
      size_4253 = m_4191
    zip_cmp_4254 = (m_4191 == n_4193)
    assert zip_cmp_4254, 'lstm.fut:19:33-19:33'
    eq_x_y_4256 = (m_4191 == np.int32(0))
    p_and_eq_x_y_4257 = (cond_4252 and eq_x_y_4256)
    not_p_4258 = not(cond_4252)
    assert_arg_4259 = (p_and_eq_x_y_4257 or not_p_4258)
    assert assert_arg_4259, 'lstm.fut:18:17-18:17'
    nesting_size_4629 = (m_4191 * n_4193)
    mem_4988 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4978) if (bytes_4978 > np.int32(0)) else np.int32(1)))
    group_size_5090 = np.int32(512)
    num_groups_5091 = squot32((((n_4193 * m_4191) + group_size_5090) - np.int32(1)),
                              group_size_5090)
    if ((np.int32(1) * (num_groups_5091 * group_size_5090)) != np.int32(0)):
      self.map_kernel_4631_var.set_args(mem_4980, np.int32(n_4193), mem_4985,
                                        np.int32(m_4191), mem_4988)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_4631_var,
                                 (long((num_groups_5091 * group_size_5090)),),
                                 (long(group_size_5090),))
      if synchronous:
        self.queue.finish()
    eq_x_z_4274 = (np.int32(0) == m_4191)
    p_and_eq_x_y_4275 = (not_p_4258 and eq_x_z_4274)
    eq_x_y_4276 = (cond_4252 or p_and_eq_x_y_4275)
    p_and_eq_x_y_4277 = (eq_x_y_4256 and eq_x_y_4276)
    not_p_4278 = not(eq_x_y_4256)
    assert_arg_4279 = (p_and_eq_x_y_4277 or not_p_4278)
    assert assert_arg_4279, 'lstm.fut:18:17-18:17'
    nesting_size_4652 = (size_4253 * m_4191)
    bytes_4989 = (bytes_4976 * size_4253)
    mem_4991 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4989) if (bytes_4989 > np.int32(0)) else np.int32(1)))
    group_size_5092 = np.int32(512)
    num_groups_5093 = squot32((((m_4191 * size_4253) + group_size_5092) - np.int32(1)),
                              group_size_5092)
    if ((np.int32(1) * (num_groups_5093 * group_size_5092)) != np.int32(0)):
      self.map_kernel_4654_var.set_args(mem_4988, np.int32(size_4253),
                                        b_bi_mem_4951, np.int32(m_4191),
                                        mem_4991)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_4654_var,
                                 (long((num_groups_5093 * group_size_5092)),),
                                 (long(group_size_5092),))
      if synchronous:
        self.queue.finish()
    p_and_eq_x_y_4289 = (not_p_4278 and eq_x_y_4276)
    eq_x_y_4290 = (eq_x_y_4256 or p_and_eq_x_y_4289)
    p_and_eq_x_y_4291 = (eq_x_y_4256 and eq_x_y_4256)
    p_and_eq_x_y_4292 = (not_p_4278 and assert_arg_4259)
    eq_x_z_4293 = (p_and_eq_x_y_4291 or p_and_eq_x_y_4292)
    p_and_eq_x_y_4294 = (cond_4252 and eq_x_y_4290)
    p_and_eq_x_y_4295 = (not_p_4258 and eq_x_z_4293)
    assert_arg_4296 = (p_and_eq_x_y_4294 or p_and_eq_x_y_4295)
    assert assert_arg_4296, 'lstm.fut:31:17-31:17'
    mem_4994 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4978) if (bytes_4978 > np.int32(0)) else np.int32(1)))
    group_size_5094 = np.int32(512)
    num_groups_5095 = squot32((((n_4193 * m_4191) + group_size_5094) - np.int32(1)),
                              group_size_5094)
    if ((np.int32(1) * (num_groups_5095 * group_size_5094)) != np.int32(0)):
      self.map_kernel_4671_var.set_args(np.int32(n_4193), np.int32(size_4253),
                                        mem_4991, np.int32(m_4191), mem_4994)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_4671_var,
                                 (long((num_groups_5095 * group_size_5094)),),
                                 (long(group_size_5094),))
      if synchronous:
        self.queue.finish()
    mem_4997 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4978) if (bytes_4978 > np.int32(0)) else np.int32(1)))
    group_size_5100 = np.int32(512)
    num_groups_5101 = squot32((((n_4193 * m_4191) + group_size_5100) - np.int32(1)),
                              group_size_5100)
    if ((np.int32(1) * (num_groups_5101 * group_size_5100)) != np.int32(0)):
      self.map_kernel_5096_var.set_args(np.int32(n_4193), mem_4977,
                                        np.int32(m_4191), mem_4997)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5096_var,
                                 (long((num_groups_5101 * group_size_5100)),),
                                 (long(group_size_5100),))
      if synchronous:
        self.queue.finish()
    i_4310 = np.int32(0)
    one_5212 = np.int32(1)
    for counter_5211 in range(m_4191):
      y_4311 = slt32(i_4310, n_4193)
      assert y_4311, 'lstm.fut:12:16-12:16'
      j_4314 = np.int32(0)
      one_5210 = np.int32(1)
      for counter_5209 in range(n_4193):
        res_4315 = np.float32(0.0)
        k_4316 = np.int32(0)
        one_5208 = np.int32(1)
        for counter_5207 in range(o_4192):
          read_res_5205 = np.empty(np.int32(1), dtype=ct.c_float)
          cl.enqueue_copy(self.queue, read_res_5205, W_ig_mem_4953,
                          device_offset=long((((i_4310 * o_4192) + k_4316) * np.int32(4))),
                          is_blocking=True)
          x_4317 = read_res_5205[np.int32(0)]
          read_res_5206 = np.empty(np.int32(1), dtype=ct.c_float)
          cl.enqueue_copy(self.queue, read_res_5206, input_mem_4971,
                          device_offset=long((((k_4316 * n_4193) + j_4314) * np.int32(4))),
                          is_blocking=True)
          y_4318 = read_res_5206[np.int32(0)]
          y_4319 = (x_4317 * y_4318)
          res_4320 = (res_4315 + y_4319)
          res_tmp_5104 = res_4320
          res_4315 = res_tmp_5104
          k_4316 += one_5208
        res_4321 = res_4315
        y_4322 = slt32(j_4314, m_4191)
        assert y_4322, 'lstm.fut:12:16-12:16'
        cl.enqueue_copy(self.queue, mem_4997, np.array(res_4321,
                                                       dtype=ct.c_float),
                        device_offset=long((((i_4310 * m_4191) + j_4314) * np.int32(4))),
                        is_blocking=synchronous)
        j_4314 += one_5210
      i_4310 += one_5212
    mem_5000 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4983) if (bytes_4983 > np.int32(0)) else np.int32(1)))
    group_size_5109 = np.int32(512)
    num_groups_5110 = squot32((((m_4191 * n_4193) + group_size_5109) - np.int32(1)),
                              group_size_5109)
    if ((np.int32(1) * (num_groups_5110 * group_size_5109)) != np.int32(0)):
      self.map_kernel_5105_var.set_args(np.int32(n_4193), mem_4982,
                                        np.int32(m_4191), mem_5000)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5105_var,
                                 (long((num_groups_5110 * group_size_5109)),),
                                 (long(group_size_5109),))
      if synchronous:
        self.queue.finish()
    i_4329 = np.int32(0)
    one_5220 = np.int32(1)
    for counter_5219 in range(n_4193):
      y_4330 = slt32(i_4329, m_4191)
      assert y_4330, 'lstm.fut:12:16-12:16'
      j_4333 = np.int32(0)
      one_5218 = np.int32(1)
      for counter_5217 in range(m_4191):
        res_4334 = np.float32(0.0)
        k_4335 = np.int32(0)
        one_5216 = np.int32(1)
        for counter_5215 in range(n_4193):
          read_res_5213 = np.empty(np.int32(1), dtype=ct.c_float)
          cl.enqueue_copy(self.queue, read_res_5213, U_ig_mem_4955,
                          device_offset=long((((i_4329 * n_4193) + k_4335) * np.int32(4))),
                          is_blocking=True)
          x_4336 = read_res_5213[np.int32(0)]
          read_res_5214 = np.empty(np.int32(1), dtype=ct.c_float)
          cl.enqueue_copy(self.queue, read_res_5214, prev_output_mem_4973,
                          device_offset=long((((k_4335 * m_4191) + j_4333) * np.int32(4))),
                          is_blocking=True)
          y_4337 = read_res_5214[np.int32(0)]
          y_4338 = (x_4336 * y_4337)
          res_4339 = (res_4334 + y_4338)
          res_tmp_5113 = res_4339
          res_4334 = res_tmp_5113
          k_4335 += one_5216
        res_4340 = res_4334
        y_4341 = slt32(j_4333, n_4193)
        assert y_4341, 'lstm.fut:12:16-12:16'
        cl.enqueue_copy(self.queue, mem_5000, np.array(res_4340,
                                                       dtype=ct.c_float),
                        device_offset=long((((i_4329 * n_4193) + j_4333) * np.int32(4))),
                        is_blocking=synchronous)
        j_4333 += one_5218
      i_4329 += one_5220
    mem_5003 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4978) if (bytes_4978 > np.int32(0)) else np.int32(1)))
    group_size_5114 = np.int32(512)
    num_groups_5115 = squot32((((n_4193 * m_4191) + group_size_5114) - np.int32(1)),
                              group_size_5114)
    if ((np.int32(1) * (num_groups_5115 * group_size_5114)) != np.int32(0)):
      self.map_kernel_4691_var.set_args(mem_5000, np.int32(n_4193), mem_4997,
                                        np.int32(m_4191), mem_5003)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_4691_var,
                                 (long((num_groups_5115 * group_size_5114)),),
                                 (long(group_size_5114),))
      if synchronous:
        self.queue.finish()
    mem_5006 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4989) if (bytes_4989 > np.int32(0)) else np.int32(1)))
    group_size_5116 = np.int32(512)
    num_groups_5117 = squot32((((m_4191 * size_4253) + group_size_5116) - np.int32(1)),
                              group_size_5116)
    if ((np.int32(1) * (num_groups_5117 * group_size_5116)) != np.int32(0)):
      self.map_kernel_4714_var.set_args(np.int32(size_4253), b_ig_mem_4957,
                                        mem_5003, np.int32(m_4191), mem_5006)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_4714_var,
                                 (long((num_groups_5117 * group_size_5116)),),
                                 (long(group_size_5116),))
      if synchronous:
        self.queue.finish()
    mem_5009 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4978) if (bytes_4978 > np.int32(0)) else np.int32(1)))
    group_size_5118 = np.int32(512)
    num_groups_5119 = squot32((((n_4193 * m_4191) + group_size_5118) - np.int32(1)),
                              group_size_5118)
    if ((np.int32(1) * (num_groups_5119 * group_size_5118)) != np.int32(0)):
      self.map_kernel_4731_var.set_args(np.int32(n_4193), np.int32(size_4253),
                                        mem_5006, np.int32(m_4191), mem_5009)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_4731_var,
                                 (long((num_groups_5119 * group_size_5118)),),
                                 (long(group_size_5118),))
      if synchronous:
        self.queue.finish()
    mem_5012 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4978) if (bytes_4978 > np.int32(0)) else np.int32(1)))
    group_size_5124 = np.int32(512)
    num_groups_5125 = squot32((((n_4193 * m_4191) + group_size_5124) - np.int32(1)),
                              group_size_5124)
    if ((np.int32(1) * (num_groups_5125 * group_size_5124)) != np.int32(0)):
      self.map_kernel_5120_var.set_args(np.int32(n_4193), mem_4977,
                                        np.int32(m_4191), mem_5012)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5120_var,
                                 (long((num_groups_5125 * group_size_5124)),),
                                 (long(group_size_5124),))
      if synchronous:
        self.queue.finish()
    i_4379 = np.int32(0)
    one_5228 = np.int32(1)
    for counter_5227 in range(m_4191):
      y_4380 = slt32(i_4379, n_4193)
      assert y_4380, 'lstm.fut:12:16-12:16'
      j_4383 = np.int32(0)
      one_5226 = np.int32(1)
      for counter_5225 in range(n_4193):
        res_4384 = np.float32(0.0)
        k_4385 = np.int32(0)
        one_5224 = np.int32(1)
        for counter_5223 in range(o_4192):
          read_res_5221 = np.empty(np.int32(1), dtype=ct.c_float)
          cl.enqueue_copy(self.queue, read_res_5221, W_fg_mem_4959,
                          device_offset=long((((i_4379 * o_4192) + k_4385) * np.int32(4))),
                          is_blocking=True)
          x_4386 = read_res_5221[np.int32(0)]
          read_res_5222 = np.empty(np.int32(1), dtype=ct.c_float)
          cl.enqueue_copy(self.queue, read_res_5222, input_mem_4971,
                          device_offset=long((((k_4385 * n_4193) + j_4383) * np.int32(4))),
                          is_blocking=True)
          y_4387 = read_res_5222[np.int32(0)]
          y_4388 = (x_4386 * y_4387)
          res_4389 = (res_4384 + y_4388)
          res_tmp_5128 = res_4389
          res_4384 = res_tmp_5128
          k_4385 += one_5224
        res_4390 = res_4384
        y_4391 = slt32(j_4383, m_4191)
        assert y_4391, 'lstm.fut:12:16-12:16'
        cl.enqueue_copy(self.queue, mem_5012, np.array(res_4390,
                                                       dtype=ct.c_float),
                        device_offset=long((((i_4379 * m_4191) + j_4383) * np.int32(4))),
                        is_blocking=synchronous)
        j_4383 += one_5226
      i_4379 += one_5228
    mem_5015 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4983) if (bytes_4983 > np.int32(0)) else np.int32(1)))
    group_size_5133 = np.int32(512)
    num_groups_5134 = squot32((((m_4191 * n_4193) + group_size_5133) - np.int32(1)),
                              group_size_5133)
    if ((np.int32(1) * (num_groups_5134 * group_size_5133)) != np.int32(0)):
      self.map_kernel_5129_var.set_args(np.int32(n_4193), mem_4982,
                                        np.int32(m_4191), mem_5015)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5129_var,
                                 (long((num_groups_5134 * group_size_5133)),),
                                 (long(group_size_5133),))
      if synchronous:
        self.queue.finish()
    i_4398 = np.int32(0)
    one_5236 = np.int32(1)
    for counter_5235 in range(n_4193):
      y_4399 = slt32(i_4398, m_4191)
      assert y_4399, 'lstm.fut:12:16-12:16'
      j_4402 = np.int32(0)
      one_5234 = np.int32(1)
      for counter_5233 in range(m_4191):
        res_4403 = np.float32(0.0)
        k_4404 = np.int32(0)
        one_5232 = np.int32(1)
        for counter_5231 in range(n_4193):
          read_res_5229 = np.empty(np.int32(1), dtype=ct.c_float)
          cl.enqueue_copy(self.queue, read_res_5229, U_fg_mem_4961,
                          device_offset=long((((i_4398 * n_4193) + k_4404) * np.int32(4))),
                          is_blocking=True)
          x_4405 = read_res_5229[np.int32(0)]
          read_res_5230 = np.empty(np.int32(1), dtype=ct.c_float)
          cl.enqueue_copy(self.queue, read_res_5230, prev_output_mem_4973,
                          device_offset=long((((k_4404 * m_4191) + j_4402) * np.int32(4))),
                          is_blocking=True)
          y_4406 = read_res_5230[np.int32(0)]
          y_4407 = (x_4405 * y_4406)
          res_4408 = (res_4403 + y_4407)
          res_tmp_5137 = res_4408
          res_4403 = res_tmp_5137
          k_4404 += one_5232
        res_4409 = res_4403
        y_4410 = slt32(j_4402, n_4193)
        assert y_4410, 'lstm.fut:12:16-12:16'
        cl.enqueue_copy(self.queue, mem_5015, np.array(res_4409,
                                                       dtype=ct.c_float),
                        device_offset=long((((i_4398 * n_4193) + j_4402) * np.int32(4))),
                        is_blocking=synchronous)
        j_4402 += one_5234
      i_4398 += one_5236
    mem_5018 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4978) if (bytes_4978 > np.int32(0)) else np.int32(1)))
    group_size_5138 = np.int32(512)
    num_groups_5139 = squot32((((n_4193 * m_4191) + group_size_5138) - np.int32(1)),
                              group_size_5138)
    if ((np.int32(1) * (num_groups_5139 * group_size_5138)) != np.int32(0)):
      self.map_kernel_4751_var.set_args(mem_5012, np.int32(n_4193), mem_5015,
                                        np.int32(m_4191), mem_5018)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_4751_var,
                                 (long((num_groups_5139 * group_size_5138)),),
                                 (long(group_size_5138),))
      if synchronous:
        self.queue.finish()
    mem_5021 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4989) if (bytes_4989 > np.int32(0)) else np.int32(1)))
    group_size_5140 = np.int32(512)
    num_groups_5141 = squot32((((m_4191 * size_4253) + group_size_5140) - np.int32(1)),
                              group_size_5140)
    if ((np.int32(1) * (num_groups_5141 * group_size_5140)) != np.int32(0)):
      self.map_kernel_4774_var.set_args(np.int32(size_4253), mem_5018,
                                        b_fg_mem_4963, np.int32(m_4191),
                                        mem_5021)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_4774_var,
                                 (long((num_groups_5141 * group_size_5140)),),
                                 (long(group_size_5140),))
      if synchronous:
        self.queue.finish()
    mem_5024 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4978) if (bytes_4978 > np.int32(0)) else np.int32(1)))
    group_size_5142 = np.int32(512)
    num_groups_5143 = squot32((((n_4193 * m_4191) + group_size_5142) - np.int32(1)),
                              group_size_5142)
    if ((np.int32(1) * (num_groups_5143 * group_size_5142)) != np.int32(0)):
      self.map_kernel_4791_var.set_args(np.int32(n_4193), mem_5021,
                                        np.int32(size_4253), np.int32(m_4191),
                                        mem_5024)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_4791_var,
                                 (long((num_groups_5143 * group_size_5142)),),
                                 (long(group_size_5142),))
      if synchronous:
        self.queue.finish()
    mem_5027 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4978) if (bytes_4978 > np.int32(0)) else np.int32(1)))
    group_size_5148 = np.int32(512)
    num_groups_5149 = squot32((((n_4193 * m_4191) + group_size_5148) - np.int32(1)),
                              group_size_5148)
    if ((np.int32(1) * (num_groups_5149 * group_size_5148)) != np.int32(0)):
      self.map_kernel_5144_var.set_args(np.int32(n_4193), mem_4977,
                                        np.int32(m_4191), mem_5027)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5144_var,
                                 (long((num_groups_5149 * group_size_5148)),),
                                 (long(group_size_5148),))
      if synchronous:
        self.queue.finish()
    i_4448 = np.int32(0)
    one_5244 = np.int32(1)
    for counter_5243 in range(m_4191):
      y_4449 = slt32(i_4448, n_4193)
      assert y_4449, 'lstm.fut:12:16-12:16'
      j_4452 = np.int32(0)
      one_5242 = np.int32(1)
      for counter_5241 in range(n_4193):
        res_4453 = np.float32(0.0)
        k_4454 = np.int32(0)
        one_5240 = np.int32(1)
        for counter_5239 in range(o_4192):
          read_res_5237 = np.empty(np.int32(1), dtype=ct.c_float)
          cl.enqueue_copy(self.queue, read_res_5237, W_og_mem_4965,
                          device_offset=long((((i_4448 * o_4192) + k_4454) * np.int32(4))),
                          is_blocking=True)
          x_4455 = read_res_5237[np.int32(0)]
          read_res_5238 = np.empty(np.int32(1), dtype=ct.c_float)
          cl.enqueue_copy(self.queue, read_res_5238, input_mem_4971,
                          device_offset=long((((k_4454 * n_4193) + j_4452) * np.int32(4))),
                          is_blocking=True)
          y_4456 = read_res_5238[np.int32(0)]
          y_4457 = (x_4455 * y_4456)
          res_4458 = (res_4453 + y_4457)
          res_tmp_5152 = res_4458
          res_4453 = res_tmp_5152
          k_4454 += one_5240
        res_4459 = res_4453
        y_4460 = slt32(j_4452, m_4191)
        assert y_4460, 'lstm.fut:12:16-12:16'
        cl.enqueue_copy(self.queue, mem_5027, np.array(res_4459,
                                                       dtype=ct.c_float),
                        device_offset=long((((i_4448 * m_4191) + j_4452) * np.int32(4))),
                        is_blocking=synchronous)
        j_4452 += one_5242
      i_4448 += one_5244
    mem_5030 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4983) if (bytes_4983 > np.int32(0)) else np.int32(1)))
    group_size_5157 = np.int32(512)
    num_groups_5158 = squot32((((m_4191 * n_4193) + group_size_5157) - np.int32(1)),
                              group_size_5157)
    if ((np.int32(1) * (num_groups_5158 * group_size_5157)) != np.int32(0)):
      self.map_kernel_5153_var.set_args(np.int32(n_4193), mem_4982,
                                        np.int32(m_4191), mem_5030)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5153_var,
                                 (long((num_groups_5158 * group_size_5157)),),
                                 (long(group_size_5157),))
      if synchronous:
        self.queue.finish()
    i_4467 = np.int32(0)
    one_5252 = np.int32(1)
    for counter_5251 in range(n_4193):
      y_4468 = slt32(i_4467, m_4191)
      assert y_4468, 'lstm.fut:12:16-12:16'
      j_4471 = np.int32(0)
      one_5250 = np.int32(1)
      for counter_5249 in range(m_4191):
        res_4472 = np.float32(0.0)
        k_4473 = np.int32(0)
        one_5248 = np.int32(1)
        for counter_5247 in range(n_4193):
          read_res_5245 = np.empty(np.int32(1), dtype=ct.c_float)
          cl.enqueue_copy(self.queue, read_res_5245, U_og_mem_4967,
                          device_offset=long((((i_4467 * n_4193) + k_4473) * np.int32(4))),
                          is_blocking=True)
          x_4474 = read_res_5245[np.int32(0)]
          read_res_5246 = np.empty(np.int32(1), dtype=ct.c_float)
          cl.enqueue_copy(self.queue, read_res_5246, prev_output_mem_4973,
                          device_offset=long((((k_4473 * m_4191) + j_4471) * np.int32(4))),
                          is_blocking=True)
          y_4475 = read_res_5246[np.int32(0)]
          y_4476 = (x_4474 * y_4475)
          res_4477 = (res_4472 + y_4476)
          res_tmp_5161 = res_4477
          res_4472 = res_tmp_5161
          k_4473 += one_5248
        res_4478 = res_4472
        y_4479 = slt32(j_4471, n_4193)
        assert y_4479, 'lstm.fut:12:16-12:16'
        cl.enqueue_copy(self.queue, mem_5030, np.array(res_4478,
                                                       dtype=ct.c_float),
                        device_offset=long((((i_4467 * n_4193) + j_4471) * np.int32(4))),
                        is_blocking=synchronous)
        j_4471 += one_5250
      i_4467 += one_5252
    mem_5033 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4978) if (bytes_4978 > np.int32(0)) else np.int32(1)))
    group_size_5162 = np.int32(512)
    num_groups_5163 = squot32((((n_4193 * m_4191) + group_size_5162) - np.int32(1)),
                              group_size_5162)
    if ((np.int32(1) * (num_groups_5163 * group_size_5162)) != np.int32(0)):
      self.map_kernel_4811_var.set_args(np.int32(n_4193), mem_5030, mem_5027,
                                        np.int32(m_4191), mem_5033)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_4811_var,
                                 (long((num_groups_5163 * group_size_5162)),),
                                 (long(group_size_5162),))
      if synchronous:
        self.queue.finish()
    mem_5036 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4989) if (bytes_4989 > np.int32(0)) else np.int32(1)))
    group_size_5164 = np.int32(512)
    num_groups_5165 = squot32((((m_4191 * size_4253) + group_size_5164) - np.int32(1)),
                              group_size_5164)
    if ((np.int32(1) * (num_groups_5165 * group_size_5164)) != np.int32(0)):
      self.map_kernel_4834_var.set_args(mem_5033, b_og_mem_4969,
                                        np.int32(size_4253), np.int32(m_4191),
                                        mem_5036)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_4834_var,
                                 (long((num_groups_5165 * group_size_5164)),),
                                 (long(group_size_5164),))
      if synchronous:
        self.queue.finish()
    mem_5039 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4978) if (bytes_4978 > np.int32(0)) else np.int32(1)))
    group_size_5166 = np.int32(512)
    num_groups_5167 = squot32((((n_4193 * m_4191) + group_size_5166) - np.int32(1)),
                              group_size_5166)
    if ((np.int32(1) * (num_groups_5167 * group_size_5166)) != np.int32(0)):
      self.map_kernel_4851_var.set_args(mem_5036, np.int32(n_4193),
                                        np.int32(size_4253), np.int32(m_4191),
                                        mem_5039)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_4851_var,
                                 (long((num_groups_5167 * group_size_5166)),),
                                 (long(group_size_5166),))
      if synchronous:
        self.queue.finish()
    nesting_size_4893 = (size_4253 * n_4193)
    bytes_5040 = (x_4979 * size_4253)
    mem_5042 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_5040) if (bytes_5040 > np.int32(0)) else np.int32(1)))
    group_size_5168 = np.int32(512)
    num_groups_5169 = squot32((((n_4193 * size_4253) + group_size_5168) - np.int32(1)),
                              group_size_5168)
    if ((np.int32(1) * (num_groups_5169 * group_size_5168)) != np.int32(0)):
      self.map_kernel_4895_var.set_args(mem_5024, np.int32(n_4193),
                                        np.int32(size_4253), mem_5009, mem_4994,
                                        prev_cell_mem_4975, np.int32(m_4191),
                                        mem_5042)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_4895_var,
                                 (long((num_groups_5169 * group_size_5168)),),
                                 (long(group_size_5168),))
      if synchronous:
        self.queue.finish()
    mem_5045 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_4978) if (bytes_4978 > np.int32(0)) else np.int32(1)))
    group_size_5170 = np.int32(512)
    num_groups_5171 = squot32((((n_4193 * m_4191) + group_size_5170) - np.int32(1)),
                              group_size_5170)
    if ((np.int32(1) * (num_groups_5171 * group_size_5170)) != np.int32(0)):
      self.map_kernel_4871_var.set_args(np.int32(n_4193), np.int32(size_4253),
                                        mem_5042, mem_5039, np.int32(m_4191),
                                        mem_5045)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_4871_var,
                                 (long((num_groups_5171 * group_size_5170)),),
                                 (long(group_size_5170),))
      if synchronous:
        self.queue.finish()
    out_mem_5060 = mem_5045
    out_memsize_5061 = bytes_4978
    out_mem_5062 = mem_5042
    out_memsize_5063 = bytes_5040
    return (out_memsize_5061, out_mem_5060, out_memsize_5063, out_mem_5062)
  def futhark_test(self, W_bi_mem_size_5046, b_bi_mem_size_5048,
                   input_mem_size_5050, W_bi_mem_5047, b_bi_mem_5049,
                   input_mem_5051, m_4573, o_4574, n_4575):
    bytes_5052 = (np.int32(4) * m_4573)
    mem_5053 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_5052) if (bytes_5052 > np.int32(0)) else np.int32(1)))
    group_size_5176 = np.int32(512)
    num_groups_5177 = squot32(((m_4573 + group_size_5176) - np.int32(1)),
                              group_size_5176)
    if ((np.int32(1) * (num_groups_5177 * group_size_5176)) != np.int32(0)):
      self.map_kernel_5174_var.set_args(np.int32(m_4573), mem_5053)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5174_var,
                                 (long((num_groups_5177 * group_size_5176)),),
                                 (long(group_size_5176),))
      if synchronous:
        self.queue.finish()
    x_5055 = (np.int32(4) * n_4575)
    bytes_5054 = (x_5055 * m_4573)
    mem_5056 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_5054) if (bytes_5054 > np.int32(0)) else np.int32(1)))
    group_size_5182 = np.int32(512)
    num_groups_5183 = squot32((((n_4575 * m_4573) + group_size_5182) - np.int32(1)),
                              group_size_5182)
    if ((np.int32(1) * (num_groups_5183 * group_size_5182)) != np.int32(0)):
      self.map_kernel_5178_var.set_args(mem_5053, np.int32(m_4573),
                                        np.int32(n_4575), mem_5056)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5178_var,
                                 (long((num_groups_5183 * group_size_5182)),),
                                 (long(group_size_5182),))
      if synchronous:
        self.queue.finish()
    i_4582 = np.int32(0)
    one_5260 = np.int32(1)
    for counter_5259 in range(m_4573):
      y_4583 = slt32(i_4582, n_4575)
      assert y_4583, 'lstm.fut:12:16-12:16'
      j_4586 = np.int32(0)
      one_5258 = np.int32(1)
      for counter_5257 in range(n_4575):
        res_4587 = np.float32(0.0)
        k_4588 = np.int32(0)
        one_5256 = np.int32(1)
        for counter_5255 in range(o_4574):
          read_res_5253 = np.empty(np.int32(1), dtype=ct.c_float)
          cl.enqueue_copy(self.queue, read_res_5253, W_bi_mem_5047,
                          device_offset=long((((i_4582 * o_4574) + k_4588) * np.int32(4))),
                          is_blocking=True)
          x_4589 = read_res_5253[np.int32(0)]
          read_res_5254 = np.empty(np.int32(1), dtype=ct.c_float)
          cl.enqueue_copy(self.queue, read_res_5254, input_mem_5051,
                          device_offset=long((((k_4588 * n_4575) + j_4586) * np.int32(4))),
                          is_blocking=True)
          y_4590 = read_res_5254[np.int32(0)]
          y_4591 = (x_4589 * y_4590)
          res_4592 = (res_4587 + y_4591)
          res_tmp_5186 = res_4592
          res_4587 = res_tmp_5186
          k_4588 += one_5256
        res_4593 = res_4587
        y_4594 = slt32(j_4586, m_4573)
        assert y_4594, 'lstm.fut:12:16-12:16'
        cl.enqueue_copy(self.queue, mem_5056, np.array(res_4593,
                                                       dtype=ct.c_float),
                        device_offset=long((((i_4582 * m_4573) + j_4586) * np.int32(4))),
                        is_blocking=synchronous)
        j_4586 += one_5258
      i_4582 += one_5260
    cond_4602 = (m_4573 == np.int32(0))
    p_and_eq_x_y_4604 = (cond_4602 and cond_4602)
    not_p_4605 = not(cond_4602)
    assert_arg_4606 = (p_and_eq_x_y_4604 or not_p_4605)
    assert assert_arg_4606, 'lstm.fut:18:17-18:17'
    nesting_size_4938 = (m_4573 * m_4573)
    bytes_5057 = (bytes_5052 * m_4573)
    mem_5059 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_5057) if (bytes_5057 > np.int32(0)) else np.int32(1)))
    group_size_5187 = np.int32(512)
    num_groups_5188 = squot32((((m_4573 * m_4573) + group_size_5187) - np.int32(1)),
                              group_size_5187)
    if ((np.int32(1) * (num_groups_5188 * group_size_5187)) != np.int32(0)):
      self.map_kernel_4940_var.set_args(mem_5056, b_bi_mem_5049,
                                        np.int32(m_4573), mem_5059)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_4940_var,
                                 (long((num_groups_5188 * group_size_5187)),),
                                 (long(group_size_5187),))
      if synchronous:
        self.queue.finish()
    assert_arg_4616 = (n_4575 == m_4573)
    assert assert_arg_4616, 'lstm.fut:37:17-37:17'
    out_mem_5172 = mem_5059
    out_memsize_5173 = bytes_5057
    return (out_memsize_5173, out_mem_5172)
  def main(self, W_bi_mem_4947_ext, U_bi_mem_4949_ext, b_bi_mem_4951_ext,
           W_ig_mem_4953_ext, U_ig_mem_4955_ext, b_ig_mem_4957_ext,
           W_fg_mem_4959_ext, U_fg_mem_4961_ext, b_fg_mem_4963_ext,
           W_og_mem_4965_ext, U_og_mem_4967_ext, b_og_mem_4969_ext,
           input_mem_4971_ext, prev_output_mem_4973_ext,
           prev_cell_mem_4975_ext):
    m_4191 = np.int32(W_bi_mem_4947_ext.shape[np.int32(0)])
    o_4192 = np.int32(W_bi_mem_4947_ext.shape[np.int32(1)])
    W_bi_mem_size_4946 = np.int32(W_bi_mem_4947_ext.nbytes)
    if (type(W_bi_mem_4947_ext) == cl.array.Array):
      W_bi_mem_4947 = W_bi_mem_4947_ext.data
    else:
      W_bi_mem_4947 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(W_bi_mem_size_4946) if (W_bi_mem_size_4946 > np.int32(0)) else np.int32(1)))
      if (W_bi_mem_size_4946 != np.int32(0)):
        cl.enqueue_copy(self.queue, W_bi_mem_4947, W_bi_mem_4947_ext,
                        is_blocking=synchronous)
    n_4193 = np.int32(U_bi_mem_4949_ext.shape[np.int32(0)])
    n_4193 = np.int32(U_bi_mem_4949_ext.shape[np.int32(1)])
    U_bi_mem_size_4948 = np.int32(U_bi_mem_4949_ext.nbytes)
    if (type(U_bi_mem_4949_ext) == cl.array.Array):
      U_bi_mem_4949 = U_bi_mem_4949_ext.data
    else:
      U_bi_mem_4949 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(U_bi_mem_size_4948) if (U_bi_mem_size_4948 > np.int32(0)) else np.int32(1)))
      if (U_bi_mem_size_4948 != np.int32(0)):
        cl.enqueue_copy(self.queue, U_bi_mem_4949, U_bi_mem_4949_ext,
                        is_blocking=synchronous)
    m_4191 = np.int32(b_bi_mem_4951_ext.shape[np.int32(0)])
    b_bi_mem_size_4950 = np.int32(b_bi_mem_4951_ext.nbytes)
    if (type(b_bi_mem_4951_ext) == cl.array.Array):
      b_bi_mem_4951 = b_bi_mem_4951_ext.data
    else:
      b_bi_mem_4951 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(b_bi_mem_size_4950) if (b_bi_mem_size_4950 > np.int32(0)) else np.int32(1)))
      if (b_bi_mem_size_4950 != np.int32(0)):
        cl.enqueue_copy(self.queue, b_bi_mem_4951, b_bi_mem_4951_ext,
                        is_blocking=synchronous)
    m_4191 = np.int32(W_ig_mem_4953_ext.shape[np.int32(0)])
    o_4192 = np.int32(W_ig_mem_4953_ext.shape[np.int32(1)])
    W_ig_mem_size_4952 = np.int32(W_ig_mem_4953_ext.nbytes)
    if (type(W_ig_mem_4953_ext) == cl.array.Array):
      W_ig_mem_4953 = W_ig_mem_4953_ext.data
    else:
      W_ig_mem_4953 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(W_ig_mem_size_4952) if (W_ig_mem_size_4952 > np.int32(0)) else np.int32(1)))
      if (W_ig_mem_size_4952 != np.int32(0)):
        cl.enqueue_copy(self.queue, W_ig_mem_4953, W_ig_mem_4953_ext,
                        is_blocking=synchronous)
    n_4193 = np.int32(U_ig_mem_4955_ext.shape[np.int32(0)])
    n_4193 = np.int32(U_ig_mem_4955_ext.shape[np.int32(1)])
    U_ig_mem_size_4954 = np.int32(U_ig_mem_4955_ext.nbytes)
    if (type(U_ig_mem_4955_ext) == cl.array.Array):
      U_ig_mem_4955 = U_ig_mem_4955_ext.data
    else:
      U_ig_mem_4955 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(U_ig_mem_size_4954) if (U_ig_mem_size_4954 > np.int32(0)) else np.int32(1)))
      if (U_ig_mem_size_4954 != np.int32(0)):
        cl.enqueue_copy(self.queue, U_ig_mem_4955, U_ig_mem_4955_ext,
                        is_blocking=synchronous)
    m_4191 = np.int32(b_ig_mem_4957_ext.shape[np.int32(0)])
    b_ig_mem_size_4956 = np.int32(b_ig_mem_4957_ext.nbytes)
    if (type(b_ig_mem_4957_ext) == cl.array.Array):
      b_ig_mem_4957 = b_ig_mem_4957_ext.data
    else:
      b_ig_mem_4957 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(b_ig_mem_size_4956) if (b_ig_mem_size_4956 > np.int32(0)) else np.int32(1)))
      if (b_ig_mem_size_4956 != np.int32(0)):
        cl.enqueue_copy(self.queue, b_ig_mem_4957, b_ig_mem_4957_ext,
                        is_blocking=synchronous)
    m_4191 = np.int32(W_fg_mem_4959_ext.shape[np.int32(0)])
    o_4192 = np.int32(W_fg_mem_4959_ext.shape[np.int32(1)])
    W_fg_mem_size_4958 = np.int32(W_fg_mem_4959_ext.nbytes)
    if (type(W_fg_mem_4959_ext) == cl.array.Array):
      W_fg_mem_4959 = W_fg_mem_4959_ext.data
    else:
      W_fg_mem_4959 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(W_fg_mem_size_4958) if (W_fg_mem_size_4958 > np.int32(0)) else np.int32(1)))
      if (W_fg_mem_size_4958 != np.int32(0)):
        cl.enqueue_copy(self.queue, W_fg_mem_4959, W_fg_mem_4959_ext,
                        is_blocking=synchronous)
    n_4193 = np.int32(U_fg_mem_4961_ext.shape[np.int32(0)])
    n_4193 = np.int32(U_fg_mem_4961_ext.shape[np.int32(1)])
    U_fg_mem_size_4960 = np.int32(U_fg_mem_4961_ext.nbytes)
    if (type(U_fg_mem_4961_ext) == cl.array.Array):
      U_fg_mem_4961 = U_fg_mem_4961_ext.data
    else:
      U_fg_mem_4961 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(U_fg_mem_size_4960) if (U_fg_mem_size_4960 > np.int32(0)) else np.int32(1)))
      if (U_fg_mem_size_4960 != np.int32(0)):
        cl.enqueue_copy(self.queue, U_fg_mem_4961, U_fg_mem_4961_ext,
                        is_blocking=synchronous)
    m_4191 = np.int32(b_fg_mem_4963_ext.shape[np.int32(0)])
    b_fg_mem_size_4962 = np.int32(b_fg_mem_4963_ext.nbytes)
    if (type(b_fg_mem_4963_ext) == cl.array.Array):
      b_fg_mem_4963 = b_fg_mem_4963_ext.data
    else:
      b_fg_mem_4963 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(b_fg_mem_size_4962) if (b_fg_mem_size_4962 > np.int32(0)) else np.int32(1)))
      if (b_fg_mem_size_4962 != np.int32(0)):
        cl.enqueue_copy(self.queue, b_fg_mem_4963, b_fg_mem_4963_ext,
                        is_blocking=synchronous)
    m_4191 = np.int32(W_og_mem_4965_ext.shape[np.int32(0)])
    o_4192 = np.int32(W_og_mem_4965_ext.shape[np.int32(1)])
    W_og_mem_size_4964 = np.int32(W_og_mem_4965_ext.nbytes)
    if (type(W_og_mem_4965_ext) == cl.array.Array):
      W_og_mem_4965 = W_og_mem_4965_ext.data
    else:
      W_og_mem_4965 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(W_og_mem_size_4964) if (W_og_mem_size_4964 > np.int32(0)) else np.int32(1)))
      if (W_og_mem_size_4964 != np.int32(0)):
        cl.enqueue_copy(self.queue, W_og_mem_4965, W_og_mem_4965_ext,
                        is_blocking=synchronous)
    n_4193 = np.int32(U_og_mem_4967_ext.shape[np.int32(0)])
    n_4193 = np.int32(U_og_mem_4967_ext.shape[np.int32(1)])
    U_og_mem_size_4966 = np.int32(U_og_mem_4967_ext.nbytes)
    if (type(U_og_mem_4967_ext) == cl.array.Array):
      U_og_mem_4967 = U_og_mem_4967_ext.data
    else:
      U_og_mem_4967 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(U_og_mem_size_4966) if (U_og_mem_size_4966 > np.int32(0)) else np.int32(1)))
      if (U_og_mem_size_4966 != np.int32(0)):
        cl.enqueue_copy(self.queue, U_og_mem_4967, U_og_mem_4967_ext,
                        is_blocking=synchronous)
    m_4191 = np.int32(b_og_mem_4969_ext.shape[np.int32(0)])
    b_og_mem_size_4968 = np.int32(b_og_mem_4969_ext.nbytes)
    if (type(b_og_mem_4969_ext) == cl.array.Array):
      b_og_mem_4969 = b_og_mem_4969_ext.data
    else:
      b_og_mem_4969 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(b_og_mem_size_4968) if (b_og_mem_size_4968 > np.int32(0)) else np.int32(1)))
      if (b_og_mem_size_4968 != np.int32(0)):
        cl.enqueue_copy(self.queue, b_og_mem_4969, b_og_mem_4969_ext,
                        is_blocking=synchronous)
    o_4192 = np.int32(input_mem_4971_ext.shape[np.int32(0)])
    n_4193 = np.int32(input_mem_4971_ext.shape[np.int32(1)])
    input_mem_size_4970 = np.int32(input_mem_4971_ext.nbytes)
    if (type(input_mem_4971_ext) == cl.array.Array):
      input_mem_4971 = input_mem_4971_ext.data
    else:
      input_mem_4971 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                 long(long(input_mem_size_4970) if (input_mem_size_4970 > np.int32(0)) else np.int32(1)))
      if (input_mem_size_4970 != np.int32(0)):
        cl.enqueue_copy(self.queue, input_mem_4971, input_mem_4971_ext,
                        is_blocking=synchronous)
    n_4193 = np.int32(prev_output_mem_4973_ext.shape[np.int32(0)])
    m_4191 = np.int32(prev_output_mem_4973_ext.shape[np.int32(1)])
    prev_output_mem_size_4972 = np.int32(prev_output_mem_4973_ext.nbytes)
    if (type(prev_output_mem_4973_ext) == cl.array.Array):
      prev_output_mem_4973 = prev_output_mem_4973_ext.data
    else:
      prev_output_mem_4973 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                       long(long(prev_output_mem_size_4972) if (prev_output_mem_size_4972 > np.int32(0)) else np.int32(1)))
      if (prev_output_mem_size_4972 != np.int32(0)):
        cl.enqueue_copy(self.queue, prev_output_mem_4973,
                        prev_output_mem_4973_ext, is_blocking=synchronous)
    n_4193 = np.int32(prev_cell_mem_4975_ext.shape[np.int32(0)])
    m_4191 = np.int32(prev_cell_mem_4975_ext.shape[np.int32(1)])
    prev_cell_mem_size_4974 = np.int32(prev_cell_mem_4975_ext.nbytes)
    if (type(prev_cell_mem_4975_ext) == cl.array.Array):
      prev_cell_mem_4975 = prev_cell_mem_4975_ext.data
    else:
      prev_cell_mem_4975 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                     long(long(prev_cell_mem_size_4974) if (prev_cell_mem_size_4974 > np.int32(0)) else np.int32(1)))
      if (prev_cell_mem_size_4974 != np.int32(0)):
        cl.enqueue_copy(self.queue, prev_cell_mem_4975, prev_cell_mem_4975_ext,
                        is_blocking=synchronous)
    (out_memsize_5061, out_mem_5060, out_memsize_5063,
     out_mem_5062) = self.futhark_main(W_bi_mem_size_4946, U_bi_mem_size_4948,
                                       b_bi_mem_size_4950, W_ig_mem_size_4952,
                                       U_ig_mem_size_4954, b_ig_mem_size_4956,
                                       W_fg_mem_size_4958, U_fg_mem_size_4960,
                                       b_fg_mem_size_4962, W_og_mem_size_4964,
                                       U_og_mem_size_4966, b_og_mem_size_4968,
                                       input_mem_size_4970,
                                       prev_output_mem_size_4972,
                                       prev_cell_mem_size_4974, W_bi_mem_4947,
                                       U_bi_mem_4949, b_bi_mem_4951,
                                       W_ig_mem_4953, U_ig_mem_4955,
                                       b_ig_mem_4957, W_fg_mem_4959,
                                       U_fg_mem_4961, b_fg_mem_4963,
                                       W_og_mem_4965, U_og_mem_4967,
                                       b_og_mem_4969, input_mem_4971,
                                       prev_output_mem_4973, prev_cell_mem_4975,
                                       m_4191, o_4192, n_4193)
    return (cl.array.Array(self.queue, (n_4193, m_4191), ct.c_float,
                           data=out_mem_5060), cl.array.Array(self.queue,
                                                              (n_4193, m_4191),
                                                              ct.c_float,
                                                              data=out_mem_5062))
  def test(self, W_bi_mem_5047_ext, b_bi_mem_5049_ext, input_mem_5051_ext):
    m_4573 = np.int32(W_bi_mem_5047_ext.shape[np.int32(0)])
    o_4574 = np.int32(W_bi_mem_5047_ext.shape[np.int32(1)])
    W_bi_mem_size_5046 = np.int32(W_bi_mem_5047_ext.nbytes)
    if (type(W_bi_mem_5047_ext) == cl.array.Array):
      W_bi_mem_5047 = W_bi_mem_5047_ext.data
    else:
      W_bi_mem_5047 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(W_bi_mem_size_5046) if (W_bi_mem_size_5046 > np.int32(0)) else np.int32(1)))
      if (W_bi_mem_size_5046 != np.int32(0)):
        cl.enqueue_copy(self.queue, W_bi_mem_5047, W_bi_mem_5047_ext,
                        is_blocking=synchronous)
    m_4573 = np.int32(b_bi_mem_5049_ext.shape[np.int32(0)])
    b_bi_mem_size_5048 = np.int32(b_bi_mem_5049_ext.nbytes)
    if (type(b_bi_mem_5049_ext) == cl.array.Array):
      b_bi_mem_5049 = b_bi_mem_5049_ext.data
    else:
      b_bi_mem_5049 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(b_bi_mem_size_5048) if (b_bi_mem_size_5048 > np.int32(0)) else np.int32(1)))
      if (b_bi_mem_size_5048 != np.int32(0)):
        cl.enqueue_copy(self.queue, b_bi_mem_5049, b_bi_mem_5049_ext,
                        is_blocking=synchronous)
    print "Hello."
    print input_mem_5051_ext.shape
    o_4574 = np.int32(input_mem_5051_ext.shape[np.int32(0)])
    n_4575 = np.int32(input_mem_5051_ext.shape[np.int32(1)])
    input_mem_size_5050 = np.int32(input_mem_5051_ext.nbytes)
    if (type(input_mem_5051_ext) == cl.array.Array):
      input_mem_5051 = input_mem_5051_ext.data
    else:
      input_mem_5051 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                 long(long(input_mem_size_5050) if (input_mem_size_5050 > np.int32(0)) else np.int32(1)))
      if (input_mem_size_5050 != np.int32(0)):
        cl.enqueue_copy(self.queue, input_mem_5051, input_mem_5051_ext,
                        is_blocking=synchronous)
    (out_memsize_5173, out_mem_5172) = self.futhark_test(W_bi_mem_size_5046,
                                                         b_bi_mem_size_5048,
                                                         input_mem_size_5050,
                                                         W_bi_mem_5047,
                                                         b_bi_mem_5049,
                                                         input_mem_5051, m_4573,
                                                         o_4574, n_4575)
    return cl.array.Array(self.queue, (n_4575, m_4573), ct.c_float,
                          data=out_mem_5172)