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
__kernel void map_kernel_5053(__global unsigned char *b_bi_mem_5412, __global
                              unsigned char *W_bi_mem_5408, int32_t m_4676,
                              int32_t size_4697, int32_t o_4677, __global
                              unsigned char *mem_5442, __global
                              unsigned char *U_bi_mem_5410, int32_t n_4678,
                              __global unsigned char *mem_5439, __global
                              unsigned char *mem_5445)
{
    const uint kernel_thread_index_5053 = get_global_id(0);
    
    if (kernel_thread_index_5053 >= m_4676 * size_4697)
        return;
    
    int32_t i_5054;
    int32_t i_5055;
    float y_5058;
    
    // compute thread index
    {
        i_5054 = squot32(kernel_thread_index_5053, size_4697);
        i_5055 = kernel_thread_index_5053 - squot32(kernel_thread_index_5053,
                                                    size_4697) * size_4697;
    }
    // read kernel parameters
    {
        y_5058 = *(__global float *) &b_bi_mem_5412[i_5054 * 4];
    }
    
    float res_5072;
    float acc_5063 = 0.0F;
    
    for (int i_5064 = 0; i_5064 < o_4677; i_5064++) {
        float binop_param_x_5066 = *(__global float *) &W_bi_mem_5408[(i_5054 *
                                                                       o_4677 +
                                                                       i_5064) *
                                                                      4];
        float binop_param_y_5067 = *(__global float *) &mem_5442[(i_5064 *
                                                                  n_4678 +
                                                                  i_5055) * 4];
        float res_5068 = binop_param_x_5066 * binop_param_y_5067;
        float res_5069 = acc_5063 + res_5068;
        float acc_tmp_5477 = res_5069;
        
        acc_5063 = acc_tmp_5477;
    }
    res_5072 = acc_5063;
    
    float res_5084;
    float acc_5075 = 0.0F;
    
    for (int i_5076 = 0; i_5076 < m_4676; i_5076++) {
        float binop_param_x_5078 = *(__global float *) &U_bi_mem_5410[(i_5054 *
                                                                       m_4676 +
                                                                       i_5076) *
                                                                      4];
        float binop_param_y_5079 = *(__global float *) &mem_5439[(i_5076 *
                                                                  n_4678 +
                                                                  i_5055) * 4];
        float res_5080 = binop_param_x_5078 * binop_param_y_5079;
        float res_5081 = acc_5075 + res_5080;
        float acc_tmp_5478 = res_5081;
        
        acc_5075 = acc_tmp_5478;
    }
    res_5084 = acc_5075;
    
    float res_5085 = res_5072 + res_5084;
    float res_5086 = res_5085 + y_5058;
    
    // write kernel result
    {
        *(__global float *) &mem_5445[(i_5054 * size_4697 + i_5055) * 4] =
            res_5086;
    }
}
__kernel void map_kernel_5098(int32_t m_4676, int32_t size_4697, __global
                              unsigned char *mem_5445, int32_t n_4678, __global
                              unsigned char *mem_5448)
{
    const uint kernel_thread_index_5098 = get_global_id(0);
    
    if (kernel_thread_index_5098 >= m_4676 * n_4678)
        return;
    
    int32_t i_5099;
    int32_t i_5100;
    float not_curried_5101;
    
    // compute thread index
    {
        i_5099 = squot32(kernel_thread_index_5098, n_4678);
        i_5100 = kernel_thread_index_5098 - squot32(kernel_thread_index_5098,
                                                    n_4678) * n_4678;
    }
    // read kernel parameters
    {
        not_curried_5101 = *(__global float *) &mem_5445[(i_5099 * size_4697 +
                                                          i_5100) * 4];
    }
    
    float arg_5102 = 0.0F - not_curried_5101;
    float res_5103 = fpow32(2.718280076980591F, arg_5102);
    float y_5104 = 1.0F + res_5103;
    float res_5105 = 1.0F / y_5104;
    
    // write kernel result
    {
        *(__global float *) &mem_5448[(i_5099 * n_4678 + i_5100) * 4] =
            res_5105;
    }
}
__kernel void map_kernel_5122(int32_t m_4676, __global
                              unsigned char *U_ig_mem_5416, int32_t size_4697,
                              int32_t o_4677, __global unsigned char *mem_5442,
                              __global unsigned char *W_ig_mem_5414,
                              int32_t n_4678, __global
                              unsigned char *b_ig_mem_5418, __global
                              unsigned char *mem_5439, __global
                              unsigned char *mem_5451)
{
    const uint kernel_thread_index_5122 = get_global_id(0);
    
    if (kernel_thread_index_5122 >= m_4676 * size_4697)
        return;
    
    int32_t i_5123;
    int32_t i_5124;
    float y_5129;
    
    // compute thread index
    {
        i_5123 = squot32(kernel_thread_index_5122, size_4697);
        i_5124 = kernel_thread_index_5122 - squot32(kernel_thread_index_5122,
                                                    size_4697) * size_4697;
    }
    // read kernel parameters
    {
        y_5129 = *(__global float *) &b_ig_mem_5418[i_5123 * 4];
    }
    
    float res_5141;
    float acc_5132 = 0.0F;
    
    for (int i_5133 = 0; i_5133 < o_4677; i_5133++) {
        float binop_param_y_5135 = *(__global float *) &mem_5442[(i_5133 *
                                                                  n_4678 +
                                                                  i_5124) * 4];
        float binop_param_x_5136 = *(__global float *) &W_ig_mem_5414[(i_5123 *
                                                                       o_4677 +
                                                                       i_5133) *
                                                                      4];
        float res_5137 = binop_param_x_5136 * binop_param_y_5135;
        float res_5138 = acc_5132 + res_5137;
        float acc_tmp_5483 = res_5138;
        
        acc_5132 = acc_tmp_5483;
    }
    res_5141 = acc_5132;
    
    float res_5153;
    float acc_5144 = 0.0F;
    
    for (int i_5145 = 0; i_5145 < m_4676; i_5145++) {
        float binop_param_x_5147 = *(__global float *) &U_ig_mem_5416[(i_5123 *
                                                                       m_4676 +
                                                                       i_5145) *
                                                                      4];
        float binop_param_y_5148 = *(__global float *) &mem_5439[(i_5145 *
                                                                  n_4678 +
                                                                  i_5124) * 4];
        float res_5149 = binop_param_x_5147 * binop_param_y_5148;
        float res_5150 = acc_5144 + res_5149;
        float acc_tmp_5484 = res_5150;
        
        acc_5144 = acc_tmp_5484;
    }
    res_5153 = acc_5144;
    
    float res_5154 = res_5141 + res_5153;
    float res_5155 = res_5154 + y_5129;
    
    // write kernel result
    {
        *(__global float *) &mem_5451[(i_5123 * size_4697 + i_5124) * 4] =
            res_5155;
    }
}
__kernel void map_kernel_5167(int32_t m_4676, int32_t size_4697, int32_t n_4678,
                              __global unsigned char *mem_5451, __global
                              unsigned char *mem_5454)
{
    const uint kernel_thread_index_5167 = get_global_id(0);
    
    if (kernel_thread_index_5167 >= m_4676 * n_4678)
        return;
    
    int32_t i_5168;
    int32_t i_5169;
    float not_curried_5170;
    
    // compute thread index
    {
        i_5168 = squot32(kernel_thread_index_5167, n_4678);
        i_5169 = kernel_thread_index_5167 - squot32(kernel_thread_index_5167,
                                                    n_4678) * n_4678;
    }
    // read kernel parameters
    {
        not_curried_5170 = *(__global float *) &mem_5451[(i_5168 * size_4697 +
                                                          i_5169) * 4];
    }
    
    float arg_5171 = 0.0F - not_curried_5170;
    float res_5172 = fpow32(2.718280076980591F, arg_5171);
    float y_5173 = 1.0F + res_5172;
    float res_5174 = 1.0F / y_5173;
    
    // write kernel result
    {
        *(__global float *) &mem_5454[(i_5168 * n_4678 + i_5169) * 4] =
            res_5174;
    }
}
__kernel void map_kernel_5191(__global unsigned char *W_fg_mem_5420,
                              int32_t m_4676, __global
                              unsigned char *b_fg_mem_5424, int32_t size_4697,
                              int32_t o_4677, __global unsigned char *mem_5442,
                              __global unsigned char *U_fg_mem_5422,
                              int32_t n_4678, __global unsigned char *mem_5439,
                              __global unsigned char *mem_5457)
{
    const uint kernel_thread_index_5191 = get_global_id(0);
    
    if (kernel_thread_index_5191 >= m_4676 * size_4697)
        return;
    
    int32_t i_5192;
    int32_t i_5193;
    float y_5197;
    
    // compute thread index
    {
        i_5192 = squot32(kernel_thread_index_5191, size_4697);
        i_5193 = kernel_thread_index_5191 - squot32(kernel_thread_index_5191,
                                                    size_4697) * size_4697;
    }
    // read kernel parameters
    {
        y_5197 = *(__global float *) &b_fg_mem_5424[i_5192 * 4];
    }
    
    float res_5210;
    float acc_5201 = 0.0F;
    
    for (int i_5202 = 0; i_5202 < o_4677; i_5202++) {
        float binop_param_x_5204 = *(__global float *) &W_fg_mem_5420[(i_5192 *
                                                                       o_4677 +
                                                                       i_5202) *
                                                                      4];
        float binop_param_y_5205 = *(__global float *) &mem_5442[(i_5202 *
                                                                  n_4678 +
                                                                  i_5193) * 4];
        float res_5206 = binop_param_x_5204 * binop_param_y_5205;
        float res_5207 = acc_5201 + res_5206;
        float acc_tmp_5489 = res_5207;
        
        acc_5201 = acc_tmp_5489;
    }
    res_5210 = acc_5201;
    
    float res_5222;
    float acc_5213 = 0.0F;
    
    for (int i_5214 = 0; i_5214 < m_4676; i_5214++) {
        float binop_param_x_5216 = *(__global float *) &U_fg_mem_5422[(i_5192 *
                                                                       m_4676 +
                                                                       i_5214) *
                                                                      4];
        float binop_param_y_5217 = *(__global float *) &mem_5439[(i_5214 *
                                                                  n_4678 +
                                                                  i_5193) * 4];
        float res_5218 = binop_param_x_5216 * binop_param_y_5217;
        float res_5219 = acc_5213 + res_5218;
        float acc_tmp_5490 = res_5219;
        
        acc_5213 = acc_tmp_5490;
    }
    res_5222 = acc_5213;
    
    float res_5223 = res_5210 + res_5222;
    float res_5224 = res_5223 + y_5197;
    
    // write kernel result
    {
        *(__global float *) &mem_5457[(i_5192 * size_4697 + i_5193) * 4] =
            res_5224;
    }
}
__kernel void map_kernel_5236(int32_t m_4676, int32_t size_4697, __global
                              unsigned char *mem_5457, int32_t n_4678, __global
                              unsigned char *mem_5460)
{
    const uint kernel_thread_index_5236 = get_global_id(0);
    
    if (kernel_thread_index_5236 >= m_4676 * n_4678)
        return;
    
    int32_t i_5237;
    int32_t i_5238;
    float not_curried_5239;
    
    // compute thread index
    {
        i_5237 = squot32(kernel_thread_index_5236, n_4678);
        i_5238 = kernel_thread_index_5236 - squot32(kernel_thread_index_5236,
                                                    n_4678) * n_4678;
    }
    // read kernel parameters
    {
        not_curried_5239 = *(__global float *) &mem_5457[(i_5237 * size_4697 +
                                                          i_5238) * 4];
    }
    
    float arg_5240 = 0.0F - not_curried_5239;
    float res_5241 = fpow32(2.718280076980591F, arg_5240);
    float y_5242 = 1.0F + res_5241;
    float res_5243 = 1.0F / y_5242;
    
    // write kernel result
    {
        *(__global float *) &mem_5460[(i_5237 * n_4678 + i_5238) * 4] =
            res_5243;
    }
}
__kernel void map_kernel_5260(__global unsigned char *U_og_mem_5428,
                              int32_t m_4676, int32_t size_4697, int32_t o_4677,
                              __global unsigned char *W_og_mem_5426, __global
                              unsigned char *mem_5442, __global
                              unsigned char *b_og_mem_5430, int32_t n_4678,
                              __global unsigned char *mem_5439, __global
                              unsigned char *mem_5463)
{
    const uint kernel_thread_index_5260 = get_global_id(0);
    
    if (kernel_thread_index_5260 >= m_4676 * size_4697)
        return;
    
    int32_t i_5261;
    int32_t i_5262;
    float y_5267;
    
    // compute thread index
    {
        i_5261 = squot32(kernel_thread_index_5260, size_4697);
        i_5262 = kernel_thread_index_5260 - squot32(kernel_thread_index_5260,
                                                    size_4697) * size_4697;
    }
    // read kernel parameters
    {
        y_5267 = *(__global float *) &b_og_mem_5430[i_5261 * 4];
    }
    
    float res_5279;
    float acc_5270 = 0.0F;
    
    for (int i_5271 = 0; i_5271 < o_4677; i_5271++) {
        float binop_param_y_5273 = *(__global float *) &mem_5442[(i_5271 *
                                                                  n_4678 +
                                                                  i_5262) * 4];
        float binop_param_x_5274 = *(__global float *) &W_og_mem_5426[(i_5261 *
                                                                       o_4677 +
                                                                       i_5271) *
                                                                      4];
        float res_5275 = binop_param_x_5274 * binop_param_y_5273;
        float res_5276 = acc_5270 + res_5275;
        float acc_tmp_5495 = res_5276;
        
        acc_5270 = acc_tmp_5495;
    }
    res_5279 = acc_5270;
    
    float res_5291;
    float acc_5282 = 0.0F;
    
    for (int i_5283 = 0; i_5283 < m_4676; i_5283++) {
        float binop_param_x_5285 = *(__global float *) &U_og_mem_5428[(i_5261 *
                                                                       m_4676 +
                                                                       i_5283) *
                                                                      4];
        float binop_param_y_5286 = *(__global float *) &mem_5439[(i_5283 *
                                                                  n_4678 +
                                                                  i_5262) * 4];
        float res_5287 = binop_param_x_5285 * binop_param_y_5286;
        float res_5288 = acc_5282 + res_5287;
        float acc_tmp_5496 = res_5288;
        
        acc_5282 = acc_tmp_5496;
    }
    res_5291 = acc_5282;
    
    float res_5292 = res_5279 + res_5291;
    float res_5293 = res_5292 + y_5267;
    
    // write kernel result
    {
        *(__global float *) &mem_5463[(i_5261 * size_4697 + i_5262) * 4] =
            res_5293;
    }
}
__kernel void map_kernel_5305(int32_t m_4676, int32_t size_4697, int32_t n_4678,
                              __global unsigned char *mem_5463, __global
                              unsigned char *mem_5466)
{
    const uint kernel_thread_index_5305 = get_global_id(0);
    
    if (kernel_thread_index_5305 >= m_4676 * n_4678)
        return;
    
    int32_t i_5306;
    int32_t i_5307;
    float not_curried_5308;
    
    // compute thread index
    {
        i_5306 = squot32(kernel_thread_index_5305, n_4678);
        i_5307 = kernel_thread_index_5305 - squot32(kernel_thread_index_5305,
                                                    n_4678) * n_4678;
    }
    // read kernel parameters
    {
        not_curried_5308 = *(__global float *) &mem_5463[(i_5306 * size_4697 +
                                                          i_5307) * 4];
    }
    
    float arg_5309 = 0.0F - not_curried_5308;
    float res_5310 = fpow32(2.718280076980591F, arg_5309);
    float y_5311 = 1.0F + res_5310;
    float res_5312 = 1.0F / y_5311;
    
    // write kernel result
    {
        *(__global float *) &mem_5466[(i_5306 * n_4678 + i_5307) * 4] =
            res_5312;
    }
}
__kernel void map_kernel_5349(__global unsigned char *mem_5448, __global
                              unsigned char *prev_cell_mem_5436, int32_t m_4676,
                              __global unsigned char *mem_5460,
                              int32_t size_4697, int32_t n_4678, __global
                              unsigned char *mem_5454, __global
                              unsigned char *mem_5469)
{
    const uint kernel_thread_index_5349 = get_global_id(0);
    
    if (kernel_thread_index_5349 >= m_4676 * size_4697)
        return;
    
    int32_t i_5350;
    int32_t i_5351;
    float x_5352;
    float x_5353;
    float y_5354;
    float y_5355;
    
    // compute thread index
    {
        i_5350 = squot32(kernel_thread_index_5349, size_4697);
        i_5351 = kernel_thread_index_5349 - squot32(kernel_thread_index_5349,
                                                    size_4697) * size_4697;
    }
    // read kernel parameters
    {
        x_5352 = *(__global float *) &mem_5448[(i_5350 * n_4678 + i_5351) * 4];
        x_5353 = *(__global float *) &prev_cell_mem_5436[(i_5350 * n_4678 +
                                                          i_5351) * 4];
        y_5354 = *(__global float *) &mem_5454[(i_5350 * n_4678 + i_5351) * 4];
        y_5355 = *(__global float *) &mem_5460[(i_5350 * n_4678 + i_5351) * 4];
    }
    
    float res_5356 = x_5352 * y_5354;
    float res_5357 = x_5353 * y_5355;
    float res_5358 = res_5356 + res_5357;
    
    // write kernel result
    {
        *(__global float *) &mem_5469[(i_5350 * size_4697 + i_5351) * 4] =
            res_5358;
    }
}
__kernel void map_kernel_5325(int32_t m_4676, int32_t size_4697, __global
                              unsigned char *mem_5469, __global
                              unsigned char *mem_5466, int32_t n_4678, __global
                              unsigned char *mem_5472)
{
    const uint kernel_thread_index_5325 = get_global_id(0);
    
    if (kernel_thread_index_5325 >= m_4676 * n_4678)
        return;
    
    int32_t i_5326;
    int32_t i_5327;
    float not_curried_5328;
    float x_5329;
    
    // compute thread index
    {
        i_5326 = squot32(kernel_thread_index_5325, n_4678);
        i_5327 = kernel_thread_index_5325 - squot32(kernel_thread_index_5325,
                                                    n_4678) * n_4678;
    }
    // read kernel parameters
    {
        not_curried_5328 = *(__global float *) &mem_5469[(i_5326 * size_4697 +
                                                          i_5327) * 4];
        x_5329 = *(__global float *) &mem_5466[(i_5326 * n_4678 + i_5327) * 4];
    }
    
    float arg_5330 = 0.0F - not_curried_5328;
    float res_5331 = fpow32(2.718280076980591F, arg_5330);
    float y_5332 = 1.0F + res_5331;
    float res_5333 = 1.0F / y_5332;
    float res_5334 = x_5329 * res_5333;
    
    // write kernel result
    {
        *(__global float *) &mem_5472[(i_5326 * n_4678 + i_5327) * 4] =
            res_5334;
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
    
    self.map_kernel_5053_var = program.map_kernel_5053
    self.map_kernel_5098_var = program.map_kernel_5098
    self.map_kernel_5122_var = program.map_kernel_5122
    self.map_kernel_5167_var = program.map_kernel_5167
    self.map_kernel_5191_var = program.map_kernel_5191
    self.map_kernel_5236_var = program.map_kernel_5236
    self.map_kernel_5260_var = program.map_kernel_5260
    self.map_kernel_5305_var = program.map_kernel_5305
    self.map_kernel_5349_var = program.map_kernel_5349
    self.map_kernel_5325_var = program.map_kernel_5325
  def futhark_main(self, W_bi_mem_size_5407, U_bi_mem_size_5409,
                   b_bi_mem_size_5411, W_ig_mem_size_5413, U_ig_mem_size_5415,
                   b_ig_mem_size_5417, W_fg_mem_size_5419, U_fg_mem_size_5421,
                   b_fg_mem_size_5423, W_og_mem_size_5425, U_og_mem_size_5427,
                   b_og_mem_size_5429, input_mem_size_5431,
                   prev_output_mem_size_5433, prev_cell_mem_size_5435,
                   W_bi_mem_5408, U_bi_mem_5410, b_bi_mem_5412, W_ig_mem_5414,
                   U_ig_mem_5416, b_ig_mem_5418, W_fg_mem_5420, U_fg_mem_5422,
                   b_fg_mem_5424, W_og_mem_5426, U_og_mem_5428, b_og_mem_5430,
                   input_mem_5432, prev_output_mem_5434, prev_cell_mem_5436,
                   m_4676, o_4677, n_4678):
    cond_4696 = (m_4676 == np.int32(0))
    if cond_4696:
      size_4697 = np.int32(0)
    else:
      size_4697 = n_4678
    eq_x_y_4698 = (n_4678 == np.int32(0))
    p_and_eq_x_y_4699 = (cond_4696 and eq_x_y_4698)
    not_p_4700 = not(cond_4696)
    assert_arg_4701 = (p_and_eq_x_y_4699 or not_p_4700)
    assert assert_arg_4701, 'lstm.fut:31:17-31:17'
    nesting_size_5051 = (size_4697 * m_4676)
    x_5438 = (np.int32(4) * m_4676)
    bytes_5437 = (x_5438 * n_4678)
    mem_5439 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_5437) if (bytes_5437 > np.int32(0)) else np.int32(1)))
    if (((m_4676 * n_4678) * np.int32(4)) != np.int32(0)):
      cl.enqueue_copy(self.queue, mem_5439, prev_output_mem_5434,
                      dest_offset=long(np.int32(0)),
                      src_offset=long(np.int32(0)),
                      byte_count=long(((m_4676 * n_4678) * np.int32(4))))
    if synchronous:
      self.queue.finish()
    x_5441 = (np.int32(4) * o_4677)
    bytes_5440 = (x_5441 * n_4678)
    mem_5442 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_5440) if (bytes_5440 > np.int32(0)) else np.int32(1)))
    if (((o_4677 * n_4678) * np.int32(4)) != np.int32(0)):
      cl.enqueue_copy(self.queue, mem_5442, input_mem_5432,
                      dest_offset=long(np.int32(0)),
                      src_offset=long(np.int32(0)),
                      byte_count=long(((o_4677 * n_4678) * np.int32(4))))
    if synchronous:
      self.queue.finish()
    bytes_5443 = (x_5438 * size_4697)
    mem_5445 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_5443) if (bytes_5443 > np.int32(0)) else np.int32(1)))
    group_size_5479 = np.int32(512)
    num_groups_5480 = squot32((((m_4676 * size_4697) + group_size_5479) - np.int32(1)),
                              group_size_5479)
    if ((np.int32(1) * (num_groups_5480 * group_size_5479)) != np.int32(0)):
      self.map_kernel_5053_var.set_args(b_bi_mem_5412, W_bi_mem_5408,
                                        np.int32(m_4676), np.int32(size_4697),
                                        np.int32(o_4677), mem_5442,
                                        U_bi_mem_5410, np.int32(n_4678),
                                        mem_5439, mem_5445)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5053_var,
                                 (long((num_groups_5480 * group_size_5479)),),
                                 (long(group_size_5479),))
      if synchronous:
        self.queue.finish()
    nesting_size_5096 = (n_4678 * m_4676)
    mem_5448 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_5437) if (bytes_5437 > np.int32(0)) else np.int32(1)))
    group_size_5481 = np.int32(512)
    num_groups_5482 = squot32((((m_4676 * n_4678) + group_size_5481) - np.int32(1)),
                              group_size_5481)
    if ((np.int32(1) * (num_groups_5482 * group_size_5481)) != np.int32(0)):
      self.map_kernel_5098_var.set_args(np.int32(m_4676), np.int32(size_4697),
                                        mem_5445, np.int32(n_4678), mem_5448)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5098_var,
                                 (long((num_groups_5482 * group_size_5481)),),
                                 (long(group_size_5481),))
      if synchronous:
        self.queue.finish()
    mem_5451 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_5443) if (bytes_5443 > np.int32(0)) else np.int32(1)))
    group_size_5485 = np.int32(512)
    num_groups_5486 = squot32((((m_4676 * size_4697) + group_size_5485) - np.int32(1)),
                              group_size_5485)
    if ((np.int32(1) * (num_groups_5486 * group_size_5485)) != np.int32(0)):
      self.map_kernel_5122_var.set_args(np.int32(m_4676), U_ig_mem_5416,
                                        np.int32(size_4697), np.int32(o_4677),
                                        mem_5442, W_ig_mem_5414,
                                        np.int32(n_4678), b_ig_mem_5418,
                                        mem_5439, mem_5451)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5122_var,
                                 (long((num_groups_5486 * group_size_5485)),),
                                 (long(group_size_5485),))
      if synchronous:
        self.queue.finish()
    mem_5454 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_5437) if (bytes_5437 > np.int32(0)) else np.int32(1)))
    group_size_5487 = np.int32(512)
    num_groups_5488 = squot32((((m_4676 * n_4678) + group_size_5487) - np.int32(1)),
                              group_size_5487)
    if ((np.int32(1) * (num_groups_5488 * group_size_5487)) != np.int32(0)):
      self.map_kernel_5167_var.set_args(np.int32(m_4676), np.int32(size_4697),
                                        np.int32(n_4678), mem_5451, mem_5454)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5167_var,
                                 (long((num_groups_5488 * group_size_5487)),),
                                 (long(group_size_5487),))
      if synchronous:
        self.queue.finish()
    mem_5457 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_5443) if (bytes_5443 > np.int32(0)) else np.int32(1)))
    group_size_5491 = np.int32(512)
    num_groups_5492 = squot32((((m_4676 * size_4697) + group_size_5491) - np.int32(1)),
                              group_size_5491)
    if ((np.int32(1) * (num_groups_5492 * group_size_5491)) != np.int32(0)):
      self.map_kernel_5191_var.set_args(W_fg_mem_5420, np.int32(m_4676),
                                        b_fg_mem_5424, np.int32(size_4697),
                                        np.int32(o_4677), mem_5442,
                                        U_fg_mem_5422, np.int32(n_4678),
                                        mem_5439, mem_5457)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5191_var,
                                 (long((num_groups_5492 * group_size_5491)),),
                                 (long(group_size_5491),))
      if synchronous:
        self.queue.finish()
    mem_5460 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_5437) if (bytes_5437 > np.int32(0)) else np.int32(1)))
    group_size_5493 = np.int32(512)
    num_groups_5494 = squot32((((m_4676 * n_4678) + group_size_5493) - np.int32(1)),
                              group_size_5493)
    if ((np.int32(1) * (num_groups_5494 * group_size_5493)) != np.int32(0)):
      self.map_kernel_5236_var.set_args(np.int32(m_4676), np.int32(size_4697),
                                        mem_5457, np.int32(n_4678), mem_5460)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5236_var,
                                 (long((num_groups_5494 * group_size_5493)),),
                                 (long(group_size_5493),))
      if synchronous:
        self.queue.finish()
    mem_5463 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_5443) if (bytes_5443 > np.int32(0)) else np.int32(1)))
    group_size_5497 = np.int32(512)
    num_groups_5498 = squot32((((m_4676 * size_4697) + group_size_5497) - np.int32(1)),
                              group_size_5497)
    if ((np.int32(1) * (num_groups_5498 * group_size_5497)) != np.int32(0)):
      self.map_kernel_5260_var.set_args(U_og_mem_5428, np.int32(m_4676),
                                        np.int32(size_4697), np.int32(o_4677),
                                        W_og_mem_5426, mem_5442, b_og_mem_5430,
                                        np.int32(n_4678), mem_5439, mem_5463)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5260_var,
                                 (long((num_groups_5498 * group_size_5497)),),
                                 (long(group_size_5497),))
      if synchronous:
        self.queue.finish()
    mem_5466 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_5437) if (bytes_5437 > np.int32(0)) else np.int32(1)))
    group_size_5499 = np.int32(512)
    num_groups_5500 = squot32((((m_4676 * n_4678) + group_size_5499) - np.int32(1)),
                              group_size_5499)
    if ((np.int32(1) * (num_groups_5500 * group_size_5499)) != np.int32(0)):
      self.map_kernel_5305_var.set_args(np.int32(m_4676), np.int32(size_4697),
                                        np.int32(n_4678), mem_5463, mem_5466)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5305_var,
                                 (long((num_groups_5500 * group_size_5499)),),
                                 (long(group_size_5499),))
      if synchronous:
        self.queue.finish()
    mem_5469 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_5443) if (bytes_5443 > np.int32(0)) else np.int32(1)))
    group_size_5501 = np.int32(512)
    num_groups_5502 = squot32((((m_4676 * size_4697) + group_size_5501) - np.int32(1)),
                              group_size_5501)
    if ((np.int32(1) * (num_groups_5502 * group_size_5501)) != np.int32(0)):
      self.map_kernel_5349_var.set_args(mem_5448, prev_cell_mem_5436,
                                        np.int32(m_4676), mem_5460,
                                        np.int32(size_4697), np.int32(n_4678),
                                        mem_5454, mem_5469)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5349_var,
                                 (long((num_groups_5502 * group_size_5501)),),
                                 (long(group_size_5501),))
      if synchronous:
        self.queue.finish()
    mem_5472 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_5437) if (bytes_5437 > np.int32(0)) else np.int32(1)))
    group_size_5503 = np.int32(512)
    num_groups_5504 = squot32((((m_4676 * n_4678) + group_size_5503) - np.int32(1)),
                              group_size_5503)
    if ((np.int32(1) * (num_groups_5504 * group_size_5503)) != np.int32(0)):
      self.map_kernel_5325_var.set_args(np.int32(m_4676), np.int32(size_4697),
                                        mem_5469, mem_5466, np.int32(n_4678),
                                        mem_5472)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_5325_var,
                                 (long((num_groups_5504 * group_size_5503)),),
                                 (long(group_size_5503),))
      if synchronous:
        self.queue.finish()
    out_mem_5473 = mem_5472
    out_memsize_5474 = bytes_5437
    out_mem_5475 = mem_5469
    out_memsize_5476 = bytes_5443
    return (out_memsize_5474, out_mem_5473, out_memsize_5476, out_mem_5475)
  def main(self, W_bi_mem_5408_ext, U_bi_mem_5410_ext, b_bi_mem_5412_ext,
           W_ig_mem_5414_ext, U_ig_mem_5416_ext, b_ig_mem_5418_ext,
           W_fg_mem_5420_ext, U_fg_mem_5422_ext, b_fg_mem_5424_ext,
           W_og_mem_5426_ext, U_og_mem_5428_ext, b_og_mem_5430_ext,
           input_mem_5432_ext, prev_output_mem_5434_ext,
           prev_cell_mem_5436_ext):
    m_4676 = np.int32(W_bi_mem_5408_ext.shape[np.int32(0)])
    o_4677 = np.int32(W_bi_mem_5408_ext.shape[np.int32(1)])
    W_bi_mem_size_5407 = np.int32(W_bi_mem_5408_ext.nbytes)
    if (type(W_bi_mem_5408_ext) == cl.array.Array):
      W_bi_mem_5408 = W_bi_mem_5408_ext.data
    else:
      W_bi_mem_5408 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(W_bi_mem_size_5407) if (W_bi_mem_size_5407 > np.int32(0)) else np.int32(1)))
      if (W_bi_mem_size_5407 != np.int32(0)):
        cl.enqueue_copy(self.queue, W_bi_mem_5408, W_bi_mem_5408_ext,
                        is_blocking=synchronous)
    m_4676 = np.int32(U_bi_mem_5410_ext.shape[np.int32(0)])
    m_4676 = np.int32(U_bi_mem_5410_ext.shape[np.int32(1)])
    U_bi_mem_size_5409 = np.int32(U_bi_mem_5410_ext.nbytes)
    if (type(U_bi_mem_5410_ext) == cl.array.Array):
      U_bi_mem_5410 = U_bi_mem_5410_ext.data
    else:
      U_bi_mem_5410 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(U_bi_mem_size_5409) if (U_bi_mem_size_5409 > np.int32(0)) else np.int32(1)))
      if (U_bi_mem_size_5409 != np.int32(0)):
        cl.enqueue_copy(self.queue, U_bi_mem_5410, U_bi_mem_5410_ext,
                        is_blocking=synchronous)
    m_4676 = np.int32(b_bi_mem_5412_ext.shape[np.int32(0)])
    b_bi_mem_size_5411 = np.int32(b_bi_mem_5412_ext.nbytes)
    if (type(b_bi_mem_5412_ext) == cl.array.Array):
      b_bi_mem_5412 = b_bi_mem_5412_ext.data
    else:
      b_bi_mem_5412 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(b_bi_mem_size_5411) if (b_bi_mem_size_5411 > np.int32(0)) else np.int32(1)))
      if (b_bi_mem_size_5411 != np.int32(0)):
        cl.enqueue_copy(self.queue, b_bi_mem_5412, b_bi_mem_5412_ext,
                        is_blocking=synchronous)
    m_4676 = np.int32(W_ig_mem_5414_ext.shape[np.int32(0)])
    o_4677 = np.int32(W_ig_mem_5414_ext.shape[np.int32(1)])
    W_ig_mem_size_5413 = np.int32(W_ig_mem_5414_ext.nbytes)
    if (type(W_ig_mem_5414_ext) == cl.array.Array):
      W_ig_mem_5414 = W_ig_mem_5414_ext.data
    else:
      W_ig_mem_5414 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(W_ig_mem_size_5413) if (W_ig_mem_size_5413 > np.int32(0)) else np.int32(1)))
      if (W_ig_mem_size_5413 != np.int32(0)):
        cl.enqueue_copy(self.queue, W_ig_mem_5414, W_ig_mem_5414_ext,
                        is_blocking=synchronous)
    m_4676 = np.int32(U_ig_mem_5416_ext.shape[np.int32(0)])
    m_4676 = np.int32(U_ig_mem_5416_ext.shape[np.int32(1)])
    U_ig_mem_size_5415 = np.int32(U_ig_mem_5416_ext.nbytes)
    if (type(U_ig_mem_5416_ext) == cl.array.Array):
      U_ig_mem_5416 = U_ig_mem_5416_ext.data
    else:
      U_ig_mem_5416 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(U_ig_mem_size_5415) if (U_ig_mem_size_5415 > np.int32(0)) else np.int32(1)))
      if (U_ig_mem_size_5415 != np.int32(0)):
        cl.enqueue_copy(self.queue, U_ig_mem_5416, U_ig_mem_5416_ext,
                        is_blocking=synchronous)
    m_4676 = np.int32(b_ig_mem_5418_ext.shape[np.int32(0)])
    b_ig_mem_size_5417 = np.int32(b_ig_mem_5418_ext.nbytes)
    if (type(b_ig_mem_5418_ext) == cl.array.Array):
      b_ig_mem_5418 = b_ig_mem_5418_ext.data
    else:
      b_ig_mem_5418 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(b_ig_mem_size_5417) if (b_ig_mem_size_5417 > np.int32(0)) else np.int32(1)))
      if (b_ig_mem_size_5417 != np.int32(0)):
        cl.enqueue_copy(self.queue, b_ig_mem_5418, b_ig_mem_5418_ext,
                        is_blocking=synchronous)
    m_4676 = np.int32(W_fg_mem_5420_ext.shape[np.int32(0)])
    o_4677 = np.int32(W_fg_mem_5420_ext.shape[np.int32(1)])
    W_fg_mem_size_5419 = np.int32(W_fg_mem_5420_ext.nbytes)
    if (type(W_fg_mem_5420_ext) == cl.array.Array):
      W_fg_mem_5420 = W_fg_mem_5420_ext.data
    else:
      W_fg_mem_5420 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(W_fg_mem_size_5419) if (W_fg_mem_size_5419 > np.int32(0)) else np.int32(1)))
      if (W_fg_mem_size_5419 != np.int32(0)):
        cl.enqueue_copy(self.queue, W_fg_mem_5420, W_fg_mem_5420_ext,
                        is_blocking=synchronous)
    m_4676 = np.int32(U_fg_mem_5422_ext.shape[np.int32(0)])
    m_4676 = np.int32(U_fg_mem_5422_ext.shape[np.int32(1)])
    U_fg_mem_size_5421 = np.int32(U_fg_mem_5422_ext.nbytes)
    if (type(U_fg_mem_5422_ext) == cl.array.Array):
      U_fg_mem_5422 = U_fg_mem_5422_ext.data
    else:
      U_fg_mem_5422 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(U_fg_mem_size_5421) if (U_fg_mem_size_5421 > np.int32(0)) else np.int32(1)))
      if (U_fg_mem_size_5421 != np.int32(0)):
        cl.enqueue_copy(self.queue, U_fg_mem_5422, U_fg_mem_5422_ext,
                        is_blocking=synchronous)
    m_4676 = np.int32(b_fg_mem_5424_ext.shape[np.int32(0)])
    b_fg_mem_size_5423 = np.int32(b_fg_mem_5424_ext.nbytes)
    if (type(b_fg_mem_5424_ext) == cl.array.Array):
      b_fg_mem_5424 = b_fg_mem_5424_ext.data
    else:
      b_fg_mem_5424 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(b_fg_mem_size_5423) if (b_fg_mem_size_5423 > np.int32(0)) else np.int32(1)))
      if (b_fg_mem_size_5423 != np.int32(0)):
        cl.enqueue_copy(self.queue, b_fg_mem_5424, b_fg_mem_5424_ext,
                        is_blocking=synchronous)
    m_4676 = np.int32(W_og_mem_5426_ext.shape[np.int32(0)])
    o_4677 = np.int32(W_og_mem_5426_ext.shape[np.int32(1)])
    W_og_mem_size_5425 = np.int32(W_og_mem_5426_ext.nbytes)
    if (type(W_og_mem_5426_ext) == cl.array.Array):
      W_og_mem_5426 = W_og_mem_5426_ext.data
    else:
      W_og_mem_5426 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(W_og_mem_size_5425) if (W_og_mem_size_5425 > np.int32(0)) else np.int32(1)))
      if (W_og_mem_size_5425 != np.int32(0)):
        cl.enqueue_copy(self.queue, W_og_mem_5426, W_og_mem_5426_ext,
                        is_blocking=synchronous)
    m_4676 = np.int32(U_og_mem_5428_ext.shape[np.int32(0)])
    m_4676 = np.int32(U_og_mem_5428_ext.shape[np.int32(1)])
    U_og_mem_size_5427 = np.int32(U_og_mem_5428_ext.nbytes)
    if (type(U_og_mem_5428_ext) == cl.array.Array):
      U_og_mem_5428 = U_og_mem_5428_ext.data
    else:
      U_og_mem_5428 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(U_og_mem_size_5427) if (U_og_mem_size_5427 > np.int32(0)) else np.int32(1)))
      if (U_og_mem_size_5427 != np.int32(0)):
        cl.enqueue_copy(self.queue, U_og_mem_5428, U_og_mem_5428_ext,
                        is_blocking=synchronous)
    m_4676 = np.int32(b_og_mem_5430_ext.shape[np.int32(0)])
    b_og_mem_size_5429 = np.int32(b_og_mem_5430_ext.nbytes)
    if (type(b_og_mem_5430_ext) == cl.array.Array):
      b_og_mem_5430 = b_og_mem_5430_ext.data
    else:
      b_og_mem_5430 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                long(long(b_og_mem_size_5429) if (b_og_mem_size_5429 > np.int32(0)) else np.int32(1)))
      if (b_og_mem_size_5429 != np.int32(0)):
        cl.enqueue_copy(self.queue, b_og_mem_5430, b_og_mem_5430_ext,
                        is_blocking=synchronous)
    o_4677 = np.int32(input_mem_5432_ext.shape[np.int32(0)])
    n_4678 = np.int32(input_mem_5432_ext.shape[np.int32(1)])
    input_mem_size_5431 = np.int32(input_mem_5432_ext.nbytes)
    if (type(input_mem_5432_ext) == cl.array.Array):
      input_mem_5432 = input_mem_5432_ext.data
    else:
      input_mem_5432 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                 long(long(input_mem_size_5431) if (input_mem_size_5431 > np.int32(0)) else np.int32(1)))
      if (input_mem_size_5431 != np.int32(0)):
        cl.enqueue_copy(self.queue, input_mem_5432, input_mem_5432_ext,
                        is_blocking=synchronous)
    m_4676 = np.int32(prev_output_mem_5434_ext.shape[np.int32(0)])
    n_4678 = np.int32(prev_output_mem_5434_ext.shape[np.int32(1)])
    prev_output_mem_size_5433 = np.int32(prev_output_mem_5434_ext.nbytes)
    if (type(prev_output_mem_5434_ext) == cl.array.Array):
      prev_output_mem_5434 = prev_output_mem_5434_ext.data
    else:
      prev_output_mem_5434 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                       long(long(prev_output_mem_size_5433) if (prev_output_mem_size_5433 > np.int32(0)) else np.int32(1)))
      if (prev_output_mem_size_5433 != np.int32(0)):
        cl.enqueue_copy(self.queue, prev_output_mem_5434,
                        prev_output_mem_5434_ext, is_blocking=synchronous)
    m_4676 = np.int32(prev_cell_mem_5436_ext.shape[np.int32(0)])
    n_4678 = np.int32(prev_cell_mem_5436_ext.shape[np.int32(1)])
    prev_cell_mem_size_5435 = np.int32(prev_cell_mem_5436_ext.nbytes)
    if (type(prev_cell_mem_5436_ext) == cl.array.Array):
      prev_cell_mem_5436 = prev_cell_mem_5436_ext.data
    else:
      prev_cell_mem_5436 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                     long(long(prev_cell_mem_size_5435) if (prev_cell_mem_size_5435 > np.int32(0)) else np.int32(1)))
      if (prev_cell_mem_size_5435 != np.int32(0)):
        cl.enqueue_copy(self.queue, prev_cell_mem_5436, prev_cell_mem_5436_ext,
                        is_blocking=synchronous)
    (out_memsize_5474, out_mem_5473, out_memsize_5476,
     out_mem_5475) = self.futhark_main(W_bi_mem_size_5407, U_bi_mem_size_5409,
                                       b_bi_mem_size_5411, W_ig_mem_size_5413,
                                       U_ig_mem_size_5415, b_ig_mem_size_5417,
                                       W_fg_mem_size_5419, U_fg_mem_size_5421,
                                       b_fg_mem_size_5423, W_og_mem_size_5425,
                                       U_og_mem_size_5427, b_og_mem_size_5429,
                                       input_mem_size_5431,
                                       prev_output_mem_size_5433,
                                       prev_cell_mem_size_5435, W_bi_mem_5408,
                                       U_bi_mem_5410, b_bi_mem_5412,
                                       W_ig_mem_5414, U_ig_mem_5416,
                                       b_ig_mem_5418, W_fg_mem_5420,
                                       U_fg_mem_5422, b_fg_mem_5424,
                                       W_og_mem_5426, U_og_mem_5428,
                                       b_og_mem_5430, input_mem_5432,
                                       prev_output_mem_5434, prev_cell_mem_5436,
                                       m_4676, o_4677, n_4678)
    return (cl.array.Array(self.queue, (m_4676, n_4678), ct.c_float,
                           data=out_mem_5473), cl.array.Array(self.queue,
                                                              (m_4676, n_4678),
                                                              ct.c_float,
                                                              data=out_mem_5475))