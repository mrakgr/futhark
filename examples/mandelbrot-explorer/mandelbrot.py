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
__kernel void map_kernel_930(float view_737, float y_745, float res_741,
                             int32_t width_734, __global unsigned char *mem_940,
                             __global unsigned char *mem_942)
{
    const uint kernel_thread_index_930 = get_global_id(0);
    
    if (kernel_thread_index_930 >= width_734)
        return;
    
    int32_t i_931;
    
    // compute thread index
    {
        i_931 = kernel_thread_index_930;
    }
    // read kernel parameters
    { }
    
    float x_933 = sitofp_i32_f32(i_931);
    float x_934 = x_933 * res_741;
    float y_935 = x_934 / y_745;
    float res_936 = view_737 + y_935;
    float x_937 = res_936 * res_936;
    
    // write kernel result
    {
        *(__global float *) &mem_940[i_931 * 4] = res_936;
        *(__global float *) &mem_942[i_931 * 4] = x_937;
    }
}
__kernel void map_kernel_881(__global unsigned char *mem_940, __global
                             unsigned char *res_mem_948, int32_t limit_736,
                             __global unsigned char *mem_944, __global
                             unsigned char *mem_946, int32_t width_734,
                             float res_742, __global unsigned char *mem_942,
                             float view_738, float y_746,
                             int32_t nesting_size_879, char x_747,
                             int32_t height_735, __global
                             unsigned char *mem_952)
{
    const uint kernel_thread_index_881 = get_global_id(0);
    
    if (kernel_thread_index_881 >= width_734 * height_735)
        return;
    
    int32_t i_882;
    int32_t i_883;
    float res_884;
    float x_885;
    
    // compute thread index
    {
        i_882 = squot32(kernel_thread_index_881, height_735);
        i_883 = kernel_thread_index_881 - squot32(kernel_thread_index_881,
                                                  height_735) * height_735;
    }
    // read kernel parameters
    {
        res_884 = *(__global float *) &mem_940[i_882 * 4];
        x_885 = *(__global float *) &mem_942[i_882 * 4];
    }
    
    float x_887 = sitofp_i32_f32(i_883);
    float x_888 = x_887 * res_742;
    float y_889 = x_888 / y_746;
    float res_890 = view_738 + y_889;
    float y_891 = res_890 * res_890;
    float res_892 = x_885 + y_891;
    char y_893 = res_892 < 4.0F;
    char loop_cond_894 = x_747 && y_893;
    char nameless_914;
    float c_915;
    float c_916;
    int32_t i_917;
    char loop_while_895;
    float c_896;
    float c_897;
    int32_t i_898;
    
    loop_while_895 = loop_cond_894;
    c_896 = res_884;
    c_897 = res_890;
    i_898 = 0;
    while (loop_while_895) {
        float x_899 = c_896 * c_896;
        float y_900 = c_897 * c_897;
        float res_901 = x_899 - y_900;
        float x_902 = c_896 * c_897;
        float y_903 = c_897 * c_896;
        float res_904 = x_902 + y_903;
        float res_905 = res_884 + res_901;
        float res_906 = res_890 + res_904;
        int32_t res_907 = i_898 + 1;
        char x_908 = slt32(res_907, limit_736);
        float x_909 = res_905 * res_905;
        float y_910 = res_906 * res_906;
        float res_911 = x_909 + y_910;
        char y_912 = res_911 < 4.0F;
        char loop_cond_913 = x_908 && y_912;
        char loop_while_tmp_967 = loop_cond_913;
        float c_tmp_968 = res_905;
        float c_tmp_969 = res_906;
        int32_t i_tmp_970;
        
        i_tmp_970 = res_907;
        loop_while_895 = loop_while_tmp_967;
        c_896 = c_tmp_968;
        c_897 = c_tmp_969;
        i_898 = i_tmp_970;
    }
    nameless_914 = loop_while_895;
    c_915 = c_896;
    c_916 = c_897;
    i_917 = i_898;
    
    char cond_918 = limit_736 == i_917;
    int32_t trunc_arg_919 = 3 * i_917;
    int8_t res_920 = sext_i32_i8(trunc_arg_919);
    int32_t trunc_arg_921 = 5 * i_917;
    int8_t res_922 = sext_i32_i8(trunc_arg_921);
    int32_t trunc_arg_923 = 7 * i_917;
    int8_t res_924 = sext_i32_i8(trunc_arg_923);
    
    if (cond_918) {
        *(__global int8_t *) &mem_944[kernel_thread_index_881] = 0;
        *(__global int8_t *) &mem_944[nesting_size_879 +
                                      kernel_thread_index_881] = 0;
        *(__global int8_t *) &mem_944[2 * nesting_size_879 +
                                      kernel_thread_index_881] = 0;
        for (int i_971 = 0; i_971 < 3; i_971++) {
            *(__global int8_t *) &res_mem_948[nesting_size_879 * i_971 +
                                              kernel_thread_index_881] =
                *(__global int8_t *) &mem_944[nesting_size_879 * i_971 +
                                              kernel_thread_index_881];
        }
    } else {
        *(__global int8_t *) &mem_946[kernel_thread_index_881] = res_920;
        *(__global int8_t *) &mem_946[nesting_size_879 +
                                      kernel_thread_index_881] = res_922;
        *(__global int8_t *) &mem_946[2 * nesting_size_879 +
                                      kernel_thread_index_881] = res_924;
        for (int i_972 = 0; i_972 < 3; i_972++) {
            *(__global int8_t *) &res_mem_948[nesting_size_879 * i_972 +
                                              kernel_thread_index_881] =
                *(__global int8_t *) &mem_946[nesting_size_879 * i_972 +
                                              kernel_thread_index_881];
        }
    }
    // write kernel result
    {
        for (int i_973 = 0; i_973 < 3; i_973++) {
            *(__global int8_t *) &mem_952[3 * (height_735 * i_882) +
                                          (height_735 * i_973 + i_883)] =
                *(__global int8_t *) &res_mem_948[nesting_size_879 * i_973 +
                                                  kernel_thread_index_881];
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
class mandelbrot:
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
    
    self.map_kernel_930_var = program.map_kernel_930
    self.map_kernel_881_var = program.map_kernel_881
    self.fut_kernel_map_transpose_i8_var = program.fut_kernel_map_transpose_i8
  def futhark_main(self, width_734, height_735, limit_736, view_737, view_738,
                   view_739, view_740):
    res_741 = (view_739 - view_737)
    res_742 = (view_740 - view_738)
    y_745 = sitofp_i32_f32(width_734)
    y_746 = sitofp_i32_f32(height_735)
    x_747 = slt32(np.int32(0), limit_736)
    bytes_939 = (np.int32(4) * width_734)
    mem_940 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                        long(long(bytes_939) if (bytes_939 > np.int32(0)) else np.int32(1)))
    mem_942 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                        long(long(bytes_939) if (bytes_939 > np.int32(0)) else np.int32(1)))
    group_size_965 = np.int32(512)
    num_groups_966 = squot32(((width_734 + group_size_965) - np.int32(1)),
                             group_size_965)
    if ((np.int32(1) * (num_groups_966 * group_size_965)) != np.int32(0)):
      self.map_kernel_930_var.set_args(np.float32(view_737), np.float32(y_745),
                                       np.float32(res_741), np.int32(width_734),
                                       mem_940, mem_942)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_930_var,
                                 (long((num_groups_966 * group_size_965)),),
                                 (long(group_size_965),))
      if synchronous:
        self.queue.finish()
    nesting_size_879 = (height_735 * width_734)
    x_951 = (width_734 * np.int32(3))
    bytes_949 = (x_951 * height_735)
    mem_952 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                        long(long(bytes_949) if (bytes_949 > np.int32(0)) else np.int32(1)))
    total_size_960 = (nesting_size_879 * np.int32(3))
    res_mem_948 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                            long(long(total_size_960) if (total_size_960 > np.int32(0)) else np.int32(1)))
    total_size_961 = (nesting_size_879 * np.int32(3))
    mem_944 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                        long(long(total_size_961) if (total_size_961 > np.int32(0)) else np.int32(1)))
    total_size_962 = (nesting_size_879 * np.int32(3))
    mem_946 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                        long(long(total_size_962) if (total_size_962 > np.int32(0)) else np.int32(1)))
    group_size_974 = np.int32(512)
    num_groups_975 = squot32((((width_734 * height_735) + group_size_974) - np.int32(1)),
                             group_size_974)
    if ((np.int32(1) * (num_groups_975 * group_size_974)) != np.int32(0)):
      self.map_kernel_881_var.set_args(mem_940, res_mem_948,
                                       np.int32(limit_736), mem_944, mem_946,
                                       np.int32(width_734), np.float32(res_742),
                                       mem_942, np.float32(view_738),
                                       np.float32(y_746),
                                       np.int32(nesting_size_879),
                                       np.byte(x_747), np.int32(height_735),
                                       mem_952)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_881_var,
                                 (long((num_groups_975 * group_size_974)),),
                                 (long(group_size_974),))
      if synchronous:
        self.queue.finish()
    x_955 = (width_734 * height_735)
    bytes_953 = (x_955 * np.int32(3))
    mem_956 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                        long(long(bytes_953) if (bytes_953 > np.int32(0)) else np.int32(1)))
    if ((((np.int32(1) * (height_735 + srem32((np.int32(16) - srem32(height_735,
                                                                     np.int32(16))),
                                              np.int32(16)))) * (np.int32(3) + srem32((np.int32(16) - srem32(np.int32(3),
                                                                                                             np.int32(16))),
                                                                                      np.int32(16)))) * width_734) != np.int32(0)):
      self.fut_kernel_map_transpose_i8_var.set_args(mem_956,
                                                    np.int32(np.int32(0)),
                                                    mem_952,
                                                    np.int32(np.int32(0)),
                                                    np.int32(height_735),
                                                    np.int32(np.int32(3)),
                                                    np.int32(((width_734 * height_735) * np.int32(3))),
                                                    cl.LocalMemory(long((((np.int32(16) + np.int32(1)) * np.int32(16)) * np.int32(1)))))
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.fut_kernel_map_transpose_i8_var,
                                 (long((height_735 + srem32((np.int32(16) - srem32(height_735,
                                                                                   np.int32(16))),
                                                            np.int32(16)))),
                                  long((np.int32(3) + srem32((np.int32(16) - srem32(np.int32(3),
                                                                                    np.int32(16))),
                                                             np.int32(16)))),
                                  long(width_734)), (long(np.int32(16)),
                                                     long(np.int32(16)),
                                                     long(np.int32(1))))
      if synchronous:
        self.queue.finish()
    out_mem_963 = mem_956
    out_memsize_964 = bytes_953
    return (out_memsize_964, out_mem_963)
  def main(self, width_734_ext, height_735_ext, limit_736_ext, view_737_ext,
           view_738_ext, view_739_ext, view_740_ext):
    width_734 = np.int32(width_734_ext)
    height_735 = np.int32(height_735_ext)
    limit_736 = np.int32(limit_736_ext)
    view_737 = np.float32(view_737_ext)
    view_738 = np.float32(view_738_ext)
    view_739 = np.float32(view_739_ext)
    view_740 = np.float32(view_740_ext)
    (out_memsize_964, out_mem_963) = self.futhark_main(width_734, height_735,
                                                       limit_736, view_737,
                                                       view_738, view_739,
                                                       view_740)
    return cl.array.Array(self.queue, (width_734, height_735, np.int32(3)),
                          ct.c_int8, data=out_mem_963)