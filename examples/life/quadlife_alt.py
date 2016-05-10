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
__kernel void map_kernel_1022(int32_t m_880, __global
                              unsigned char *world_mem_1109, int32_t n_879,
                              __global unsigned char *mem_1112)
{
    const uint kernel_thread_index_1022 = get_global_id(0);
    
    if (kernel_thread_index_1022 >= n_879 * m_880)
        return;
    
    int32_t i_1023;
    int32_t i_1024;
    char b_1025;
    
    // compute thread index
    {
        i_1023 = squot32(kernel_thread_index_1022, m_880);
        i_1024 = kernel_thread_index_1022 - squot32(kernel_thread_index_1022,
                                                    m_880) * m_880;
    }
    // read kernel parameters
    {
        b_1025 = *(__global char *) &world_mem_1109[i_1023 * m_880 + i_1024];
    }
    
    int8_t res_1026;
    
    if (b_1025) {
        res_1026 = 1;
    } else {
        res_1026 = 0;
    }
    // write kernel result
    {
        *(__global int8_t *) &mem_1112[i_1023 * m_880 + i_1024] = res_1026;
    }
}
__kernel void map_kernel_1176(int32_t m_880, __global unsigned char *mem_1114)
{
    const uint global_thread_index_1176 = get_global_id(0);
    
    if (global_thread_index_1176 >= m_880)
        return;
    
    int32_t i_1177;
    
    // compute thread index
    {
        i_1177 = global_thread_index_1176;
    }
    // read kernel parameters
    { }
    // write kernel result
    {
        *(__global int32_t *) &mem_1114[i_1177 * 4] = 0;
    }
}
__kernel void map_kernel_1180(int32_t m_880, __global unsigned char *mem_1114,
                              int32_t n_879, __global unsigned char *mem_1117)
{
    const uint global_thread_index_1180 = get_global_id(0);
    
    if (global_thread_index_1180 >= n_879 * m_880)
        return;
    
    int32_t i_1181;
    int32_t j_1182;
    int32_t input_1183;
    
    // compute thread index
    {
        i_1181 = squot32(global_thread_index_1180, m_880);
        j_1182 = global_thread_index_1180 - squot32(global_thread_index_1180,
                                                    m_880) * m_880;
    }
    // read kernel parameters
    {
        input_1183 = *(__global int32_t *) &mem_1114[j_1182 * 4];
    }
    // write kernel result
    {
        *(__global int32_t *) &mem_1117[(i_1181 * m_880 + j_1182) * 4] =
            input_1183;
    }
}
__kernel void map_kernel_1048(int32_t n_889, int32_t m_890, __global
                              unsigned char *mem_1130, __global
                              unsigned char *all_history_mem_1119, __global
                              unsigned char *mem_1133, __global
                              unsigned char *mem_1137)
{
    const uint kernel_thread_index_1048 = get_global_id(0);
    
    if (kernel_thread_index_1048 >= n_889 * m_890)
        return;
    
    int32_t i_1049;
    int32_t i_1050;
    int32_t not_curried_1051;
    
    // compute thread index
    {
        i_1049 = squot32(kernel_thread_index_1048, m_890);
        i_1050 = kernel_thread_index_1048 - squot32(kernel_thread_index_1048,
                                                    m_890) * m_890;
    }
    // read kernel parameters
    {
        not_curried_1051 = *(__global int32_t *) &all_history_mem_1119[(i_1049 *
                                                                        m_890 +
                                                                        i_1050) *
                                                                       4];
    }
    
    int32_t res_1052 = not_curried_1051 & 3;
    int32_t arg_1053 = ashr32(not_curried_1051, 2);
    char cond_1054 = slt32(255, arg_1053);
    int32_t res_1055;
    
    if (cond_1054) {
        res_1055 = 255;
    } else {
        res_1055 = arg_1053;
    }
    
    int8_t y_1057 = sext_i32_i8(res_1055);
    
    // write kernel result
    {
        *(__global int8_t *) &mem_1133[i_1049 * m_890 + i_1050] = y_1057;
        for (int i_1188 = 0; i_1188 < 3; i_1188++) {
            *(__global int8_t *) &mem_1137[3 * (m_890 * i_1049) + (m_890 *
                                                                   i_1188 +
                                                                   i_1050)] =
                *(__global int8_t *) &mem_1130[3 * res_1052 + i_1188];
        }
    }
}
__kernel void map_kernel_1037(__global unsigned char *mem_1137, int32_t n_889,
                              __global unsigned char *mem_1133, int32_t m_890,
                              __global unsigned char *mem_1141)
{
    const uint kernel_thread_index_1037 = get_global_id(0);
    
    if (kernel_thread_index_1037 >= n_889 * m_890 * 3)
        return;
    
    int32_t i_1038;
    int32_t i_1039;
    int32_t i_1040;
    int8_t y_1041;
    int8_t binop_param_noncurried_1042;
    
    // compute thread index
    {
        i_1038 = squot32(kernel_thread_index_1037, m_890 * 3);
        i_1039 = squot32(kernel_thread_index_1037 -
                         squot32(kernel_thread_index_1037, m_890 * 3) * (m_890 *
                                                                         3), 3);
        i_1040 = kernel_thread_index_1037 - squot32(kernel_thread_index_1037,
                                                    m_890 * 3) * (m_890 * 3) -
            squot32(kernel_thread_index_1037 - squot32(kernel_thread_index_1037,
                                                       m_890 * 3) * (m_890 * 3),
                    3) * 3;
    }
    // read kernel parameters
    {
        y_1041 = *(__global int8_t *) &mem_1133[i_1038 * m_890 + i_1039];
        binop_param_noncurried_1042 = *(__global int8_t *) &mem_1137[i_1038 *
                                                                     (3 *
                                                                      m_890) +
                                                                     i_1040 *
                                                                     m_890 +
                                                                     i_1039];
    }
    
    int8_t res_1043 = binop_param_noncurried_1042 - y_1041;
    
    // write kernel result
    {
        *(__global int8_t *) &mem_1141[i_1038 * (m_890 * 3) + i_1039 * 3 +
                                       i_1040] = res_1043;
    }
}
__kernel void map_kernel_1100(int32_t n_910, __global unsigned char *mem_1149,
                              __global unsigned char *mem_1151)
{
    const uint kernel_thread_index_1100 = get_global_id(0);
    
    if (kernel_thread_index_1100 >= n_910)
        return;
    
    int32_t i_1101;
    
    // compute thread index
    {
        i_1101 = kernel_thread_index_1100;
    }
    // read kernel parameters
    { }
    
    int32_t x_1103 = i_1101 - 1;
    int32_t res_1104 = smod32(x_1103, n_910);
    int32_t x_1105 = i_1101 + 1;
    int32_t res_1106 = smod32(x_1105, n_910);
    
    // write kernel result
    {
        *(__global int32_t *) &mem_1149[i_1101 * 4] = res_1106;
        *(__global int32_t *) &mem_1151[i_1101 * 4] = res_1104;
    }
}
__kernel void map_kernel_1064(__global unsigned char *mem_1149, __global
                              unsigned char *world_mem_1153, int32_t n_910,
                              __global unsigned char *mem_1151, int32_t m_911,
                              __global unsigned char *mem_1147, __global
                              unsigned char *history_mem_1155, __global
                              unsigned char *mem_1158, __global
                              unsigned char *mem_1161)
{
    const uint kernel_thread_index_1064 = get_global_id(0);
    
    if (kernel_thread_index_1064 >= n_910 * m_911)
        return;
    
    int32_t i_1065;
    int32_t i_1066;
    int32_t res_1068;
    int32_t res_1069;
    int32_t x_1070;
    
    // compute thread index
    {
        i_1065 = squot32(kernel_thread_index_1064, m_911);
        i_1066 = kernel_thread_index_1064 - squot32(kernel_thread_index_1064,
                                                    m_911) * m_911;
    }
    // read kernel parameters
    {
        res_1068 = *(__global int32_t *) &mem_1149[i_1065 * 4];
        res_1069 = *(__global int32_t *) &mem_1151[i_1065 * 4];
        x_1070 = *(__global int32_t *) &history_mem_1155[(i_1065 * m_911 +
                                                          i_1066) * 4];
    }
    
    int32_t x_1072 = i_1066 + 1;
    int32_t res_1073 = smod32(x_1072, m_911);
    int32_t x_1074 = i_1066 - 1;
    int32_t res_1075 = smod32(x_1074, m_911);
    int8_t x_1076 = *(__global int8_t *) &world_mem_1153[res_1069 * m_911 +
                                                         i_1066];
    int8_t y_1077 = *(__global int8_t *) &world_mem_1153[i_1065 * m_911 +
                                                         res_1075];
    int8_t x_1078 = x_1076 + y_1077;
    int8_t y_1079 = *(__global int8_t *) &world_mem_1153[i_1065 * m_911 +
                                                         i_1066];
    int8_t x_1080 = x_1078 + y_1079;
    int8_t y_1081 = *(__global int8_t *) &world_mem_1153[i_1065 * m_911 +
                                                         res_1073];
    int8_t x_1082 = x_1080 + y_1081;
    int8_t y_1083 = *(__global int8_t *) &world_mem_1153[res_1068 * m_911 +
                                                         i_1066];
    int8_t res_1084 = x_1082 + y_1083;
    int32_t i_1085 = sext_i8_i32(res_1084);
    int8_t res_1086 = *(__global int8_t *) &mem_1147[i_1085];
    int32_t res_1087 = x_1070 & 3;
    int32_t arg_1088 = ashr32(x_1070, 2);
    char cond_1089 = slt32(128, arg_1088);
    int32_t res_1090;
    
    if (cond_1089) {
        res_1090 = 128;
    } else {
        res_1090 = arg_1088;
    }
    
    int8_t y_1091 = sext_i32_i8(res_1087);
    char cond_1092 = res_1086 == y_1091;
    int32_t x_1093 = res_1090 + 1;
    int32_t x_1094 = x_1093 << 2;
    int32_t y_1095 = sext_i8_i32(res_1086);
    int32_t res_1096 = x_1094 | y_1095;
    int32_t res_1097;
    
    if (cond_1092) {
        res_1097 = res_1096;
    } else {
        res_1097 = y_1095;
    }
    // write kernel result
    {
        *(__global int32_t *) &mem_1158[(i_1065 * m_911 + i_1066) * 4] =
            res_1097;
        *(__global int8_t *) &mem_1161[i_1065 * m_911 + i_1066] = res_1086;
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
class quadlife_alt:
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
    
    self.map_kernel_1022_var = program.map_kernel_1022
    self.map_kernel_1176_var = program.map_kernel_1176
    self.map_kernel_1180_var = program.map_kernel_1180
    self.map_kernel_1048_var = program.map_kernel_1048
    self.map_kernel_1037_var = program.map_kernel_1037
    self.map_kernel_1100_var = program.map_kernel_1100
    self.map_kernel_1064_var = program.map_kernel_1064
  def futhark_init(self, world_mem_size_1108, world_mem_1109, n_879, m_880):
    nesting_size_1020 = (m_880 * n_879)
    bytes_1110 = (n_879 * m_880)
    mem_1112 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1110) if (bytes_1110 > np.int32(0)) else np.int32(1)))
    group_size_1174 = np.int32(512)
    num_groups_1175 = squot32((((n_879 * m_880) + group_size_1174) - np.int32(1)),
                              group_size_1174)
    if ((np.int32(1) * (num_groups_1175 * group_size_1174)) != np.int32(0)):
      self.map_kernel_1022_var.set_args(np.int32(m_880), world_mem_1109,
                                        np.int32(n_879), mem_1112)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1022_var,
                                 (long((num_groups_1175 * group_size_1174)),),
                                 (long(group_size_1174),))
      if synchronous:
        self.queue.finish()
    bytes_1113 = (np.int32(4) * m_880)
    mem_1114 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1113) if (bytes_1113 > np.int32(0)) else np.int32(1)))
    group_size_1178 = np.int32(512)
    num_groups_1179 = squot32(((m_880 + group_size_1178) - np.int32(1)),
                              group_size_1178)
    if ((np.int32(1) * (num_groups_1179 * group_size_1178)) != np.int32(0)):
      self.map_kernel_1176_var.set_args(np.int32(m_880), mem_1114)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1176_var,
                                 (long((num_groups_1179 * group_size_1178)),),
                                 (long(group_size_1178),))
      if synchronous:
        self.queue.finish()
    x_1116 = (np.int32(4) * n_879)
    bytes_1115 = (x_1116 * m_880)
    mem_1117 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1115) if (bytes_1115 > np.int32(0)) else np.int32(1)))
    group_size_1184 = np.int32(512)
    num_groups_1185 = squot32((((n_879 * m_880) + group_size_1184) - np.int32(1)),
                              group_size_1184)
    if ((np.int32(1) * (num_groups_1185 * group_size_1184)) != np.int32(0)):
      self.map_kernel_1180_var.set_args(np.int32(m_880), mem_1114,
                                        np.int32(n_879), mem_1117)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1180_var,
                                 (long((num_groups_1185 * group_size_1184)),),
                                 (long(group_size_1184),))
      if synchronous:
        self.queue.finish()
    out_mem_1170 = mem_1112
    out_memsize_1171 = bytes_1110
    out_mem_1172 = mem_1117
    out_memsize_1173 = bytes_1115
    return (out_memsize_1171, out_mem_1170, out_memsize_1173, out_mem_1172)
  def futhark_render_frame(self, all_history_mem_size_1118,
                           all_history_mem_1119, n_889, m_890):
    mem_1121 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(np.int32(3)) if (np.int32(3) > np.int32(0)) else np.int32(1)))
    cl.enqueue_copy(self.queue, mem_1121, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(0)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1121, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(1)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1121, np.array(np.int8(-1),
                                                   dtype=ct.c_int8),
                    device_offset=long(np.int32(2)), is_blocking=synchronous)
    mem_1123 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(np.int32(3)) if (np.int32(3) > np.int32(0)) else np.int32(1)))
    cl.enqueue_copy(self.queue, mem_1123, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(0)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1123, np.array(np.int8(-1),
                                                   dtype=ct.c_int8),
                    device_offset=long(np.int32(1)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1123, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(2)), is_blocking=synchronous)
    mem_1125 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(np.int32(3)) if (np.int32(3) > np.int32(0)) else np.int32(1)))
    cl.enqueue_copy(self.queue, mem_1125, np.array(np.int8(-1),
                                                   dtype=ct.c_int8),
                    device_offset=long(np.int32(0)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1125, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(1)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1125, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(2)), is_blocking=synchronous)
    mem_1127 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(np.int32(3)) if (np.int32(3) > np.int32(0)) else np.int32(1)))
    cl.enqueue_copy(self.queue, mem_1127, np.array(np.int8(-1),
                                                   dtype=ct.c_int8),
                    device_offset=long(np.int32(0)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1127, np.array(np.int8(-1),
                                                   dtype=ct.c_int8),
                    device_offset=long(np.int32(1)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1127, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(2)), is_blocking=synchronous)
    mem_1130 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(np.int32(12)) if (np.int32(12) > np.int32(0)) else np.int32(1)))
    if ((np.int32(3) * np.int32(1)) != np.int32(0)):
      cl.enqueue_copy(self.queue, mem_1130, mem_1121,
                      dest_offset=long(np.int32(0)),
                      src_offset=long(np.int32(0)),
                      byte_count=long((np.int32(3) * np.int32(1))))
    if synchronous:
      self.queue.finish()
    if ((np.int32(3) * np.int32(1)) != np.int32(0)):
      cl.enqueue_copy(self.queue, mem_1130, mem_1123,
                      dest_offset=long(np.int32(3)),
                      src_offset=long(np.int32(0)),
                      byte_count=long((np.int32(3) * np.int32(1))))
    if synchronous:
      self.queue.finish()
    if ((np.int32(3) * np.int32(1)) != np.int32(0)):
      cl.enqueue_copy(self.queue, mem_1130, mem_1125,
                      dest_offset=long((np.int32(3) * np.int32(2))),
                      src_offset=long(np.int32(0)),
                      byte_count=long((np.int32(3) * np.int32(1))))
    if synchronous:
      self.queue.finish()
    if ((np.int32(3) * np.int32(1)) != np.int32(0)):
      cl.enqueue_copy(self.queue, mem_1130, mem_1127,
                      dest_offset=long((np.int32(3) * np.int32(3))),
                      src_offset=long(np.int32(0)),
                      byte_count=long((np.int32(3) * np.int32(1))))
    if synchronous:
      self.queue.finish()
    nesting_size_1046 = (m_890 * n_889)
    bytes_1131 = (n_889 * m_890)
    mem_1133 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1131) if (bytes_1131 > np.int32(0)) else np.int32(1)))
    x_1136 = (n_889 * np.int32(3))
    bytes_1134 = (x_1136 * m_890)
    mem_1137 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1134) if (bytes_1134 > np.int32(0)) else np.int32(1)))
    group_size_1189 = np.int32(512)
    num_groups_1190 = squot32((((n_889 * m_890) + group_size_1189) - np.int32(1)),
                              group_size_1189)
    if ((np.int32(1) * (num_groups_1190 * group_size_1189)) != np.int32(0)):
      self.map_kernel_1048_var.set_args(np.int32(n_889), np.int32(m_890),
                                        mem_1130, all_history_mem_1119,
                                        mem_1133, mem_1137)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1048_var,
                                 (long((num_groups_1190 * group_size_1189)),),
                                 (long(group_size_1189),))
      if synchronous:
        self.queue.finish()
    nesting_size_1033 = (np.int32(3) * m_890)
    nesting_size_1035 = (nesting_size_1033 * n_889)
    bytes_1138 = (bytes_1131 * np.int32(3))
    mem_1141 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1138) if (bytes_1138 > np.int32(0)) else np.int32(1)))
    group_size_1191 = np.int32(512)
    num_groups_1192 = squot32(((((n_889 * m_890) * np.int32(3)) + group_size_1191) - np.int32(1)),
                              group_size_1191)
    if ((np.int32(1) * (num_groups_1192 * group_size_1191)) != np.int32(0)):
      self.map_kernel_1037_var.set_args(mem_1137, np.int32(n_889), mem_1133,
                                        np.int32(m_890), mem_1141)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1037_var,
                                 (long((num_groups_1192 * group_size_1191)),),
                                 (long(group_size_1191),))
      if synchronous:
        self.queue.finish()
    out_mem_1186 = mem_1141
    out_memsize_1187 = bytes_1138
    return (out_memsize_1187, out_mem_1186)
  def futhark_steps(self, world_mem_size_1142, history_mem_size_1144,
                    world_mem_1143, history_mem_1145, n_910, m_911, steps_914):
    mem_1147 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(np.int32(16)) if (np.int32(16) > np.int32(0)) else np.int32(1)))
    cl.enqueue_copy(self.queue, mem_1147, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(0)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1147, np.array(np.int8(1), dtype=ct.c_int8),
                    device_offset=long(np.int32(1)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1147, np.array(np.int8(1), dtype=ct.c_int8),
                    device_offset=long(np.int32(2)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1147, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(3)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1147, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(4)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1147, np.array(np.int8(1), dtype=ct.c_int8),
                    device_offset=long(np.int32(5)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1147, np.array(np.int8(1), dtype=ct.c_int8),
                    device_offset=long(np.int32(6)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1147, np.array(np.int8(1), dtype=ct.c_int8),
                    device_offset=long(np.int32(7)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1147, np.array(np.int8(2), dtype=ct.c_int8),
                    device_offset=long(np.int32(8)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1147, np.array(np.int8(2), dtype=ct.c_int8),
                    device_offset=long(np.int32(9)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1147, np.array(np.int8(2), dtype=ct.c_int8),
                    device_offset=long(np.int32(10)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1147, np.array(np.int8(3), dtype=ct.c_int8),
                    device_offset=long(np.int32(11)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1147, np.array(np.int8(3), dtype=ct.c_int8),
                    device_offset=long(np.int32(12)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1147, np.array(np.int8(2), dtype=ct.c_int8),
                    device_offset=long(np.int32(13)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1147, np.array(np.int8(2), dtype=ct.c_int8),
                    device_offset=long(np.int32(14)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1147, np.array(np.int8(3), dtype=ct.c_int8),
                    device_offset=long(np.int32(15)), is_blocking=synchronous)
    bytes_1148 = (np.int32(4) * n_910)
    mem_1149 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1148) if (bytes_1148 > np.int32(0)) else np.int32(1)))
    mem_1151 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1148) if (bytes_1148 > np.int32(0)) else np.int32(1)))
    group_size_1197 = np.int32(512)
    num_groups_1198 = squot32(((n_910 + group_size_1197) - np.int32(1)),
                              group_size_1197)
    if ((np.int32(1) * (num_groups_1198 * group_size_1197)) != np.int32(0)):
      self.map_kernel_1100_var.set_args(np.int32(n_910), mem_1149, mem_1151)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1100_var,
                                 (long((num_groups_1198 * group_size_1197)),),
                                 (long(group_size_1197),))
      if synchronous:
        self.queue.finish()
    nesting_size_1062 = (m_911 * n_910)
    bytes_1156 = (bytes_1148 * m_911)
    bytes_1159 = (n_910 * m_911)
    double_buffer_mem_1166 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                       long(long(bytes_1159) if (bytes_1159 > np.int32(0)) else np.int32(1)))
    double_buffer_mem_1167 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                       long(long(bytes_1156) if (bytes_1156 > np.int32(0)) else np.int32(1)))
    mem_1158 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1156) if (bytes_1156 > np.int32(0)) else np.int32(1)))
    mem_1161 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1159) if (bytes_1159 > np.int32(0)) else np.int32(1)))
    world_mem_size_1152 = world_mem_size_1142
    history_mem_size_1154 = history_mem_size_1144
    world_mem_1153 = world_mem_1143
    history_mem_1155 = history_mem_1145
    i_920 = np.int32(0)
    one_1208 = np.int32(1)
    for counter_1207 in range(steps_914):
      group_size_1205 = np.int32(512)
      num_groups_1206 = squot32((((n_910 * m_911) + group_size_1205) - np.int32(1)),
                                group_size_1205)
      if ((np.int32(1) * (num_groups_1206 * group_size_1205)) != np.int32(0)):
        self.map_kernel_1064_var.set_args(mem_1149, world_mem_1153,
                                          np.int32(n_910), mem_1151,
                                          np.int32(m_911), mem_1147,
                                          history_mem_1155, mem_1158, mem_1161)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1064_var,
                                   (long((num_groups_1206 * group_size_1205)),),
                                   (long(group_size_1205),))
        if synchronous:
          self.queue.finish()
      if (((n_910 * m_911) * np.int32(1)) != np.int32(0)):
        cl.enqueue_copy(self.queue, double_buffer_mem_1166, mem_1161,
                        dest_offset=long(np.int32(0)),
                        src_offset=long(np.int32(0)),
                        byte_count=long(((n_910 * m_911) * np.int32(1))))
      if synchronous:
        self.queue.finish()
      if (((n_910 * m_911) * np.int32(4)) != np.int32(0)):
        cl.enqueue_copy(self.queue, double_buffer_mem_1167, mem_1158,
                        dest_offset=long(np.int32(0)),
                        src_offset=long(np.int32(0)),
                        byte_count=long(((n_910 * m_911) * np.int32(4))))
      if synchronous:
        self.queue.finish()
      world_mem_size_tmp_1199 = bytes_1159
      history_mem_size_tmp_1200 = bytes_1156
      world_mem_tmp_1201 = double_buffer_mem_1166
      history_mem_tmp_1202 = double_buffer_mem_1167
      world_mem_size_1152 = world_mem_size_tmp_1199
      history_mem_size_1154 = history_mem_size_tmp_1200
      world_mem_1153 = world_mem_tmp_1201
      history_mem_1155 = history_mem_tmp_1202
      i_920 += one_1208
    world_mem_1163 = world_mem_1153
    world_mem_size_1162 = world_mem_size_1152
    history_mem_1165 = history_mem_1155
    history_mem_size_1164 = history_mem_size_1154
    out_mem_1193 = world_mem_1163
    out_memsize_1194 = world_mem_size_1162
    out_mem_1195 = history_mem_1165
    out_memsize_1196 = history_mem_size_1164
    return (out_memsize_1194, out_mem_1193, out_memsize_1196, out_mem_1195)
  def init(self, world_mem_1109_ext):
    n_879 = np.int32(world_mem_1109_ext.shape[np.int32(0)])
    m_880 = np.int32(world_mem_1109_ext.shape[np.int32(1)])
    world_mem_size_1108 = np.int32(world_mem_1109_ext.nbytes)
    if (type(world_mem_1109_ext) == cl.array.Array):
      world_mem_1109 = world_mem_1109_ext.data
    else:
      world_mem_1109 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                 long(long(world_mem_size_1108) if (world_mem_size_1108 > np.int32(0)) else np.int32(1)))
      if (world_mem_size_1108 != np.int32(0)):
        cl.enqueue_copy(self.queue, world_mem_1109, world_mem_1109_ext,
                        is_blocking=synchronous)
    (out_memsize_1171, out_mem_1170, out_memsize_1173,
     out_mem_1172) = self.futhark_init(world_mem_size_1108, world_mem_1109,
                                       n_879, m_880)
    return (cl.array.Array(self.queue, (n_879, m_880), ct.c_int8,
                           data=out_mem_1170), cl.array.Array(self.queue,
                                                              (n_879, m_880),
                                                              ct.c_int32,
                                                              data=out_mem_1172))
  def render_frame(self, all_history_mem_1119_ext):
    n_889 = np.int32(all_history_mem_1119_ext.shape[np.int32(0)])
    m_890 = np.int32(all_history_mem_1119_ext.shape[np.int32(1)])
    all_history_mem_size_1118 = np.int32(all_history_mem_1119_ext.nbytes)
    if (type(all_history_mem_1119_ext) == cl.array.Array):
      all_history_mem_1119 = all_history_mem_1119_ext.data
    else:
      all_history_mem_1119 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                       long(long(all_history_mem_size_1118) if (all_history_mem_size_1118 > np.int32(0)) else np.int32(1)))
      if (all_history_mem_size_1118 != np.int32(0)):
        cl.enqueue_copy(self.queue, all_history_mem_1119,
                        all_history_mem_1119_ext, is_blocking=synchronous)
    (out_memsize_1187,
     out_mem_1186) = self.futhark_render_frame(all_history_mem_size_1118,
                                               all_history_mem_1119, n_889,
                                               m_890)
    return cl.array.Array(self.queue, (n_889, m_890, np.int32(3)), ct.c_int8,
                          data=out_mem_1186)
  def steps(self, world_mem_1143_ext, history_mem_1145_ext, steps_914_ext):
    n_910 = np.int32(world_mem_1143_ext.shape[np.int32(0)])
    m_911 = np.int32(world_mem_1143_ext.shape[np.int32(1)])
    world_mem_size_1142 = np.int32(world_mem_1143_ext.nbytes)
    if (type(world_mem_1143_ext) == cl.array.Array):
      world_mem_1143 = world_mem_1143_ext.data
    else:
      world_mem_1143 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                 long(long(world_mem_size_1142) if (world_mem_size_1142 > np.int32(0)) else np.int32(1)))
      if (world_mem_size_1142 != np.int32(0)):
        cl.enqueue_copy(self.queue, world_mem_1143, world_mem_1143_ext,
                        is_blocking=synchronous)
    n_910 = np.int32(history_mem_1145_ext.shape[np.int32(0)])
    m_911 = np.int32(history_mem_1145_ext.shape[np.int32(1)])
    history_mem_size_1144 = np.int32(history_mem_1145_ext.nbytes)
    if (type(history_mem_1145_ext) == cl.array.Array):
      history_mem_1145 = history_mem_1145_ext.data
    else:
      history_mem_1145 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                   long(long(history_mem_size_1144) if (history_mem_size_1144 > np.int32(0)) else np.int32(1)))
      if (history_mem_size_1144 != np.int32(0)):
        cl.enqueue_copy(self.queue, history_mem_1145, history_mem_1145_ext,
                        is_blocking=synchronous)
    steps_914 = np.int32(steps_914_ext)
    (out_memsize_1194, out_mem_1193, out_memsize_1196,
     out_mem_1195) = self.futhark_steps(world_mem_size_1142,
                                        history_mem_size_1144, world_mem_1143,
                                        history_mem_1145, n_910, m_911,
                                        steps_914)
    return (cl.array.Array(self.queue, (n_910, m_911), ct.c_int8,
                           data=out_mem_1193), cl.array.Array(self.queue,
                                                              (n_910, m_911),
                                                              ct.c_int32,
                                                              data=out_mem_1195))