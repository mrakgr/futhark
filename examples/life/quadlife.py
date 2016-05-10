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
__kernel void map_kernel_1121(int32_t n_953, __global
                              unsigned char *world_mem_1217, int32_t m_954,
                              __global unsigned char *mem_1220)
{
    const uint kernel_thread_index_1121 = get_global_id(0);
    
    if (kernel_thread_index_1121 >= n_953 * m_954)
        return;
    
    int32_t i_1122;
    int32_t i_1123;
    char b_1124;
    
    // compute thread index
    {
        i_1122 = squot32(kernel_thread_index_1121, m_954);
        i_1123 = kernel_thread_index_1121 - squot32(kernel_thread_index_1121,
                                                    m_954) * m_954;
    }
    // read kernel parameters
    {
        b_1124 = *(__global char *) &world_mem_1217[i_1122 * m_954 + i_1123];
    }
    
    int8_t res_1125;
    
    if (b_1124) {
        res_1125 = 1;
    } else {
        res_1125 = 0;
    }
    // write kernel result
    {
        *(__global int8_t *) &mem_1220[i_1122 * m_954 + i_1123] = res_1125;
    }
}
__kernel void map_kernel_1284(int32_t m_954, __global unsigned char *mem_1222)
{
    const uint global_thread_index_1284 = get_global_id(0);
    
    if (global_thread_index_1284 >= m_954)
        return;
    
    int32_t i_1285;
    
    // compute thread index
    {
        i_1285 = global_thread_index_1284;
    }
    // read kernel parameters
    { }
    // write kernel result
    {
        *(__global int32_t *) &mem_1222[i_1285 * 4] = 1020;
    }
}
__kernel void map_kernel_1288(int32_t n_953, int32_t m_954, __global
                              unsigned char *mem_1222, __global
                              unsigned char *mem_1225)
{
    const uint global_thread_index_1288 = get_global_id(0);
    
    if (global_thread_index_1288 >= n_953 * m_954)
        return;
    
    int32_t i_1289;
    int32_t j_1290;
    int32_t input_1291;
    
    // compute thread index
    {
        i_1289 = squot32(global_thread_index_1288, m_954);
        j_1290 = global_thread_index_1288 - squot32(global_thread_index_1288,
                                                    m_954) * m_954;
    }
    // read kernel parameters
    {
        input_1291 = *(__global int32_t *) &mem_1222[j_1290 * 4];
    }
    // write kernel result
    {
        *(__global int32_t *) &mem_1225[(i_1289 * m_954 + j_1290) * 4] =
            input_1291;
    }
}
__kernel void map_kernel_1147(int32_t m_964, __global unsigned char *mem_1238,
                              __global unsigned char *all_history_mem_1227,
                              int32_t n_963, __global unsigned char *mem_1241,
                              __global unsigned char *mem_1245)
{
    const uint kernel_thread_index_1147 = get_global_id(0);
    
    if (kernel_thread_index_1147 >= n_963 * m_964)
        return;
    
    int32_t i_1148;
    int32_t i_1149;
    int32_t not_curried_1150;
    
    // compute thread index
    {
        i_1148 = squot32(kernel_thread_index_1147, m_964);
        i_1149 = kernel_thread_index_1147 - squot32(kernel_thread_index_1147,
                                                    m_964) * m_964;
    }
    // read kernel parameters
    {
        not_curried_1150 = *(__global int32_t *) &all_history_mem_1227[(i_1148 *
                                                                        m_964 +
                                                                        i_1149) *
                                                                       4];
    }
    
    int32_t res_1151 = not_curried_1150 & 3;
    int32_t res_1152 = ashr32(not_curried_1150, 2);
    int8_t y_1154 = sext_i32_i8(res_1152);
    
    // write kernel result
    {
        *(__global int8_t *) &mem_1241[i_1148 * m_964 + i_1149] = y_1154;
        for (int i_1296 = 0; i_1296 < 3; i_1296++) {
            *(__global int8_t *) &mem_1245[3 * (m_964 * i_1148) + (m_964 *
                                                                   i_1296 +
                                                                   i_1149)] =
                *(__global int8_t *) &mem_1238[3 * res_1151 + i_1296];
        }
    }
}
__kernel void map_kernel_1136(int32_t m_964, __global unsigned char *mem_1241,
                              __global unsigned char *mem_1245, int32_t n_963,
                              __global unsigned char *mem_1249)
{
    const uint kernel_thread_index_1136 = get_global_id(0);
    
    if (kernel_thread_index_1136 >= n_963 * m_964 * 3)
        return;
    
    int32_t i_1137;
    int32_t i_1138;
    int32_t i_1139;
    int8_t y_1140;
    int8_t binop_param_noncurried_1141;
    
    // compute thread index
    {
        i_1137 = squot32(kernel_thread_index_1136, m_964 * 3);
        i_1138 = squot32(kernel_thread_index_1136 -
                         squot32(kernel_thread_index_1136, m_964 * 3) * (m_964 *
                                                                         3), 3);
        i_1139 = kernel_thread_index_1136 - squot32(kernel_thread_index_1136,
                                                    m_964 * 3) * (m_964 * 3) -
            squot32(kernel_thread_index_1136 - squot32(kernel_thread_index_1136,
                                                       m_964 * 3) * (m_964 * 3),
                    3) * 3;
    }
    // read kernel parameters
    {
        y_1140 = *(__global int8_t *) &mem_1241[i_1137 * m_964 + i_1138];
        binop_param_noncurried_1141 = *(__global int8_t *) &mem_1245[i_1137 *
                                                                     (3 *
                                                                      m_964) +
                                                                     i_1139 *
                                                                     m_964 +
                                                                     i_1138];
    }
    
    int8_t res_1142 = binop_param_noncurried_1141 - y_1140;
    
    // write kernel result
    {
        *(__global int8_t *) &mem_1249[i_1137 * (m_964 * 3) + i_1138 * 3 +
                                       i_1139] = res_1142;
    }
}
__kernel void map_kernel_1208(int32_t n_982, __global unsigned char *mem_1257,
                              __global unsigned char *mem_1259)
{
    const uint kernel_thread_index_1208 = get_global_id(0);
    
    if (kernel_thread_index_1208 >= n_982)
        return;
    
    int32_t i_1209;
    
    // compute thread index
    {
        i_1209 = kernel_thread_index_1208;
    }
    // read kernel parameters
    { }
    
    int32_t x_1211 = i_1209 - 1;
    int32_t res_1212 = smod32(x_1211, n_982);
    int32_t x_1213 = i_1209 + 1;
    int32_t res_1214 = smod32(x_1213, n_982);
    
    // write kernel result
    {
        *(__global int32_t *) &mem_1257[i_1209 * 4] = res_1214;
        *(__global int32_t *) &mem_1259[i_1209 * 4] = res_1212;
    }
}
__kernel void map_kernel_1161(__global unsigned char *mem_1257, __global
                              unsigned char *world_mem_1261, int32_t n_982,
                              __global unsigned char *history_mem_1263, __global
                              unsigned char *mem_1259, int32_t m_983, __global
                              unsigned char *mem_1255, __global
                              unsigned char *mem_1266, __global
                              unsigned char *mem_1269)
{
    const uint kernel_thread_index_1161 = get_global_id(0);
    
    if (kernel_thread_index_1161 >= n_982 * m_983)
        return;
    
    int32_t i_1162;
    int32_t i_1163;
    int32_t res_1165;
    int32_t res_1166;
    int32_t x_1168;
    
    // compute thread index
    {
        i_1162 = squot32(kernel_thread_index_1161, m_983);
        i_1163 = kernel_thread_index_1161 - squot32(kernel_thread_index_1161,
                                                    m_983) * m_983;
    }
    // read kernel parameters
    {
        res_1165 = *(__global int32_t *) &mem_1257[i_1162 * 4];
        res_1166 = *(__global int32_t *) &mem_1259[i_1162 * 4];
        x_1168 = *(__global int32_t *) &history_mem_1263[(i_1162 * m_983 +
                                                          i_1163) * 4];
    }
    
    int32_t x_1169 = i_1163 + 1;
    int32_t res_1170 = smod32(x_1169, m_983);
    int32_t x_1171 = i_1163 - 1;
    int32_t res_1172 = smod32(x_1171, m_983);
    int8_t x_1173 = *(__global int8_t *) &world_mem_1261[res_1166 * m_983 +
                                                         res_1172];
    int8_t y_1174 = *(__global int8_t *) &world_mem_1261[res_1166 * m_983 +
                                                         i_1163];
    int8_t x_1175 = x_1173 + y_1174;
    int8_t y_1176 = *(__global int8_t *) &world_mem_1261[res_1166 * m_983 +
                                                         res_1170];
    int8_t x_1177 = x_1175 + y_1176;
    int8_t y_1178 = *(__global int8_t *) &world_mem_1261[i_1162 * m_983 +
                                                         res_1172];
    int8_t x_1179 = x_1177 + y_1178;
    int8_t y_1180 = *(__global int8_t *) &world_mem_1261[i_1162 * m_983 +
                                                         i_1163];
    int8_t x_1181 = x_1179 + y_1180;
    int8_t y_1182 = *(__global int8_t *) &world_mem_1261[i_1162 * m_983 +
                                                         res_1170];
    int8_t x_1183 = x_1181 + y_1182;
    int8_t y_1184 = *(__global int8_t *) &world_mem_1261[res_1165 * m_983 +
                                                         res_1172];
    int8_t x_1185 = x_1183 + y_1184;
    int8_t y_1186 = *(__global int8_t *) &world_mem_1261[res_1165 * m_983 +
                                                         i_1163];
    int8_t x_1187 = x_1185 + y_1186;
    int8_t y_1188 = *(__global int8_t *) &world_mem_1261[res_1165 * m_983 +
                                                         res_1170];
    int8_t res_1189 = x_1187 + y_1188;
    int32_t i_1190 = sext_i8_i32(res_1189);
    int8_t res_1191 = *(__global int8_t *) &mem_1255[i_1190];
    int32_t res_1192 = x_1168 & 3;
    int32_t res_1193 = ashr32(x_1168, 2);
    char cond_1194 = res_1191 == 1;
    char condtrue_1195 = res_1193 == 0;
    char cond_1196 = !condtrue_1195;
    int32_t arg_1197 = res_1193 + 1;
    int32_t res_1198 = 512 | res_1192;
    int32_t res_1199 = sext_i8_i32(res_1191);
    char cond_1200 = slt32(arg_1197, 255);
    int32_t res_1201;
    
    if (cond_1200) {
        res_1201 = arg_1197;
    } else {
        res_1201 = 255;
    }
    
    int32_t x_1202 = res_1201 << 2;
    int32_t res_1203 = x_1202 | res_1192;
    int32_t res_1204;
    
    if (cond_1196) {
        res_1204 = res_1203;
    } else {
        res_1204 = res_1198;
    }
    
    int32_t res_1205;
    
    if (cond_1194) {
        res_1205 = res_1204;
    } else {
        res_1205 = res_1199;
    }
    // write kernel result
    {
        *(__global int32_t *) &mem_1266[(i_1162 * m_983 + i_1163) * 4] =
            res_1205;
        *(__global int8_t *) &mem_1269[i_1162 * m_983 + i_1163] = res_1191;
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
class quadlife:
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
    
    self.map_kernel_1121_var = program.map_kernel_1121
    self.map_kernel_1284_var = program.map_kernel_1284
    self.map_kernel_1288_var = program.map_kernel_1288
    self.map_kernel_1147_var = program.map_kernel_1147
    self.map_kernel_1136_var = program.map_kernel_1136
    self.map_kernel_1208_var = program.map_kernel_1208
    self.map_kernel_1161_var = program.map_kernel_1161
  def futhark_init(self, world_mem_size_1216, world_mem_1217, n_953, m_954):
    nesting_size_1119 = (m_954 * n_953)
    bytes_1218 = (n_953 * m_954)
    mem_1220 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1218) if (bytes_1218 > np.int32(0)) else np.int32(1)))
    group_size_1282 = np.int32(512)
    num_groups_1283 = squot32((((n_953 * m_954) + group_size_1282) - np.int32(1)),
                              group_size_1282)
    if ((np.int32(1) * (num_groups_1283 * group_size_1282)) != np.int32(0)):
      self.map_kernel_1121_var.set_args(np.int32(n_953), world_mem_1217,
                                        np.int32(m_954), mem_1220)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1121_var,
                                 (long((num_groups_1283 * group_size_1282)),),
                                 (long(group_size_1282),))
      if synchronous:
        self.queue.finish()
    bytes_1221 = (np.int32(4) * m_954)
    mem_1222 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1221) if (bytes_1221 > np.int32(0)) else np.int32(1)))
    group_size_1286 = np.int32(512)
    num_groups_1287 = squot32(((m_954 + group_size_1286) - np.int32(1)),
                              group_size_1286)
    if ((np.int32(1) * (num_groups_1287 * group_size_1286)) != np.int32(0)):
      self.map_kernel_1284_var.set_args(np.int32(m_954), mem_1222)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1284_var,
                                 (long((num_groups_1287 * group_size_1286)),),
                                 (long(group_size_1286),))
      if synchronous:
        self.queue.finish()
    x_1224 = (np.int32(4) * n_953)
    bytes_1223 = (x_1224 * m_954)
    mem_1225 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1223) if (bytes_1223 > np.int32(0)) else np.int32(1)))
    group_size_1292 = np.int32(512)
    num_groups_1293 = squot32((((n_953 * m_954) + group_size_1292) - np.int32(1)),
                              group_size_1292)
    if ((np.int32(1) * (num_groups_1293 * group_size_1292)) != np.int32(0)):
      self.map_kernel_1288_var.set_args(np.int32(n_953), np.int32(m_954),
                                        mem_1222, mem_1225)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1288_var,
                                 (long((num_groups_1293 * group_size_1292)),),
                                 (long(group_size_1292),))
      if synchronous:
        self.queue.finish()
    out_mem_1278 = mem_1220
    out_memsize_1279 = bytes_1218
    out_mem_1280 = mem_1225
    out_memsize_1281 = bytes_1223
    return (out_memsize_1279, out_mem_1278, out_memsize_1281, out_mem_1280)
  def futhark_render_frame(self, all_history_mem_size_1226,
                           all_history_mem_1227, n_963, m_964):
    mem_1229 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(np.int32(3)) if (np.int32(3) > np.int32(0)) else np.int32(1)))
    cl.enqueue_copy(self.queue, mem_1229, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(0)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1229, np.array(np.int8(-1),
                                                   dtype=ct.c_int8),
                    device_offset=long(np.int32(1)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1229, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(2)), is_blocking=synchronous)
    mem_1231 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(np.int32(3)) if (np.int32(3) > np.int32(0)) else np.int32(1)))
    cl.enqueue_copy(self.queue, mem_1231, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(0)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1231, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(1)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1231, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(2)), is_blocking=synchronous)
    mem_1233 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(np.int32(3)) if (np.int32(3) > np.int32(0)) else np.int32(1)))
    cl.enqueue_copy(self.queue, mem_1233, np.array(np.int8(-1),
                                                   dtype=ct.c_int8),
                    device_offset=long(np.int32(0)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1233, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(1)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1233, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(2)), is_blocking=synchronous)
    mem_1235 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(np.int32(3)) if (np.int32(3) > np.int32(0)) else np.int32(1)))
    cl.enqueue_copy(self.queue, mem_1235, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(0)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1235, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(1)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1235, np.array(np.int8(-1),
                                                   dtype=ct.c_int8),
                    device_offset=long(np.int32(2)), is_blocking=synchronous)
    mem_1238 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(np.int32(12)) if (np.int32(12) > np.int32(0)) else np.int32(1)))
    if ((np.int32(3) * np.int32(1)) != np.int32(0)):
      cl.enqueue_copy(self.queue, mem_1238, mem_1229,
                      dest_offset=long(np.int32(0)),
                      src_offset=long(np.int32(0)),
                      byte_count=long((np.int32(3) * np.int32(1))))
    if synchronous:
      self.queue.finish()
    if ((np.int32(3) * np.int32(1)) != np.int32(0)):
      cl.enqueue_copy(self.queue, mem_1238, mem_1231,
                      dest_offset=long(np.int32(3)),
                      src_offset=long(np.int32(0)),
                      byte_count=long((np.int32(3) * np.int32(1))))
    if synchronous:
      self.queue.finish()
    if ((np.int32(3) * np.int32(1)) != np.int32(0)):
      cl.enqueue_copy(self.queue, mem_1238, mem_1233,
                      dest_offset=long((np.int32(3) * np.int32(2))),
                      src_offset=long(np.int32(0)),
                      byte_count=long((np.int32(3) * np.int32(1))))
    if synchronous:
      self.queue.finish()
    if ((np.int32(3) * np.int32(1)) != np.int32(0)):
      cl.enqueue_copy(self.queue, mem_1238, mem_1235,
                      dest_offset=long((np.int32(3) * np.int32(3))),
                      src_offset=long(np.int32(0)),
                      byte_count=long((np.int32(3) * np.int32(1))))
    if synchronous:
      self.queue.finish()
    nesting_size_1145 = (m_964 * n_963)
    bytes_1239 = (n_963 * m_964)
    mem_1241 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1239) if (bytes_1239 > np.int32(0)) else np.int32(1)))
    x_1244 = (n_963 * np.int32(3))
    bytes_1242 = (x_1244 * m_964)
    mem_1245 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1242) if (bytes_1242 > np.int32(0)) else np.int32(1)))
    group_size_1297 = np.int32(512)
    num_groups_1298 = squot32((((n_963 * m_964) + group_size_1297) - np.int32(1)),
                              group_size_1297)
    if ((np.int32(1) * (num_groups_1298 * group_size_1297)) != np.int32(0)):
      self.map_kernel_1147_var.set_args(np.int32(m_964), mem_1238,
                                        all_history_mem_1227, np.int32(n_963),
                                        mem_1241, mem_1245)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1147_var,
                                 (long((num_groups_1298 * group_size_1297)),),
                                 (long(group_size_1297),))
      if synchronous:
        self.queue.finish()
    nesting_size_1132 = (np.int32(3) * m_964)
    nesting_size_1134 = (nesting_size_1132 * n_963)
    bytes_1246 = (bytes_1239 * np.int32(3))
    mem_1249 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1246) if (bytes_1246 > np.int32(0)) else np.int32(1)))
    group_size_1299 = np.int32(512)
    num_groups_1300 = squot32(((((n_963 * m_964) * np.int32(3)) + group_size_1299) - np.int32(1)),
                              group_size_1299)
    if ((np.int32(1) * (num_groups_1300 * group_size_1299)) != np.int32(0)):
      self.map_kernel_1136_var.set_args(np.int32(m_964), mem_1241, mem_1245,
                                        np.int32(n_963), mem_1249)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1136_var,
                                 (long((num_groups_1300 * group_size_1299)),),
                                 (long(group_size_1299),))
      if synchronous:
        self.queue.finish()
    out_mem_1294 = mem_1249
    out_memsize_1295 = bytes_1246
    return (out_memsize_1295, out_mem_1294)
  def futhark_steps(self, world_mem_size_1250, history_mem_size_1252,
                    world_mem_1251, history_mem_1253, n_982, m_983, steps_986):
    mem_1255 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(np.int32(28)) if (np.int32(28) > np.int32(0)) else np.int32(1)))
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(0)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(1)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(2)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(2), dtype=ct.c_int8),
                    device_offset=long(np.int32(3)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(2), dtype=ct.c_int8),
                    device_offset=long(np.int32(4)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(3), dtype=ct.c_int8),
                    device_offset=long(np.int32(5)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(3), dtype=ct.c_int8),
                    device_offset=long(np.int32(6)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(1), dtype=ct.c_int8),
                    device_offset=long(np.int32(7)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(1), dtype=ct.c_int8),
                    device_offset=long(np.int32(8)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(1), dtype=ct.c_int8),
                    device_offset=long(np.int32(9)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(1), dtype=ct.c_int8),
                    device_offset=long(np.int32(10)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(1), dtype=ct.c_int8),
                    device_offset=long(np.int32(11)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(12)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(13)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(2), dtype=ct.c_int8),
                    device_offset=long(np.int32(14)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(15)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(2), dtype=ct.c_int8),
                    device_offset=long(np.int32(16)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(2), dtype=ct.c_int8),
                    device_offset=long(np.int32(17)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(2), dtype=ct.c_int8),
                    device_offset=long(np.int32(18)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(2), dtype=ct.c_int8),
                    device_offset=long(np.int32(19)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(2), dtype=ct.c_int8),
                    device_offset=long(np.int32(20)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(0), dtype=ct.c_int8),
                    device_offset=long(np.int32(21)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(2), dtype=ct.c_int8),
                    device_offset=long(np.int32(22)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(1), dtype=ct.c_int8),
                    device_offset=long(np.int32(23)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(2), dtype=ct.c_int8),
                    device_offset=long(np.int32(24)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(3), dtype=ct.c_int8),
                    device_offset=long(np.int32(25)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(3), dtype=ct.c_int8),
                    device_offset=long(np.int32(26)), is_blocking=synchronous)
    cl.enqueue_copy(self.queue, mem_1255, np.array(np.int8(3), dtype=ct.c_int8),
                    device_offset=long(np.int32(27)), is_blocking=synchronous)
    bytes_1256 = (np.int32(4) * n_982)
    mem_1257 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1256) if (bytes_1256 > np.int32(0)) else np.int32(1)))
    mem_1259 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1256) if (bytes_1256 > np.int32(0)) else np.int32(1)))
    group_size_1305 = np.int32(512)
    num_groups_1306 = squot32(((n_982 + group_size_1305) - np.int32(1)),
                              group_size_1305)
    if ((np.int32(1) * (num_groups_1306 * group_size_1305)) != np.int32(0)):
      self.map_kernel_1208_var.set_args(np.int32(n_982), mem_1257, mem_1259)
      cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1208_var,
                                 (long((num_groups_1306 * group_size_1305)),),
                                 (long(group_size_1305),))
      if synchronous:
        self.queue.finish()
    nesting_size_1159 = (m_983 * n_982)
    bytes_1264 = (bytes_1256 * m_983)
    bytes_1267 = (n_982 * m_983)
    double_buffer_mem_1274 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                       long(long(bytes_1267) if (bytes_1267 > np.int32(0)) else np.int32(1)))
    double_buffer_mem_1275 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                       long(long(bytes_1264) if (bytes_1264 > np.int32(0)) else np.int32(1)))
    mem_1266 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1264) if (bytes_1264 > np.int32(0)) else np.int32(1)))
    mem_1269 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                         long(long(bytes_1267) if (bytes_1267 > np.int32(0)) else np.int32(1)))
    world_mem_size_1260 = world_mem_size_1250
    history_mem_size_1262 = history_mem_size_1252
    world_mem_1261 = world_mem_1251
    history_mem_1263 = history_mem_1253
    i_992 = np.int32(0)
    one_1316 = np.int32(1)
    for counter_1315 in range(steps_986):
      group_size_1313 = np.int32(512)
      num_groups_1314 = squot32((((n_982 * m_983) + group_size_1313) - np.int32(1)),
                                group_size_1313)
      if ((np.int32(1) * (num_groups_1314 * group_size_1313)) != np.int32(0)):
        self.map_kernel_1161_var.set_args(mem_1257, world_mem_1261,
                                          np.int32(n_982), history_mem_1263,
                                          mem_1259, np.int32(m_983), mem_1255,
                                          mem_1266, mem_1269)
        cl.enqueue_nd_range_kernel(self.queue, self.map_kernel_1161_var,
                                   (long((num_groups_1314 * group_size_1313)),),
                                   (long(group_size_1313),))
        if synchronous:
          self.queue.finish()
      if (((n_982 * m_983) * np.int32(1)) != np.int32(0)):
        cl.enqueue_copy(self.queue, double_buffer_mem_1274, mem_1269,
                        dest_offset=long(np.int32(0)),
                        src_offset=long(np.int32(0)),
                        byte_count=long(((n_982 * m_983) * np.int32(1))))
      if synchronous:
        self.queue.finish()
      if (((n_982 * m_983) * np.int32(4)) != np.int32(0)):
        cl.enqueue_copy(self.queue, double_buffer_mem_1275, mem_1266,
                        dest_offset=long(np.int32(0)),
                        src_offset=long(np.int32(0)),
                        byte_count=long(((n_982 * m_983) * np.int32(4))))
      if synchronous:
        self.queue.finish()
      world_mem_size_tmp_1307 = bytes_1267
      history_mem_size_tmp_1308 = bytes_1264
      world_mem_tmp_1309 = double_buffer_mem_1274
      history_mem_tmp_1310 = double_buffer_mem_1275
      world_mem_size_1260 = world_mem_size_tmp_1307
      history_mem_size_1262 = history_mem_size_tmp_1308
      world_mem_1261 = world_mem_tmp_1309
      history_mem_1263 = history_mem_tmp_1310
      i_992 += one_1316
    world_mem_1271 = world_mem_1261
    world_mem_size_1270 = world_mem_size_1260
    history_mem_1273 = history_mem_1263
    history_mem_size_1272 = history_mem_size_1262
    out_mem_1301 = world_mem_1271
    out_memsize_1302 = world_mem_size_1270
    out_mem_1303 = history_mem_1273
    out_memsize_1304 = history_mem_size_1272
    return (out_memsize_1302, out_mem_1301, out_memsize_1304, out_mem_1303)
  def init(self, world_mem_1217_ext):
    n_953 = np.int32(world_mem_1217_ext.shape[np.int32(0)])
    m_954 = np.int32(world_mem_1217_ext.shape[np.int32(1)])
    world_mem_size_1216 = np.int32(world_mem_1217_ext.nbytes)
    if (type(world_mem_1217_ext) == cl.array.Array):
      world_mem_1217 = world_mem_1217_ext.data
    else:
      world_mem_1217 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                 long(long(world_mem_size_1216) if (world_mem_size_1216 > np.int32(0)) else np.int32(1)))
      if (world_mem_size_1216 != np.int32(0)):
        cl.enqueue_copy(self.queue, world_mem_1217, world_mem_1217_ext,
                        is_blocking=synchronous)
    (out_memsize_1279, out_mem_1278, out_memsize_1281,
     out_mem_1280) = self.futhark_init(world_mem_size_1216, world_mem_1217,
                                       n_953, m_954)
    return (cl.array.Array(self.queue, (n_953, m_954), ct.c_int8,
                           data=out_mem_1278), cl.array.Array(self.queue,
                                                              (n_953, m_954),
                                                              ct.c_int32,
                                                              data=out_mem_1280))
  def render_frame(self, all_history_mem_1227_ext):
    n_963 = np.int32(all_history_mem_1227_ext.shape[np.int32(0)])
    m_964 = np.int32(all_history_mem_1227_ext.shape[np.int32(1)])
    all_history_mem_size_1226 = np.int32(all_history_mem_1227_ext.nbytes)
    if (type(all_history_mem_1227_ext) == cl.array.Array):
      all_history_mem_1227 = all_history_mem_1227_ext.data
    else:
      all_history_mem_1227 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                       long(long(all_history_mem_size_1226) if (all_history_mem_size_1226 > np.int32(0)) else np.int32(1)))
      if (all_history_mem_size_1226 != np.int32(0)):
        cl.enqueue_copy(self.queue, all_history_mem_1227,
                        all_history_mem_1227_ext, is_blocking=synchronous)
    (out_memsize_1295,
     out_mem_1294) = self.futhark_render_frame(all_history_mem_size_1226,
                                               all_history_mem_1227, n_963,
                                               m_964)
    return cl.array.Array(self.queue, (n_963, m_964, np.int32(3)), ct.c_int8,
                          data=out_mem_1294)
  def steps(self, world_mem_1251_ext, history_mem_1253_ext, steps_986_ext):
    n_982 = np.int32(world_mem_1251_ext.shape[np.int32(0)])
    m_983 = np.int32(world_mem_1251_ext.shape[np.int32(1)])
    world_mem_size_1250 = np.int32(world_mem_1251_ext.nbytes)
    if (type(world_mem_1251_ext) == cl.array.Array):
      world_mem_1251 = world_mem_1251_ext.data
    else:
      world_mem_1251 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                 long(long(world_mem_size_1250) if (world_mem_size_1250 > np.int32(0)) else np.int32(1)))
      if (world_mem_size_1250 != np.int32(0)):
        cl.enqueue_copy(self.queue, world_mem_1251, world_mem_1251_ext,
                        is_blocking=synchronous)
    n_982 = np.int32(history_mem_1253_ext.shape[np.int32(0)])
    m_983 = np.int32(history_mem_1253_ext.shape[np.int32(1)])
    history_mem_size_1252 = np.int32(history_mem_1253_ext.nbytes)
    if (type(history_mem_1253_ext) == cl.array.Array):
      history_mem_1253 = history_mem_1253_ext.data
    else:
      history_mem_1253 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                   long(long(history_mem_size_1252) if (history_mem_size_1252 > np.int32(0)) else np.int32(1)))
      if (history_mem_size_1252 != np.int32(0)):
        cl.enqueue_copy(self.queue, history_mem_1253, history_mem_1253_ext,
                        is_blocking=synchronous)
    steps_986 = np.int32(steps_986_ext)
    (out_memsize_1302, out_mem_1301, out_memsize_1304,
     out_mem_1303) = self.futhark_steps(world_mem_size_1250,
                                        history_mem_size_1252, world_mem_1251,
                                        history_mem_1253, n_982, m_983,
                                        steps_986)
    return (cl.array.Array(self.queue, (n_982, m_983), ct.c_int8,
                           data=out_mem_1301), cl.array.Array(self.queue,
                                                              (n_982, m_983),
                                                              ct.c_int32,
                                                              data=out_mem_1303))