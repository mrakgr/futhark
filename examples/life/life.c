#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
#include <getopt.h>
/* Crash and burn. */

#include <stdarg.h>

static const char *fut_progname;

void panic(int eval, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
        fprintf(stderr, "%s: ", fut_progname);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
        exit(eval);
}

/* Some simple utilities for wall-clock timing.

   The function get_wall_time() returns the wall time in microseconds
   (with an unspecified offset).
*/

#ifdef _WIN32

#include <windows.h>

int64_t get_wall_time() {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

#else
/* Assuming POSIX */

#include <time.h>
#include <sys/time.h>

int64_t get_wall_time() {
  struct timeval time;
  assert(gettimeofday(&time,NULL) == 0);
  return time.tv_sec * 1000000 + time.tv_usec;
}

#endif

#define FUT_BLOCK_DIM 16
/* The simple OpenCL runtime framework used by Futhark. */

#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif

#define FUT_KERNEL(s) #s
#define OPENCL_SUCCEED(e) opencl_succeed(e, #e, __FILE__, __LINE__)

static cl_context fut_cl_context;
static cl_command_queue fut_cl_queue;
static const char *cl_preferred_platform = "";
static const char *cl_preferred_device = "";
static int cl_debug = 0;

static size_t cl_group_size = 256;
static size_t cl_num_groups = 128;
static size_t cl_lockstep_width = 1;

struct opencl_device_option {
  cl_platform_id platform;
  cl_device_id device;
  cl_device_type device_type;
  char *platform_name;
  char *device_name;
};

/* This function must be defined by the user.  It is invoked by
   setup_opencl() after the platform and device has been found, but
   before the program is loaded.  Its intended use is to tune
   constants based on the selected platform and device. */
static void post_opencl_setup(struct opencl_device_option*);

static char *strclone(const char *str) {
  size_t size = strlen(str) + 1;
  char *copy = malloc(size);
  if (copy == NULL) {
    return NULL;
  }

  memcpy(copy, str, size);
  return copy;
}

static const char* opencl_error_string(unsigned int err)
{
    switch (err) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default:                                    return "Unknown";
    }
}

static void opencl_succeed(unsigned int ret,
                    const char *call,
                    const char *file,
                    int line) {
  if (ret != CL_SUCCESS) {
    panic(-1, "%s:%d: OpenCL call\n  %s\nfailed with error code %d (%s)\n",
          file, line, call, ret, opencl_error_string(ret));
  }
}

static char* opencl_platform_info(cl_platform_id platform,
                                  cl_platform_info param) {
  size_t req_bytes;
  char *info;

  OPENCL_SUCCEED(clGetPlatformInfo(platform, param, 0, NULL, &req_bytes));

  info = malloc(req_bytes);

  OPENCL_SUCCEED(clGetPlatformInfo(platform, param, req_bytes, info, NULL));

  return info;
}

static char* opencl_device_info(cl_device_id device,
                                cl_device_info param) {
  size_t req_bytes;
  char *info;

  OPENCL_SUCCEED(clGetDeviceInfo(device, param, 0, NULL, &req_bytes));

  info = malloc(req_bytes);

  OPENCL_SUCCEED(clGetDeviceInfo(device, param, req_bytes, info, NULL));

  return info;
}

static void opencl_all_device_options(struct opencl_device_option **devices_out,
                                      size_t *num_devices_out) {
  size_t num_devices = 0, num_devices_added = 0;

  cl_platform_id *all_platforms;
  cl_uint *platform_num_devices;

  cl_uint num_platforms;

  // Find the number of platforms.
  OPENCL_SUCCEED(clGetPlatformIDs(0, NULL, &num_platforms));

  // Make room for them.
  all_platforms = calloc(num_platforms, sizeof(cl_platform_id));
  platform_num_devices = calloc(num_platforms, sizeof(cl_uint));

  // Fetch all the platforms.
  OPENCL_SUCCEED(clGetPlatformIDs(num_platforms, all_platforms, NULL));

  // Count the number of devices for each platform, as well as the
  // total number of devices.
  for (cl_uint i = 0; i < num_platforms; i++) {
    if (clGetDeviceIDs(all_platforms[i], CL_DEVICE_TYPE_ALL,
                       0, NULL, &platform_num_devices[i]) == CL_SUCCESS) {
      num_devices += platform_num_devices[i];
    } else {
      platform_num_devices[i] = 0;
    }
  }

  // Make room for all the device options.
  struct opencl_device_option *devices =
    calloc(num_devices, sizeof(struct opencl_device_option));

  // Loop through the platforms, getting information about their devices.
  for (cl_uint i = 0; i < num_platforms; i++) {
    cl_platform_id platform = all_platforms[i];
    cl_uint num_platform_devices = platform_num_devices[i];

    if (num_platform_devices == 0) {
      continue;
    }

    char *platform_name = opencl_platform_info(platform, CL_PLATFORM_NAME);
    cl_device_id *platform_devices =
      calloc(num_platform_devices, sizeof(cl_device_id));

    // Fetch all the devices.
    OPENCL_SUCCEED(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                                  num_platform_devices, platform_devices, NULL));

    // Loop through the devices, adding them to the devices array.
    for (cl_uint i = 0; i < num_platform_devices; i++) {
      char *device_name = opencl_device_info(platform_devices[i], CL_DEVICE_NAME);
      devices[num_devices_added].platform = platform;
      devices[num_devices_added].device = platform_devices[i];
      OPENCL_SUCCEED(clGetDeviceInfo(platform_devices[i], CL_DEVICE_TYPE,
                                     sizeof(cl_device_type),
                                     &devices[num_devices_added].device_type,
                                     NULL));
      // We don't want the structs to share memory, so copy the platform name.
      // Each device name is already unique.
      devices[num_devices_added].platform_name = strclone(platform_name);
      devices[num_devices_added].device_name = device_name;
      num_devices_added++;
    }
    free(platform_devices);
    free(platform_name);
  }
  free(all_platforms);
  free(platform_num_devices);

  *devices_out = devices;
  *num_devices_out = num_devices;
}

static struct opencl_device_option get_preferred_device() {
  struct opencl_device_option *devices;
  size_t num_devices;

  opencl_all_device_options(&devices, &num_devices);

  for (size_t i = 0; i < num_devices; i++) {
    struct opencl_device_option device = devices[i];
    if (strstr(device.platform_name, cl_preferred_platform) != NULL &&
        strstr(device.device_name, cl_preferred_device) != NULL) {
      // Free all the platform and device names, except the ones we have chosen.
      for (size_t j = 0; j < num_devices; j++) {
        if (j != i) {
          free(devices[j].platform_name);
          free(devices[j].device_name);
        }
      }
      free(devices);
      return device;
    }
  }

  panic(1, "Could not find acceptable OpenCL device.");
}

static void describe_device_option(struct opencl_device_option device) {
  fprintf(stderr, "Using platform: %s\n", device.platform_name);
  fprintf(stderr, "Using device: %s\n", device.device_name);
}

static cl_build_status build_opencl_program(cl_program program, cl_device_id device, const char* options) {
  cl_int ret_val = clBuildProgram(program, 1, &device, options, NULL, NULL);

  // Avoid termination due to CL_BUILD_PROGRAM_FAILURE
  if (ret_val != CL_SUCCESS && ret_val != CL_BUILD_PROGRAM_FAILURE) {
    assert(ret_val == 0);
  }

  cl_build_status build_status;
  ret_val = clGetProgramBuildInfo(program,
                                  device,
                                  CL_PROGRAM_BUILD_STATUS,
                                  sizeof(cl_build_status),
                                  &build_status,
                                  NULL);
  assert(ret_val == 0);

  if (build_status != CL_SUCCESS) {
    char *build_log;
    size_t ret_val_size;
    ret_val = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
    assert(ret_val == 0);

    build_log = malloc(ret_val_size+1);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
    assert(ret_val == 0);

    // The spec technically does not say whether the build log is zero-terminated, so let's be careful.
    build_log[ret_val_size] = '\0';

    fprintf(stderr, "Build log:\n%s\n", build_log);

    free(build_log);
  }

  return build_status;
}

static cl_program setup_opencl(const char *prelude_src, const char *src) {

  cl_int error;
  cl_platform_id platform;
  cl_device_id device;
  cl_uint platforms, devices;
  size_t max_group_size;

  struct opencl_device_option device_option = get_preferred_device();

  if (cl_debug) {
    describe_device_option(device_option);
  }

  device = device_option.device;
  platform = device_option.platform;

  OPENCL_SUCCEED(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                 sizeof(size_t), &max_group_size, NULL));

  if (max_group_size < cl_group_size) {
    fprintf(stderr, "Warning: Device limits group size to %zu (setting was %zu)\n",
            max_group_size, cl_group_size);
    cl_group_size = max_group_size;
  }

  cl_context_properties properties[] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)platform,
    0
  };
  // Note that nVidia's OpenCL requires the platform property
  fut_cl_context = clCreateContext(properties, 1, &device, NULL, NULL, &error);
  assert(error == 0);

  fut_cl_queue = clCreateCommandQueue(fut_cl_context, device, 0, &error);
  assert(error == 0);

  /* Make sure this function is defined. */
  post_opencl_setup(&device_option);

  // Build the OpenCL program.  First we have to prepend the prelude to the program source.
  size_t prelude_size = strlen(prelude_src);
  size_t program_size = strlen(src);
  size_t src_size = prelude_size + program_size;
  char *fut_opencl_src = malloc(src_size + 1);
  strncpy(fut_opencl_src, prelude_src, src_size);
  strncpy(fut_opencl_src+prelude_size, src, src_size-prelude_size);
  fut_opencl_src[src_size] = '0';

  cl_program prog;
  error = 0;
  const char* src_ptr[] = {fut_opencl_src};
  prog = clCreateProgramWithSource(fut_cl_context, 1, src_ptr, &src_size, &error);
  assert(error == 0);
  char compile_opts[1024];
  snprintf(compile_opts, sizeof(compile_opts), "-DFUT_BLOCK_DIM=%d -DLOCKSTEP_WIDTH=%d", FUT_BLOCK_DIM, cl_lockstep_width);
  OPENCL_SUCCEED(build_opencl_program(prog, device, compile_opts));
  free(fut_opencl_src);

  return prog;
}

static const char fut_opencl_prelude[] =
                  "typedef char int8_t;\ntypedef short int16_t;\ntypedef int int32_t;\ntypedef long int64_t;\ntypedef uchar uint8_t;\ntypedef ushort uint16_t;\ntypedef uint uint32_t;\ntypedef ulong uint64_t;\nstatic inline int8_t add8(int8_t x, int8_t y)\n{\n    return x + y;\n}\nstatic inline int16_t add16(int16_t x, int16_t y)\n{\n    return x + y;\n}\nstatic inline int32_t add32(int32_t x, int32_t y)\n{\n    return x + y;\n}\nstatic inline int64_t add64(int64_t x, int64_t y)\n{\n    return x + y;\n}\nstatic inline int8_t sub8(int8_t x, int8_t y)\n{\n    return x - y;\n}\nstatic inline int16_t sub16(int16_t x, int16_t y)\n{\n    return x - y;\n}\nstatic inline int32_t sub32(int32_t x, int32_t y)\n{\n    return x - y;\n}\nstatic inline int64_t sub64(int64_t x, int64_t y)\n{\n    return x - y;\n}\nstatic inline int8_t mul8(int8_t x, int8_t y)\n{\n    return x * y;\n}\nstatic inline int16_t mul16(int16_t x, int16_t y)\n{\n    return x * y;\n}\nstatic inline int32_t mul32(int32_t x, int32_t y)\n{\n    return x * y;\n}\nstatic inline int64_t mul64(int64_t x, int64_t y)\n{\n    return x * y;\n}\nstatic inline uint8_t udiv8(uint8_t x, uint8_t y)\n{\n    return x / y;\n}\nstatic inline uint16_t udiv16(uint16_t x, uint16_t y)\n{\n    return x / y;\n}\nstatic inline uint32_t udiv32(uint32_t x, uint32_t y)\n{\n    return x / y;\n}\nstatic inline uint64_t udiv64(uint64_t x, uint64_t y)\n{\n    return x / y;\n}\nstatic inline uint8_t umod8(uint8_t x, uint8_t y)\n{\n    return x % y;\n}\nstatic inline uint16_t umod16(uint16_t x, uint16_t y)\n{\n    return x % y;\n}\nstatic inline uint32_t umod32(uint32_t x, uint32_t y)\n{\n    return x % y;\n}\nstatic inline uint64_t umod64(uint64_t x, uint64_t y)\n{\n    return x % y;\n}\nstatic inline int8_t sdiv8(int8_t x, int8_t y)\n{\n    int8_t q = x / y;\n    int8_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int16_t sdiv16(int16_t x, int16_t y)\n{\n    int16_t q = x / y;\n    int16_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int32_t sdiv32(int32_t x, int32_t y)\n{\n    int32_t q = x / y;\n    int32_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int64_t sdiv64(int64_t x, int64_t y)\n{\n    int64_t q = x / y;\n    int64_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int8_t smod8(int8_t x, int8_t y)\n{\n    int8_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int16_t smod16(int16_t x, int16_t y)\n{\n    int16_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int32_t smod32(int32_t x, int32_t y)\n{\n    int32_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int64_t smod64(int64_t x, int64_t y)\n{\n    int64_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int8_t squot8(int8_t x, int8_t y)\n{\n    return x / y;\n}\nstatic inline int16_t squot16(int16_t x, int16_t y)\n{\n    return x / y;\n}\nstatic inline int32_t squot32(int32_t x, int32_t y)\n{\n    return x / y;\n}\nstatic inline int64_t squot64(int64_t x, int64_t y)\n{\n    return x / y;\n}\nstatic inline int8_t srem8(int8_t x, int8_t y)\n{\n    return x % y;\n}\nstatic inline int16_t srem16(int16_t x, int16_t y)\n{\n    return x % y;\n}\nstatic inline int32_t srem32(int32_t x, int32_t y)\n{\n    return x % y;\n}\nstatic inline int64_t srem64(int64_t x, int64_t y)\n{\n    return x % y;\n}\nstatic inline uint8_t shl8(uint8_t x, uint8_t y)\n{\n    return x << y;\n}\nstatic inline uint16_t shl16(uint16_t x, uint16_t y)\n{\n    return x << y;\n}\nstatic inline uint32_t shl32(uint32_t x, uint32_t y)\n{\n    return x << y;\n}\nstatic inline uint64_t shl64(uint64_t x, uint64_t y)\n{\n    return x << y;\n}\nstatic inline uint8_t lshr8(uint8_t x, uint8_t y)\n{\n    return x >> y;\n}\nstatic inline uint16_t lshr16(uint16_t x, uint16_t y)\n{\n    return x >> y;\n}\nstatic inline uint32_t lshr32(uint32_t x, uint32_t y)\n{\n    return x >> y;\n}\nstatic inline uint64_t lshr64(uint64_t x, uint64_t y)\n{\n    return x >> y;\n}\nstatic inline int8_t ashr8(int8_t x, int8_t y)\n{\n    return x >> y;\n}\nstatic inline int16_t ashr16(int16_t x, int16_t y)\n{\n    return x >> y;\n}\nstatic inline int32_t ashr32(int32_t x, int32_t y)\n{\n    return x >> y;\n}\nstatic inline int64_t ashr64(int64_t x, int64_t y)\n{\n    return x >> y;\n}\nstatic inline uint8_t and8(uint8_t x, uint8_t y)\n{\n    return x & y;\n}\nstatic inline uint16_t and16(uint16_t x, uint16_t y)\n{\n    return x & y;\n}\nstatic inline uint32_t and32(uint32_t x, uint32_t y)\n{\n    return x & y;\n}\nstatic inline uint64_t and64(uint64_t x, uint64_t y)\n{\n    return x & y;\n}\nstatic inline uint8_t or8(uint8_t x, uint8_t y)\n{\n    return x | y;\n}\nstatic inline uint16_t or16(uint16_t x, uint16_t y)\n{\n    return x | y;\n}\nstatic inline uint32_t or32(uint32_t x, uint32_t y)\n{\n    return x | y;\n}\nstatic inline uint64_t or64(uint64_t x, uint64_t y)\n{\n    return x | y;\n}\nstatic inline uint8_t xor8(uint8_t x, uint8_t y)\n{\n    return x ^ y;\n}\nstatic inline uint16_t xor16(uint16_t x, uint16_t y)\n{\n    return x ^ y;\n}\nstatic inline uint32_t xor32(uint32_t x, uint32_t y)\n{\n    return x ^ y;\n}\nstatic inline uint64_t xor64(uint64_t x, uint64_t y)\n{\n    return x ^ y;\n}\nstatic inline char ult8(uint8_t x, uint8_t y)\n{\n    return x < y;\n}\nstatic inline char ult16(uint16_t x, uint16_t y)\n{\n    return x < y;\n}\nstatic inline char ult32(uint32_t x, uint32_t y)\n{\n    return x < y;\n}\nstatic inline char ult64(uint64_t x, uint64_t y)\n{\n    return x < y;\n}\nstatic inline char ule8(uint8_t x, uint8_t y)\n{\n    return x <= y;\n}\nstatic inline char ule16(uint16_t x, uint16_t y)\n{\n    return x <= y;\n}\nstatic inline char ule32(uint32_t x, uint32_t y)\n{\n    return x <= y;\n}\nstatic inline char ule64(uint64_t x, uint64_t y)\n{\n    return x <= y;\n}\nstatic inline char slt8(int8_t x, int8_t y)\n{\n    return x < y;\n}\nstatic inline char slt16(int16_t x, int16_t y)\n{\n    return x < y;\n}\nstatic inline char slt32(int32_t x, int32_t y)\n{\n    return x < y;\n}\nstatic inline char slt64(int64_t x, int64_t y)\n{\n    return x < y;\n}\nstatic inline char sle8(int8_t x, int8_t y)\n{\n    return x <= y;\n}\nstatic inline char sle16(int16_t x, int16_t y)\n{\n    return x <= y;\n}\nstatic inline char sle32(int32_t x, int32_t y)\n{\n    return x <= y;\n}\nstatic inline char sle64(int64_t x, int64_t y)\n{\n    return x <= y;\n}\nstatic inline int8_t pow8(int8_t x, int8_t y)\n{\n    int8_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int16_t pow16(int16_t x, int16_t y)\n{\n    int16_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int32_t pow32(int32_t x, int32_t y)\n{\n    int32_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int64_t pow64(int64_t x, int64_t y)\n{\n    int64_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int8_t sext_i8_i8(int8_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i8_i16(int8_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i8_i32(int8_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i8_i64(int8_t x)\n{\n    return x;\n}\nstatic inline int8_t sext_i16_i8(int16_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i16_i16(int16_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i16_i32(int16_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i16_i64(int16_t x)\n{\n    return x;\n}\nstatic inline int8_t sext_i32_i8(int32_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i32_i16(int32_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i32_i32(int32_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i32_i64(int32_t x)\n{\n    return x;\n}\nstatic inline int8_t sext_i64_i8(int64_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i64_i16(int64_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i64_i32(int64_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i64_i64(int64_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i8_i8(uint8_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i8_i16(uint8_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i8_i32(uint8_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i8_i64(uint8_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i16_i8(uint16_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i16_i16(uint16_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i16_i32(uint16_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i16_i64(uint16_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i32_i8(uint32_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i32_i16(uint32_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i32_i32(uint32_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i32_i64(uint32_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i64_i8(uint64_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i64_i16(uint64_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i64_i32(uint64_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i64_i64(uint64_t x)\n{\n    return x;\n}\nstatic inline float fdiv32(float x, float y)\n{\n    return x / y;\n}\nstatic inline float fadd32(float x, float y)\n{\n    return x + y;\n}\nstatic inline float fsub32(float x, float y)\n{\n    return x - y;\n}\nstatic inline float fmul32(float x, float y)\n{\n    return x * y;\n}\nstatic inline float fpow32(float x, float y)\n{\n    return pow(x, y);\n}\nstatic inline char cmplt32(float x, float y)\n{\n    return x < y;\n}\nstatic inline char cmple32(float x, float y)\n{\n    return x <= y;\n}\nstatic inline float sitofp_i8_f32(int8_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i16_f32(int16_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i32_f32(int32_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i64_f32(int64_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i8_f32(uint8_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i16_f32(uint16_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i32_f32(uint32_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i64_f32(uint64_t x)\n{\n    return x;\n}\nstatic inline int8_t fptosi_f32_i8(float x)\n{\n    return x;\n}\nstatic inline int16_t fptosi_f32_i16(float x)\n{\n    return x;\n}\nstatic inline int32_t fptosi_f32_i32(float x)\n{\n    return x;\n}\nstatic inline int64_t fptosi_f32_i64(float x)\n{\n    return x;\n}\nstatic inline uint8_t fptoui_f32_i8(float x)\n{\n    return x;\n}\nstatic inline uint16_t fptoui_f32_i16(float x)\n{\n    return x;\n}\nstatic inline uint32_t fptoui_f32_i32(float x)\n{\n    return x;\n}\nstatic inline uint64_t fptoui_f32_i64(float x)\n{\n    return x;\n}\n";
static const char fut_opencl_program[] = FUT_KERNEL(
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
);
static cl_kernel map_kernel_1274;
static int map_kernel_1274total_runtime = 0;
static int map_kernel_1274runs = 0;
static cl_kernel map_kernel_1278;
static int map_kernel_1278total_runtime = 0;
static int map_kernel_1278runs = 0;
static cl_kernel map_kernel_1107;
static int map_kernel_1107total_runtime = 0;
static int map_kernel_1107runs = 0;
static cl_kernel fut_kernel_map_transpose_i8;
static int fut_kernel_map_transpose_i8total_runtime = 0;
static int fut_kernel_map_transpose_i8runs = 0;
static cl_kernel map_kernel_1168;
static int map_kernel_1168total_runtime = 0;
static int map_kernel_1168runs = 0;
static cl_kernel map_kernel_1132;
static int map_kernel_1132total_runtime = 0;
static int map_kernel_1132runs = 0;
static cl_kernel map_kernel_1186;
static int map_kernel_1186total_runtime = 0;
static int map_kernel_1186runs = 0;
static cl_kernel map_kernel_1204;
static int map_kernel_1204total_runtime = 0;
static int map_kernel_1204runs = 0;
void setup_opencl_and_load_kernels()

{
    cl_int error;
    cl_program prog = setup_opencl(fut_opencl_prelude, fut_opencl_program);
    
    {
        map_kernel_1274 = clCreateKernel(prog, "map_kernel_1274", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_1274");
    }
    {
        map_kernel_1278 = clCreateKernel(prog, "map_kernel_1278", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_1278");
    }
    {
        map_kernel_1107 = clCreateKernel(prog, "map_kernel_1107", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_1107");
    }
    {
        fut_kernel_map_transpose_i8 = clCreateKernel(prog,
                                                     "fut_kernel_map_transpose_i8",
                                                     &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n",
                    "fut_kernel_map_transpose_i8");
    }
    {
        map_kernel_1168 = clCreateKernel(prog, "map_kernel_1168", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_1168");
    }
    {
        map_kernel_1132 = clCreateKernel(prog, "map_kernel_1132", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_1132");
    }
    {
        map_kernel_1186 = clCreateKernel(prog, "map_kernel_1186", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_1186");
    }
    {
        map_kernel_1204 = clCreateKernel(prog, "map_kernel_1204", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_1204");
    }
}
void post_opencl_setup(struct opencl_device_option *option)
{
    if (strcmp(option->platform_name, "NVIDIA CUDA") == 0 &&
        option->device_type == CL_DEVICE_TYPE_GPU) {
        cl_lockstep_width = 32;
        if (cl_debug)
            fprintf(stderr, "Setting lockstep width to: %d\n",
                    cl_lockstep_width);
    }
    if (strcmp(option->platform_name, "AMD Accelerated Parallel Processing") ==
        0 && option->device_type == CL_DEVICE_TYPE_GPU) {
        cl_lockstep_width = 64;
        if (cl_debug)
            fprintf(stderr, "Setting lockstep width to: %d\n",
                    cl_lockstep_width);
    }
}
struct memblock_device {
    int *references;
    cl_mem mem;
} ;
static void memblock_unref_device(struct memblock_device *block)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (*block->references == 0) {
            OPENCL_SUCCEED(clReleaseMemObject(block->mem));
            free(block->references);
            block->references = NULL;
        }
    }
}
static void memblock_alloc_device(struct memblock_device *block, int32_t size)
{
    memblock_unref_device(block);
    
    cl_int clCreateBuffer_succeeded_1359;
    
    block->mem = clCreateBuffer(fut_cl_context, CL_MEM_READ_WRITE, size >
                                0 ? size : 1, NULL,
                                &clCreateBuffer_succeeded_1359);
    OPENCL_SUCCEED(clCreateBuffer_succeeded_1359);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
}
static void memblock_set_device(struct memblock_device *lhs,
                                struct memblock_device *rhs)
{
    memblock_unref_device(lhs);
    (*rhs->references)++;
    *lhs = *rhs;
}
struct memblock_local {
    int *references;
    unsigned char mem;
} ;
static void memblock_unref_local(struct memblock_local *block)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (*block->references == 0) {
            free(block->references);
            block->references = NULL;
        }
    }
}
static void memblock_alloc_local(struct memblock_local *block, int32_t size)
{
    memblock_unref_local(block);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
}
static void memblock_set_local(struct memblock_local *lhs,
                               struct memblock_local *rhs)
{
    memblock_unref_local(lhs);
    (*rhs->references)++;
    *lhs = *rhs;
}
struct memblock {
    int *references;
    char *mem;
} ;
static void memblock_unref(struct memblock *block)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (*block->references == 0) {
            free(block->mem);
            free(block->references);
            block->references = NULL;
        }
    }
}
static void memblock_alloc(struct memblock *block, int32_t size)
{
    memblock_unref(block);
    block->mem = (char *) malloc(size);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
}
static void memblock_set(struct memblock *lhs, struct memblock *rhs)
{
    memblock_unref(lhs);
    (*rhs->references)++;
    *lhs = *rhs;
}
struct tuple_int32_t_device_mem_int32_t_device_mem {
    int32_t elem_0;
    struct memblock_device elem_1;
    int32_t elem_2;
    struct memblock_device elem_3;
} ;
struct tuple_int32_t_device_mem {
    int32_t elem_0;
    struct memblock_device elem_1;
} ;
static struct tuple_int32_t_device_mem_int32_t_device_mem
futhark_init(int32_t world_mem_size_1212, struct memblock_device world_mem_1213, int32_t n_658, int32_t m_659);
static struct tuple_int32_t_device_mem
futhark_render_frame(int32_t history_mem_size_1219, struct memblock_device history_mem_1220, int32_t n_679, int32_t m_680);
static int32_t futhark_main();
static struct tuple_int32_t_device_mem_int32_t_device_mem
futhark_steps(int32_t world_mem_size_1236, int32_t history_mem_size_1238, struct memblock_device world_mem_1237, struct memblock_device history_mem_1239, int32_t n_699, int32_t m_700, int32_t steps_703);
static inline float futhark_log32(float x)
{
    return log(x);
}
static inline float futhark_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futhark_exp32(float x)
{
    return exp(x);
}
static inline float futhark_cos32(float x)
{
    return cos(x);
}
static inline float futhark_sin32(float x)
{
    return sin(x);
}
static inline double futhark_atan2_32(double x, double y)
{
    return atan2(x, y);
}
static inline char futhark_isnan32(float x)
{
    return isnan(x);
}
static inline char futhark_isinf32(float x)
{
    return isinf(x);
}
static inline double futhark_log64(double x)
{
    return log(x);
}
static inline double futhark_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futhark_exp64(double x)
{
    return exp(x);
}
static inline double futhark_cos64(double x)
{
    return cos(x);
}
static inline double futhark_sin64(double x)
{
    return sin(x);
}
static inline double futhark_atan2_64(double x, double y)
{
    return atan2(x, y);
}
static inline char futhark_isnan64(double x)
{
    return isnan(x);
}
static inline char futhark_isinf64(double x)
{
    return isinf(x);
}
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
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline char cmplt64(double x, double y)
{
    return x < y;
}
static inline char cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return x;
}
static inline float fpconv_f32_f32(float x)
{
    return x;
}
static inline double fpconv_f32_f64(float x)
{
    return x;
}
static inline float fpconv_f64_f32(double x)
{
    return x;
}
static inline double fpconv_f64_f64(double x)
{
    return x;
}
static int detail_timing = 0;
static
struct tuple_int32_t_device_mem_int32_t_device_mem futhark_init(int32_t world_mem_size_1212,
                                                                struct memblock_device world_mem_1213,
                                                                int32_t n_658,
                                                                int32_t m_659)
{
    int32_t out_memsize_1271;
    struct memblock_device out_mem_1270;
    
    out_mem_1270.references = NULL;
    
    int32_t out_memsize_1273;
    struct memblock_device out_mem_1272;
    
    out_mem_1272.references = NULL;
    
    int32_t bytes_1214 = 4 * m_659;
    struct memblock_device mem_1215;
    
    mem_1215.references = NULL;
    memblock_alloc_device(&mem_1215, bytes_1214);
    
    int32_t group_size_1276;
    int32_t num_groups_1277;
    
    group_size_1276 = cl_group_size;
    num_groups_1277 = squot32(m_659 + group_size_1276 - 1, group_size_1276);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_1274, 0, sizeof(m_659), &m_659));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_1274, 1, sizeof(mem_1215.mem),
                                  &mem_1215.mem));
    if (1 * (num_groups_1277 * group_size_1276) != 0) {
        const size_t global_work_size_1311[1] = {num_groups_1277 *
                     group_size_1276};
        const size_t local_work_size_1315[1] = {group_size_1276};
        int64_t time_start_1312, time_end_1313;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_1274");
            fprintf(stderr, "%zu", global_work_size_1311[0]);
            fprintf(stderr, "].\n");
            time_start_1312 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_1274, 1,
                                              NULL, global_work_size_1311,
                                              local_work_size_1315, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_1313 = get_wall_time();
            
            long time_diff_1314 = time_end_1313 - time_start_1312;
            
            if (detail_timing) {
                map_kernel_1274total_runtime += time_diff_1314;
                map_kernel_1274runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_1274",
                        (int) time_diff_1314);
            }
        }
    }
    
    int32_t x_1217 = 4 * n_658;
    int32_t bytes_1216 = x_1217 * m_659;
    struct memblock_device mem_1218;
    
    mem_1218.references = NULL;
    memblock_alloc_device(&mem_1218, bytes_1216);
    
    int32_t group_size_1282;
    int32_t num_groups_1283;
    
    group_size_1282 = cl_group_size;
    num_groups_1283 = squot32(n_658 * m_659 + group_size_1282 - 1,
                              group_size_1282);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_1278, 0, sizeof(n_658), &n_658));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_1278, 1, sizeof(m_659), &m_659));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_1278, 2, sizeof(mem_1215.mem),
                                  &mem_1215.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_1278, 3, sizeof(mem_1218.mem),
                                  &mem_1218.mem));
    if (1 * (num_groups_1283 * group_size_1282) != 0) {
        const size_t global_work_size_1316[1] = {num_groups_1283 *
                     group_size_1282};
        const size_t local_work_size_1320[1] = {group_size_1282};
        int64_t time_start_1317, time_end_1318;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_1278");
            fprintf(stderr, "%zu", global_work_size_1316[0]);
            fprintf(stderr, "].\n");
            time_start_1317 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_1278, 1,
                                              NULL, global_work_size_1316,
                                              local_work_size_1320, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_1318 = get_wall_time();
            
            long time_diff_1319 = time_end_1318 - time_start_1317;
            
            if (detail_timing) {
                map_kernel_1278total_runtime += time_diff_1319;
                map_kernel_1278runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_1278",
                        (int) time_diff_1319);
            }
        }
    }
    memblock_set_device(&out_mem_1270, &world_mem_1213);
    out_memsize_1271 = world_mem_size_1212;
    memblock_set_device(&out_mem_1272, &mem_1218);
    out_memsize_1273 = bytes_1216;
    
    struct tuple_int32_t_device_mem_int32_t_device_mem retval_1310;
    
    retval_1310.elem_0 = out_memsize_1271;
    retval_1310.elem_1.references = NULL;
    memblock_set_device(&retval_1310.elem_1, &out_mem_1270);
    retval_1310.elem_2 = out_memsize_1273;
    retval_1310.elem_3.references = NULL;
    memblock_set_device(&retval_1310.elem_3, &out_mem_1272);
    memblock_unref_device(&out_mem_1270);
    memblock_unref_device(&out_mem_1272);
    memblock_unref_device(&mem_1215);
    memblock_unref_device(&mem_1218);
    return retval_1310;
}
static
struct tuple_int32_t_device_mem futhark_render_frame(int32_t history_mem_size_1219,
                                                     struct memblock_device history_mem_1220,
                                                     int32_t n_679,
                                                     int32_t m_680)
{
    int32_t out_memsize_1285;
    struct memblock_device out_mem_1284;
    
    out_mem_1284.references = NULL;
    
    int32_t nesting_size_1105 = m_680 * n_679;
    int32_t x_1229 = n_679 * 3;
    int32_t bytes_1227 = x_1229 * m_680;
    struct memblock_device mem_1230;
    
    mem_1230.references = NULL;
    memblock_alloc_device(&mem_1230, bytes_1227);
    
    int32_t total_size_1267 = nesting_size_1105 * 3;
    struct memblock_device mem_1224;
    
    mem_1224.references = NULL;
    memblock_alloc_device(&mem_1224, total_size_1267);
    
    int32_t total_size_1268 = nesting_size_1105 * 3;
    struct memblock_device res_mem_1226;
    
    res_mem_1226.references = NULL;
    memblock_alloc_device(&res_mem_1226, total_size_1268);
    
    int32_t total_size_1269 = nesting_size_1105 * 3;
    struct memblock_device mem_1222;
    
    mem_1222.references = NULL;
    memblock_alloc_device(&mem_1222, total_size_1269);
    
    int32_t group_size_1289;
    int32_t num_groups_1290;
    
    group_size_1289 = cl_group_size;
    num_groups_1290 = squot32(n_679 * m_680 + group_size_1289 - 1,
                              group_size_1289);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_1107, 0, sizeof(mem_1224.mem),
                                  &mem_1224.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_1107, 1,
                                  sizeof(history_mem_1220.mem),
                                  &history_mem_1220.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_1107, 2, sizeof(m_680), &m_680));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_1107, 3, sizeof(nesting_size_1105),
                                  &nesting_size_1105));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_1107, 4, sizeof(res_mem_1226.mem),
                                  &res_mem_1226.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_1107, 5, sizeof(mem_1222.mem),
                                  &mem_1222.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_1107, 6, sizeof(n_679), &n_679));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_1107, 7, sizeof(mem_1230.mem),
                                  &mem_1230.mem));
    if (1 * (num_groups_1290 * group_size_1289) != 0) {
        const size_t global_work_size_1322[1] = {num_groups_1290 *
                     group_size_1289};
        const size_t local_work_size_1326[1] = {group_size_1289};
        int64_t time_start_1323, time_end_1324;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_1107");
            fprintf(stderr, "%zu", global_work_size_1322[0]);
            fprintf(stderr, "].\n");
            time_start_1323 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_1107, 1,
                                              NULL, global_work_size_1322,
                                              local_work_size_1326, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_1324 = get_wall_time();
            
            long time_diff_1325 = time_end_1324 - time_start_1323;
            
            if (detail_timing) {
                map_kernel_1107total_runtime += time_diff_1325;
                map_kernel_1107runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_1107",
                        (int) time_diff_1325);
            }
        }
    }
    
    int32_t x_1233 = n_679 * m_680;
    int32_t bytes_1231 = x_1233 * 3;
    struct memblock_device mem_1234;
    
    mem_1234.references = NULL;
    memblock_alloc_device(&mem_1234, bytes_1231);
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i8, 0,
                                  sizeof(mem_1234.mem), &mem_1234.mem));
    
    int32_t kernel_arg_1327 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i8, 1,
                                  sizeof(kernel_arg_1327), &kernel_arg_1327));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i8, 2,
                                  sizeof(mem_1230.mem), &mem_1230.mem));
    
    int32_t kernel_arg_1328 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i8, 3,
                                  sizeof(kernel_arg_1328), &kernel_arg_1328));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i8, 4, sizeof(m_680),
                                  &m_680));
    
    int32_t kernel_arg_1329 = 3;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i8, 5,
                                  sizeof(kernel_arg_1329), &kernel_arg_1329));
    
    int32_t kernel_arg_1330 = n_679 * m_680 * 3;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i8, 6,
                                  sizeof(kernel_arg_1330), &kernel_arg_1330));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i8, 7, (16 + 1) *
                                  16 * sizeof(int8_t), NULL));
    if (1 * (m_680 + srem32(16 - srem32(m_680, 16), 16)) * (3 + srem32(16 -
                                                                       srem32(3,
                                                                              16),
                                                                       16)) *
        n_679 != 0) {
        const size_t global_work_size_1331[3] = {m_680 + srem32(16 -
                                                                srem32(m_680,
                                                                       16), 16),
                                                 3 + srem32(16 - srem32(3, 16),
                                                            16), n_679};
        const size_t local_work_size_1335[3] = {16, 16, 1};
        int64_t time_start_1332, time_end_1333;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "fut_kernel_map_transpose_i8");
            fprintf(stderr, "%zu", global_work_size_1331[0]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_1331[1]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_1331[2]);
            fprintf(stderr, "].\n");
            time_start_1332 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue,
                                              fut_kernel_map_transpose_i8, 3,
                                              NULL, global_work_size_1331,
                                              local_work_size_1335, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_1333 = get_wall_time();
            
            long time_diff_1334 = time_end_1333 - time_start_1332;
            
            if (detail_timing) {
                fut_kernel_map_transpose_i8total_runtime += time_diff_1334;
                fut_kernel_map_transpose_i8runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "fut_kernel_map_transpose_i8", (int) time_diff_1334);
            }
        }
    }
    memblock_set_device(&out_mem_1284, &mem_1234);
    out_memsize_1285 = bytes_1231;
    
    struct tuple_int32_t_device_mem retval_1321;
    
    retval_1321.elem_0 = out_memsize_1285;
    retval_1321.elem_1.references = NULL;
    memblock_set_device(&retval_1321.elem_1, &out_mem_1284);
    memblock_unref_device(&out_mem_1284);
    memblock_unref_device(&mem_1230);
    memblock_unref_device(&mem_1224);
    memblock_unref_device(&res_mem_1226);
    memblock_unref_device(&mem_1222);
    memblock_unref_device(&mem_1234);
    return retval_1321;
}
static int32_t futhark_main()
{
    int32_t scalar_out_1291;
    
    scalar_out_1291 = 2;
    
    int32_t retval_1336;
    
    retval_1336 = scalar_out_1291;
    return retval_1336;
}
static
struct tuple_int32_t_device_mem_int32_t_device_mem futhark_steps(int32_t world_mem_size_1236,
                                                                 int32_t history_mem_size_1238,
                                                                 struct memblock_device world_mem_1237,
                                                                 struct memblock_device history_mem_1239,
                                                                 int32_t n_699,
                                                                 int32_t m_700,
                                                                 int32_t steps_703)
{
    int32_t out_memsize_1293;
    struct memblock_device out_mem_1292;
    
    out_mem_1292.references = NULL;
    
    int32_t out_memsize_1295;
    struct memblock_device out_mem_1294;
    
    out_mem_1294.references = NULL;
    
    int32_t bytes_1240 = 4 * n_699;
    struct memblock_device mem_1241;
    
    mem_1241.references = NULL;
    memblock_alloc_device(&mem_1241, bytes_1240);
    
    struct memblock_device mem_1243;
    
    mem_1243.references = NULL;
    memblock_alloc_device(&mem_1243, bytes_1240);
    
    int32_t group_size_1296;
    int32_t num_groups_1297;
    
    group_size_1296 = cl_group_size;
    num_groups_1297 = squot32(n_699 + group_size_1296 - 1, group_size_1296);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_1168, 0, sizeof(n_699), &n_699));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_1168, 1, sizeof(mem_1241.mem),
                                  &mem_1241.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_1168, 2, sizeof(mem_1243.mem),
                                  &mem_1243.mem));
    if (1 * (num_groups_1297 * group_size_1296) != 0) {
        const size_t global_work_size_1338[1] = {num_groups_1297 *
                     group_size_1296};
        const size_t local_work_size_1342[1] = {group_size_1296};
        int64_t time_start_1339, time_end_1340;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_1168");
            fprintf(stderr, "%zu", global_work_size_1338[0]);
            fprintf(stderr, "].\n");
            time_start_1339 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_1168, 1,
                                              NULL, global_work_size_1338,
                                              local_work_size_1342, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_1340 = get_wall_time();
            
            long time_diff_1341 = time_end_1340 - time_start_1339;
            
            if (detail_timing) {
                map_kernel_1168total_runtime += time_diff_1341;
                map_kernel_1168runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_1168",
                        (int) time_diff_1341);
            }
        }
    }
    
    int32_t nesting_size_1130 = m_700 * n_699;
    int32_t bytes_1248 = bytes_1240 * m_700;
    struct memblock_device mem_1250;
    
    mem_1250.references = NULL;
    memblock_alloc_device(&mem_1250, bytes_1248);
    
    int32_t bytes_1251 = n_699 * m_700;
    struct memblock_device double_buffer_mem_1263;
    
    double_buffer_mem_1263.references = NULL;
    memblock_alloc_device(&double_buffer_mem_1263, bytes_1251);
    
    struct memblock_device double_buffer_mem_1264;
    
    double_buffer_mem_1264.references = NULL;
    memblock_alloc_device(&double_buffer_mem_1264, bytes_1248);
    
    struct memblock_device mem_1253;
    
    mem_1253.references = NULL;
    memblock_alloc_device(&mem_1253, bytes_1251);
    
    struct memblock_device mem_1256;
    
    mem_1256.references = NULL;
    memblock_alloc_device(&mem_1256, bytes_1248);
    
    int32_t world_mem_size_1257;
    int32_t history_mem_size_1259;
    struct memblock_device world_mem_1258;
    
    world_mem_1258.references = NULL;
    
    struct memblock_device history_mem_1260;
    
    history_mem_1260.references = NULL;
    
    int32_t world_mem_size_1244;
    int32_t history_mem_size_1246;
    struct memblock_device world_mem_1245;
    
    world_mem_1245.references = NULL;
    
    struct memblock_device history_mem_1247;
    
    history_mem_1247.references = NULL;
    world_mem_size_1244 = world_mem_size_1236;
    history_mem_size_1246 = history_mem_size_1238;
    memblock_set_device(&world_mem_1245, &world_mem_1237);
    memblock_set_device(&history_mem_1247, &history_mem_1239);
    for (int i_706 = 0; i_706 < steps_703; i_706++) {
        int32_t group_size_1304;
        int32_t num_groups_1305;
        
        group_size_1304 = cl_group_size;
        num_groups_1305 = squot32(n_699 * m_700 + group_size_1304 - 1,
                                  group_size_1304);
        OPENCL_SUCCEED(clSetKernelArg(map_kernel_1132, 0, sizeof(m_700),
                                      &m_700));
        OPENCL_SUCCEED(clSetKernelArg(map_kernel_1132, 1, sizeof(mem_1241.mem),
                                      &mem_1241.mem));
        OPENCL_SUCCEED(clSetKernelArg(map_kernel_1132, 2,
                                      sizeof(world_mem_1245.mem),
                                      &world_mem_1245.mem));
        OPENCL_SUCCEED(clSetKernelArg(map_kernel_1132, 3, sizeof(mem_1243.mem),
                                      &mem_1243.mem));
        OPENCL_SUCCEED(clSetKernelArg(map_kernel_1132, 4, sizeof(n_699),
                                      &n_699));
        OPENCL_SUCCEED(clSetKernelArg(map_kernel_1132, 5, sizeof(mem_1250.mem),
                                      &mem_1250.mem));
        if (1 * (num_groups_1305 * group_size_1304) != 0) {
            const size_t global_work_size_1343[1] = {num_groups_1305 *
                         group_size_1304};
            const size_t local_work_size_1347[1] = {group_size_1304};
            int64_t time_start_1344, time_end_1345;
            
            if (cl_debug) {
                fprintf(stderr, "Launching %s with global work size [",
                        "map_kernel_1132");
                fprintf(stderr, "%zu", global_work_size_1343[0]);
                fprintf(stderr, "].\n");
                time_start_1344 = get_wall_time();
            }
            OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_1132,
                                                  1, NULL,
                                                  global_work_size_1343,
                                                  local_work_size_1347, 0, NULL,
                                                  NULL));
            if (cl_debug) {
                OPENCL_SUCCEED(clFinish(fut_cl_queue));
                time_end_1345 = get_wall_time();
                
                long time_diff_1346 = time_end_1345 - time_start_1344;
                
                if (detail_timing) {
                    map_kernel_1132total_runtime += time_diff_1346;
                    map_kernel_1132runs++;
                    fprintf(stderr, "kernel %s runtime: %ldus\n",
                            "map_kernel_1132", (int) time_diff_1346);
                }
            }
        }
        
        int32_t group_size_1306;
        int32_t num_groups_1307;
        
        group_size_1306 = cl_group_size;
        num_groups_1307 = squot32(n_699 * m_700 + group_size_1306 - 1,
                                  group_size_1306);
        OPENCL_SUCCEED(clSetKernelArg(map_kernel_1186, 0, sizeof(m_700),
                                      &m_700));
        OPENCL_SUCCEED(clSetKernelArg(map_kernel_1186, 1,
                                      sizeof(world_mem_1245.mem),
                                      &world_mem_1245.mem));
        OPENCL_SUCCEED(clSetKernelArg(map_kernel_1186, 2, sizeof(mem_1250.mem),
                                      &mem_1250.mem));
        OPENCL_SUCCEED(clSetKernelArg(map_kernel_1186, 3, sizeof(n_699),
                                      &n_699));
        OPENCL_SUCCEED(clSetKernelArg(map_kernel_1186, 4, sizeof(mem_1253.mem),
                                      &mem_1253.mem));
        if (1 * (num_groups_1307 * group_size_1306) != 0) {
            const size_t global_work_size_1348[1] = {num_groups_1307 *
                         group_size_1306};
            const size_t local_work_size_1352[1] = {group_size_1306};
            int64_t time_start_1349, time_end_1350;
            
            if (cl_debug) {
                fprintf(stderr, "Launching %s with global work size [",
                        "map_kernel_1186");
                fprintf(stderr, "%zu", global_work_size_1348[0]);
                fprintf(stderr, "].\n");
                time_start_1349 = get_wall_time();
            }
            OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_1186,
                                                  1, NULL,
                                                  global_work_size_1348,
                                                  local_work_size_1352, 0, NULL,
                                                  NULL));
            if (cl_debug) {
                OPENCL_SUCCEED(clFinish(fut_cl_queue));
                time_end_1350 = get_wall_time();
                
                long time_diff_1351 = time_end_1350 - time_start_1349;
                
                if (detail_timing) {
                    map_kernel_1186total_runtime += time_diff_1351;
                    map_kernel_1186runs++;
                    fprintf(stderr, "kernel %s runtime: %ldus\n",
                            "map_kernel_1186", (int) time_diff_1351);
                }
            }
        }
        
        int32_t group_size_1308;
        int32_t num_groups_1309;
        
        group_size_1308 = cl_group_size;
        num_groups_1309 = squot32(n_699 * m_700 + group_size_1308 - 1,
                                  group_size_1308);
        OPENCL_SUCCEED(clSetKernelArg(map_kernel_1204, 0, sizeof(m_700),
                                      &m_700));
        OPENCL_SUCCEED(clSetKernelArg(map_kernel_1204, 1, sizeof(mem_1253.mem),
                                      &mem_1253.mem));
        OPENCL_SUCCEED(clSetKernelArg(map_kernel_1204, 2, sizeof(n_699),
                                      &n_699));
        OPENCL_SUCCEED(clSetKernelArg(map_kernel_1204, 3,
                                      sizeof(history_mem_1247.mem),
                                      &history_mem_1247.mem));
        OPENCL_SUCCEED(clSetKernelArg(map_kernel_1204, 4, sizeof(mem_1256.mem),
                                      &mem_1256.mem));
        if (1 * (num_groups_1309 * group_size_1308) != 0) {
            const size_t global_work_size_1353[1] = {num_groups_1309 *
                         group_size_1308};
            const size_t local_work_size_1357[1] = {group_size_1308};
            int64_t time_start_1354, time_end_1355;
            
            if (cl_debug) {
                fprintf(stderr, "Launching %s with global work size [",
                        "map_kernel_1204");
                fprintf(stderr, "%zu", global_work_size_1353[0]);
                fprintf(stderr, "].\n");
                time_start_1354 = get_wall_time();
            }
            OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_1204,
                                                  1, NULL,
                                                  global_work_size_1353,
                                                  local_work_size_1357, 0, NULL,
                                                  NULL));
            if (cl_debug) {
                OPENCL_SUCCEED(clFinish(fut_cl_queue));
                time_end_1355 = get_wall_time();
                
                long time_diff_1356 = time_end_1355 - time_start_1354;
                
                if (detail_timing) {
                    map_kernel_1204total_runtime += time_diff_1356;
                    map_kernel_1204runs++;
                    fprintf(stderr, "kernel %s runtime: %ldus\n",
                            "map_kernel_1204", (int) time_diff_1356);
                }
            }
        }
        if (n_699 * m_700 * sizeof(char) > 0) {
            OPENCL_SUCCEED(clEnqueueCopyBuffer(fut_cl_queue, mem_1253.mem,
                                               double_buffer_mem_1263.mem, 0, 0,
                                               n_699 * m_700 * sizeof(char), 0,
                                               NULL, NULL));
            if (cl_debug)
                OPENCL_SUCCEED(clFinish(fut_cl_queue));
        }
        if (n_699 * m_700 * sizeof(int32_t) > 0) {
            OPENCL_SUCCEED(clEnqueueCopyBuffer(fut_cl_queue, mem_1256.mem,
                                               double_buffer_mem_1264.mem, 0, 0,
                                               n_699 * m_700 * sizeof(int32_t),
                                               0, NULL, NULL));
            if (cl_debug)
                OPENCL_SUCCEED(clFinish(fut_cl_queue));
        }
        
        int32_t world_mem_size_tmp_1298 = bytes_1251;
        int32_t history_mem_size_tmp_1299 = bytes_1248;
        struct memblock_device world_mem_tmp_1300;
        
        world_mem_tmp_1300.references = NULL;
        memblock_set_device(&world_mem_tmp_1300, &double_buffer_mem_1263);
        
        struct memblock_device history_mem_tmp_1301;
        
        history_mem_tmp_1301.references = NULL;
        memblock_set_device(&history_mem_tmp_1301, &double_buffer_mem_1264);
        world_mem_size_1244 = world_mem_size_tmp_1298;
        history_mem_size_1246 = history_mem_size_tmp_1299;
        memblock_set_device(&world_mem_1245, &world_mem_tmp_1300);
        memblock_set_device(&history_mem_1247, &history_mem_tmp_1301);
        memblock_unref_device(&world_mem_tmp_1300);
        memblock_unref_device(&history_mem_tmp_1301);
    }
    memblock_set_device(&world_mem_1258, &world_mem_1245);
    world_mem_size_1257 = world_mem_size_1244;
    memblock_set_device(&history_mem_1260, &history_mem_1247);
    history_mem_size_1259 = history_mem_size_1246;
    memblock_set_device(&out_mem_1292, &world_mem_1258);
    out_memsize_1293 = world_mem_size_1257;
    memblock_set_device(&out_mem_1294, &history_mem_1260);
    out_memsize_1295 = history_mem_size_1259;
    
    struct tuple_int32_t_device_mem_int32_t_device_mem retval_1337;
    
    retval_1337.elem_0 = out_memsize_1293;
    retval_1337.elem_1.references = NULL;
    memblock_set_device(&retval_1337.elem_1, &out_mem_1292);
    retval_1337.elem_2 = out_memsize_1295;
    retval_1337.elem_3.references = NULL;
    memblock_set_device(&retval_1337.elem_3, &out_mem_1294);
    memblock_unref_device(&out_mem_1292);
    memblock_unref_device(&out_mem_1294);
    memblock_unref_device(&mem_1241);
    memblock_unref_device(&mem_1243);
    memblock_unref_device(&mem_1250);
    memblock_unref_device(&double_buffer_mem_1263);
    memblock_unref_device(&double_buffer_mem_1264);
    memblock_unref_device(&mem_1253);
    memblock_unref_device(&mem_1256);
    memblock_unref_device(&world_mem_1258);
    memblock_unref_device(&history_mem_1260);
    memblock_unref_device(&world_mem_1245);
    memblock_unref_device(&history_mem_1247);
    return retval_1337;
}
struct array_reader {
  char* elems;
  int64_t n_elems_space;
  int64_t elem_size;
  int64_t n_elems_used;
  int64_t *shape;
  int (*elem_reader)(void*);
};

static int peekc() {
  int c = getchar();
  ungetc(c,stdin);
  return c;
}

static int next_is_not_constituent() {
  int c = peekc();
  return c == EOF || !isalnum(c);
}

static void skipspaces() {
  int c = getchar();
  if (isspace(c)) {
    skipspaces();
  } else if (c == '-' && peekc() == '-') {
    // Skip to end of line.
    for (; c != '\n' && c != EOF; c = getchar());
    // Next line may have more spaces.
    skipspaces();
  } else if (c != EOF) {
    ungetc(c, stdin);
  }
}

static int read_elem(struct array_reader *reader) {
  int ret;
  if (reader->n_elems_used == reader->n_elems_space) {
    reader->n_elems_space *= 2;
    reader->elems = (char*) realloc(reader->elems,
                                    reader->n_elems_space * reader->elem_size);
  }

  ret = reader->elem_reader(reader->elems + reader->n_elems_used * reader->elem_size);

  if (ret == 0) {
    reader->n_elems_used++;
  }

  return ret;
}

static int read_array_elems(struct array_reader *reader, int dims) {
  int c;
  int ret;
  int first = 1;
  char *knows_dimsize = (char*) calloc(dims,sizeof(char));
  int cur_dim = dims-1;
  int64_t *elems_read_in_dim = (int64_t*) calloc(dims,sizeof(int64_t));
  while (1) {
    skipspaces();

    c = getchar();
    if (c == ']') {
      if (knows_dimsize[cur_dim]) {
        if (reader->shape[cur_dim] != elems_read_in_dim[cur_dim]) {
          ret = 1;
          break;
        }
      } else {
        knows_dimsize[cur_dim] = 1;
        reader->shape[cur_dim] = elems_read_in_dim[cur_dim];
      }
      if (cur_dim == 0) {
        ret = 0;
        break;
      } else {
        cur_dim--;
        elems_read_in_dim[cur_dim]++;
      }
    } else if (c == ',') {
      skipspaces();
      c = getchar();
      if (c == '[') {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        first = 1;
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else if (cur_dim == dims - 1) {
        ungetc(c, stdin);
        ret = read_elem(reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
      } else {
        ret = 1;
        break;
      }
    } else if (c == EOF) {
      ret = 1;
      break;
    } else if (first) {
      if (c == '[') {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else {
        ungetc(c, stdin);
        ret = read_elem(reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
        first = 0;
      }
    } else {
      ret = 1;
      break;
    }
  }

  free(knows_dimsize);
  free(elems_read_in_dim);
  return ret;
}

static int read_array(int64_t elem_size, int (*elem_reader)(void*),
               void **data, int64_t *shape, int64_t dims) {
  int ret;
  struct array_reader reader;
  int64_t read_dims = 0;
  while (1) {
    int c;
    skipspaces();
    c = getchar();
    if (c=='[') {
      read_dims++;
    } else {
      if (c != EOF) {
        ungetc(c, stdin);
      }
      break;
    }
  }

  if (read_dims != dims) {
    return 1;
  }

  reader.shape = shape;
  reader.n_elems_used = 0;
  reader.elem_size = elem_size;
  reader.n_elems_space = 16;
  reader.elems = (char*) realloc(*data, elem_size*reader.n_elems_space);
  reader.elem_reader = elem_reader;

  ret = read_array_elems(&reader, dims);

  *data = reader.elems;

  return ret;
}

static int read_int8(void* dest) {
  skipspaces();
  if (scanf("%hhi", (int8_t*)dest) == 1) {
    scanf("i8");
    return next_is_not_constituent() ? 0 : 1;
  } else {
    return 1;
  }
}

static int read_int16(void* dest) {
  skipspaces();
  if (scanf("%hi", (int16_t*)dest) == 1) {
    scanf("i16");
    return next_is_not_constituent() ? 0 : 1;
  } else {
    return 1;
  }
}

static int read_int32(void* dest) {
  skipspaces();
  if (scanf("%i", (int32_t*)dest) == 1) {
    scanf("i32");
    return next_is_not_constituent() ? 0 : 1;
  } else {
    return 1;
  }
}

static int read_int64(void* dest) {
  skipspaces();
  if (scanf("%Li", (int64_t*)dest) == 1) {
    scanf("i64");
    return next_is_not_constituent() ? 0 : 1;
  } else {
    return 1;
  }
}

static int read_char(void* dest) {
  skipspaces();
  if (scanf("%c", (char*)dest) == 1) {
    return 0;
  } else {
    return 1;
  }
}

static int read_double(void* dest) {
  skipspaces();
  if (scanf("%lf", (double*)dest) == 1) {
    scanf("f64");
    return next_is_not_constituent() ? 0 : 1;
  } else {
    return 1;
  }
}

static int read_float(void* dest) {
  skipspaces();
  if (scanf("%f", (float*)dest) == 1) {
    scanf("f32");
    return next_is_not_constituent() ? 0 : 1;
  } else {
    return 1;
  }
}

static int read_bool(void* dest) {
  /* This is a monstrous hack.  Maybe we should get a proper lexer in here. */
  char b[4];
  skipspaces();
  if (scanf("%4c", b) == 1) {
    if (strncmp(b, "True", 4) == 0) {
      *(int*)dest = 1;
      return 0;
    } else if (strncmp(b, "Fals", 4) == 0 && getchar() == 'e') {
      *(int*)dest = 0;
      return 0;
    } else {
      return 1;
    }
  } else {
    return 1;
  }
}

static FILE *runtime_file;
static int perform_warmup = 0;
static int num_runs = 1;
int parse_options(int argc, char *const argv[])
{
    int ch;
    static struct option long_options[] = {{"write-runtime-to",
                                            required_argument, NULL, 1},
                                           {"runs", required_argument, NULL, 2},
                                           {"platform", required_argument, NULL,
                                            3}, {"device", required_argument,
                                                 NULL, 4}, {"synchronous",
                                                            no_argument, NULL,
                                                            5}, {"group-size",
                                                                 required_argument,
                                                                 NULL, 6},
                                           {"num-groups", required_argument,
                                            NULL, 7}, {0, 0, 0, 0}};
    
    while ((ch = getopt_long(argc, argv, ":t:r:p:d:s", long_options, NULL)) !=
           -1) {
        if (ch == 1 || ch == 't') {
            runtime_file = fopen(optarg, "w");
            if (runtime_file == NULL)
                panic(1, "Cannot open %s: %s", optarg, strerror(errno));
        }
        if (ch == 2 || ch == 'r') {
            num_runs = atoi(optarg);
            perform_warmup = 1;
            if (num_runs <= 0)
                panic(1, "Need a positive number of runs, not %s", optarg);
        }
        if (ch == 3 || ch == 'p')
            cl_preferred_platform = optarg;
        if (ch == 4 || ch == 'd')
            cl_preferred_device = optarg;
        if (ch == 5 || ch == 's')
            cl_debug = 1;
        if (ch == 6)
            cl_group_size = atoi(optarg);
        if (ch == 7)
            cl_num_groups = atoi(optarg);
        if (ch == ':')
            panic(-1, "Missing argument for option %s", argv[optind - 1]);
        if (ch == '?')
            panic(-1, "Unknown option %s", argv[optind - 1]);
    }
    return optind;
}
int main(int argc, char **argv)
{
    int64_t t_start, t_end;
    int time_runs;
    
    fut_progname = argv[0];
    
    int parsed_options = parse_options(argc, argv);
    
    argc -= parsed_options;
    argv += parsed_options;
    setup_opencl_and_load_kernels();
    
    int32_t main_ret_1358;
    int32_t scalar_out_1291;
    
    if (perform_warmup) {
        time_runs = 0;
        t_start = get_wall_time();
        main_ret_1358 = futhark_main();
        OPENCL_SUCCEED(clFinish(fut_cl_queue));
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%ld\n", elapsed_usec);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        if (run == num_runs - 1)
            detail_timing = 1;
        t_start = get_wall_time();
        main_ret_1358 = futhark_main();
        OPENCL_SUCCEED(clFinish(fut_cl_queue));
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%ld\n", elapsed_usec);
        if (run < num_runs - 1) { }
    }
    scalar_out_1291 = main_ret_1358;
    printf("%di32", scalar_out_1291);
    printf("\n");
    
    int total_runtime = 0;
    int total_runs = 0;
    
    if (cl_debug) {
        fprintf(stderr,
                "Kernel map_kernel_1274             executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_1274runs, (long) map_kernel_1274total_runtime /
                (map_kernel_1274runs != 0 ? map_kernel_1274runs : 1),
                (long) map_kernel_1274total_runtime);
        total_runtime += map_kernel_1274total_runtime;
        total_runs += map_kernel_1274runs;
        fprintf(stderr,
                "Kernel map_kernel_1278             executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_1278runs, (long) map_kernel_1278total_runtime /
                (map_kernel_1278runs != 0 ? map_kernel_1278runs : 1),
                (long) map_kernel_1278total_runtime);
        total_runtime += map_kernel_1278total_runtime;
        total_runs += map_kernel_1278runs;
        fprintf(stderr,
                "Kernel map_kernel_1107             executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_1107runs, (long) map_kernel_1107total_runtime /
                (map_kernel_1107runs != 0 ? map_kernel_1107runs : 1),
                (long) map_kernel_1107total_runtime);
        total_runtime += map_kernel_1107total_runtime;
        total_runs += map_kernel_1107runs;
        fprintf(stderr,
                "Kernel fut_kernel_map_transpose_i8 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                fut_kernel_map_transpose_i8runs,
                (long) fut_kernel_map_transpose_i8total_runtime /
                (fut_kernel_map_transpose_i8runs !=
                 0 ? fut_kernel_map_transpose_i8runs : 1),
                (long) fut_kernel_map_transpose_i8total_runtime);
        total_runtime += fut_kernel_map_transpose_i8total_runtime;
        total_runs += fut_kernel_map_transpose_i8runs;
        fprintf(stderr,
                "Kernel map_kernel_1168             executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_1168runs, (long) map_kernel_1168total_runtime /
                (map_kernel_1168runs != 0 ? map_kernel_1168runs : 1),
                (long) map_kernel_1168total_runtime);
        total_runtime += map_kernel_1168total_runtime;
        total_runs += map_kernel_1168runs;
        fprintf(stderr,
                "Kernel map_kernel_1132             executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_1132runs, (long) map_kernel_1132total_runtime /
                (map_kernel_1132runs != 0 ? map_kernel_1132runs : 1),
                (long) map_kernel_1132total_runtime);
        total_runtime += map_kernel_1132total_runtime;
        total_runs += map_kernel_1132runs;
        fprintf(stderr,
                "Kernel map_kernel_1186             executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_1186runs, (long) map_kernel_1186total_runtime /
                (map_kernel_1186runs != 0 ? map_kernel_1186runs : 1),
                (long) map_kernel_1186total_runtime);
        total_runtime += map_kernel_1186total_runtime;
        total_runs += map_kernel_1186runs;
        fprintf(stderr,
                "Kernel map_kernel_1204             executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_1204runs, (long) map_kernel_1204total_runtime /
                (map_kernel_1204runs != 0 ? map_kernel_1204runs : 1),
                (long) map_kernel_1204total_runtime);
        total_runtime += map_kernel_1204total_runtime;
        total_runs += map_kernel_1204runs;
    }
    if (cl_debug)
        fprintf(stderr, "Ran %d kernels with cumulative runtime: %6ldus\n",
                total_runs, total_runtime);
    if (runtime_file != NULL)
        fclose(runtime_file);
    return 0;
}
