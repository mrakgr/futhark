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
__kernel void map_kernel_4739(int32_t m_3943, __global unsigned char *mem_4666)
{
    const uint global_thread_index_4739 = get_global_id(0);
    
    if (global_thread_index_4739 >= m_3943)
        return;
    
    int32_t i_4740;
    
    // compute thread index
    {
        i_4740 = global_thread_index_4739;
    }
    // read kernel parameters
    { }
    // write kernel result
    {
        *(__global float *) &mem_4666[i_4740 * 4] = 0.0F;
    }
}
__kernel void map_kernel_4743(int32_t n_3945, __global unsigned char *mem_4666,
                              int32_t m_3943, __global unsigned char *mem_4669)
{
    const uint global_thread_index_4743 = get_global_id(0);
    
    if (global_thread_index_4743 >= n_3945 * m_3943)
        return;
    
    int32_t i_4744;
    int32_t j_4745;
    float input_4746;
    
    // compute thread index
    {
        i_4744 = squot32(global_thread_index_4743, m_3943);
        j_4745 = global_thread_index_4743 - squot32(global_thread_index_4743,
                                                    m_3943) * m_3943;
    }
    // read kernel parameters
    {
        input_4746 = *(__global float *) &mem_4666[j_4745 * 4];
    }
    // write kernel result
    {
        *(__global float *) &mem_4669[(i_4744 * m_3943 + j_4745) * 4] =
            input_4746;
    }
}
__kernel void map_kernel_4752(int32_t n_3945, __global unsigned char *mem_4671)
{
    const uint global_thread_index_4752 = get_global_id(0);
    
    if (global_thread_index_4752 >= n_3945)
        return;
    
    int32_t i_4753;
    
    // compute thread index
    {
        i_4753 = global_thread_index_4752;
    }
    // read kernel parameters
    { }
    // write kernel result
    {
        *(__global float *) &mem_4671[i_4753 * 4] = 0.0F;
    }
}
__kernel void map_kernel_4756(int32_t n_3945, __global unsigned char *mem_4671,
                              int32_t m_3943, __global unsigned char *mem_4674)
{
    const uint global_thread_index_4756 = get_global_id(0);
    
    if (global_thread_index_4756 >= m_3943 * n_3945)
        return;
    
    int32_t i_4757;
    int32_t j_4758;
    float input_4759;
    
    // compute thread index
    {
        i_4757 = squot32(global_thread_index_4756, n_3945);
        j_4758 = global_thread_index_4756 - squot32(global_thread_index_4756,
                                                    n_3945) * n_3945;
    }
    // read kernel parameters
    {
        input_4759 = *(__global float *) &mem_4671[j_4758 * 4];
    }
    // write kernel result
    {
        *(__global float *) &mem_4674[(i_4757 * n_3945 + j_4758) * 4] =
            input_4759;
    }
}
__kernel void map_kernel_4337(__global unsigned char *mem_4669, int32_t n_3945,
                              __global unsigned char *mem_4674, int32_t m_3943,
                              __global unsigned char *mem_4677)
{
    const uint kernel_thread_index_4337 = get_global_id(0);
    
    if (kernel_thread_index_4337 >= n_3945 * m_3943)
        return;
    
    int32_t i_4338;
    int32_t i_4339;
    float x_4340;
    float y_4341;
    
    // compute thread index
    {
        i_4338 = squot32(kernel_thread_index_4337, m_3943);
        i_4339 = kernel_thread_index_4337 - squot32(kernel_thread_index_4337,
                                                    m_3943) * m_3943;
    }
    // read kernel parameters
    {
        x_4340 = *(__global float *) &mem_4669[(i_4338 * m_3943 + i_4339) * 4];
        y_4341 = *(__global float *) &mem_4674[(i_4338 * n_3945 + i_4339) * 4];
    }
    
    float res_4342 = x_4340 + y_4341;
    
    // write kernel result
    {
        *(__global float *) &mem_4677[(i_4338 * m_3943 + i_4339) * 4] =
            res_4342;
    }
}
__kernel void map_kernel_4360(__global unsigned char *b_bi_mem_4640, __global
                              unsigned char *mem_4677, int32_t size_4005,
                              int32_t m_3943, __global unsigned char *mem_4680)
{
    const uint kernel_thread_index_4360 = get_global_id(0);
    
    if (kernel_thread_index_4360 >= m_3943 * size_4005)
        return;
    
    int32_t i_4361;
    int32_t i_4362;
    float x_4363;
    float y_4364;
    
    // compute thread index
    {
        i_4361 = squot32(kernel_thread_index_4360, size_4005);
        i_4362 = kernel_thread_index_4360 - squot32(kernel_thread_index_4360,
                                                    size_4005) * size_4005;
    }
    // read kernel parameters
    {
        x_4363 = *(__global float *) &mem_4677[(i_4361 * m_3943 + i_4362) * 4];
        y_4364 = *(__global float *) &b_bi_mem_4640[i_4361 * 4];
    }
    
    float res_4365 = x_4363 + y_4364;
    
    // write kernel result
    {
        *(__global float *) &mem_4680[(i_4361 * size_4005 + i_4362) * 4] =
            res_4365;
    }
}
__kernel void map_kernel_4377(__global unsigned char *mem_4680, int32_t n_3945,
                              int32_t size_4005, int32_t m_3943, __global
                              unsigned char *mem_4683)
{
    const uint kernel_thread_index_4377 = get_global_id(0);
    
    if (kernel_thread_index_4377 >= n_3945 * m_3943)
        return;
    
    int32_t i_4378;
    int32_t i_4379;
    float not_curried_4380;
    
    // compute thread index
    {
        i_4378 = squot32(kernel_thread_index_4377, m_3943);
        i_4379 = kernel_thread_index_4377 - squot32(kernel_thread_index_4377,
                                                    m_3943) * m_3943;
    }
    // read kernel parameters
    {
        not_curried_4380 = *(__global float *) &mem_4680[(i_4378 * size_4005 +
                                                          i_4379) * 4];
    }
    
    float arg_4381 = 0.0F - not_curried_4380;
    float res_4382 = fpow32(2.718280076980591F, arg_4381);
    float y_4383 = 1.0F + res_4382;
    float res_4384 = 1.0F / y_4383;
    
    // write kernel result
    {
        *(__global float *) &mem_4683[(i_4378 * m_3943 + i_4379) * 4] =
            res_4384;
    }
}
__kernel void map_kernel_4771(int32_t n_3945, __global unsigned char *mem_4666,
                              int32_t m_3943, __global unsigned char *mem_4686)
{
    const uint global_thread_index_4771 = get_global_id(0);
    
    if (global_thread_index_4771 >= n_3945 * m_3943)
        return;
    
    int32_t i_4772;
    int32_t j_4773;
    float input_4774;
    
    // compute thread index
    {
        i_4772 = squot32(global_thread_index_4771, m_3943);
        j_4773 = global_thread_index_4771 - squot32(global_thread_index_4771,
                                                    m_3943) * m_3943;
    }
    // read kernel parameters
    {
        input_4774 = *(__global float *) &mem_4666[j_4773 * 4];
    }
    // write kernel result
    {
        *(__global float *) &mem_4686[(i_4772 * m_3943 + j_4773) * 4] =
            input_4774;
    }
}
__kernel void map_kernel_4780(int32_t n_3945, __global unsigned char *mem_4671,
                              int32_t m_3943, __global unsigned char *mem_4689)
{
    const uint global_thread_index_4780 = get_global_id(0);
    
    if (global_thread_index_4780 >= m_3943 * n_3945)
        return;
    
    int32_t i_4781;
    int32_t j_4782;
    float input_4783;
    
    // compute thread index
    {
        i_4781 = squot32(global_thread_index_4780, n_3945);
        j_4782 = global_thread_index_4780 - squot32(global_thread_index_4780,
                                                    n_3945) * n_3945;
    }
    // read kernel parameters
    {
        input_4783 = *(__global float *) &mem_4671[j_4782 * 4];
    }
    // write kernel result
    {
        *(__global float *) &mem_4689[(i_4781 * n_3945 + j_4782) * 4] =
            input_4783;
    }
}
__kernel void map_kernel_4397(int32_t n_3945, __global unsigned char *mem_4689,
                              __global unsigned char *mem_4686, int32_t m_3943,
                              __global unsigned char *mem_4692)
{
    const uint kernel_thread_index_4397 = get_global_id(0);
    
    if (kernel_thread_index_4397 >= n_3945 * m_3943)
        return;
    
    int32_t i_4398;
    int32_t i_4399;
    float x_4400;
    float y_4401;
    
    // compute thread index
    {
        i_4398 = squot32(kernel_thread_index_4397, m_3943);
        i_4399 = kernel_thread_index_4397 - squot32(kernel_thread_index_4397,
                                                    m_3943) * m_3943;
    }
    // read kernel parameters
    {
        x_4400 = *(__global float *) &mem_4686[(i_4398 * m_3943 + i_4399) * 4];
        y_4401 = *(__global float *) &mem_4689[(i_4398 * n_3945 + i_4399) * 4];
    }
    
    float res_4402 = x_4400 + y_4401;
    
    // write kernel result
    {
        *(__global float *) &mem_4692[(i_4398 * m_3943 + i_4399) * 4] =
            res_4402;
    }
}
__kernel void map_kernel_4420(__global unsigned char *mem_4692,
                              int32_t size_4005, __global
                              unsigned char *b_ig_mem_4646, int32_t m_3943,
                              __global unsigned char *mem_4695)
{
    const uint kernel_thread_index_4420 = get_global_id(0);
    
    if (kernel_thread_index_4420 >= m_3943 * size_4005)
        return;
    
    int32_t i_4421;
    int32_t i_4422;
    float x_4423;
    float y_4424;
    
    // compute thread index
    {
        i_4421 = squot32(kernel_thread_index_4420, size_4005);
        i_4422 = kernel_thread_index_4420 - squot32(kernel_thread_index_4420,
                                                    size_4005) * size_4005;
    }
    // read kernel parameters
    {
        x_4423 = *(__global float *) &mem_4692[(i_4421 * m_3943 + i_4422) * 4];
        y_4424 = *(__global float *) &b_ig_mem_4646[i_4421 * 4];
    }
    
    float res_4425 = x_4423 + y_4424;
    
    // write kernel result
    {
        *(__global float *) &mem_4695[(i_4421 * size_4005 + i_4422) * 4] =
            res_4425;
    }
}
__kernel void map_kernel_4437(int32_t n_3945, int32_t size_4005, __global
                              unsigned char *mem_4695, int32_t m_3943, __global
                              unsigned char *mem_4698)
{
    const uint kernel_thread_index_4437 = get_global_id(0);
    
    if (kernel_thread_index_4437 >= n_3945 * m_3943)
        return;
    
    int32_t i_4438;
    int32_t i_4439;
    float not_curried_4440;
    
    // compute thread index
    {
        i_4438 = squot32(kernel_thread_index_4437, m_3943);
        i_4439 = kernel_thread_index_4437 - squot32(kernel_thread_index_4437,
                                                    m_3943) * m_3943;
    }
    // read kernel parameters
    {
        not_curried_4440 = *(__global float *) &mem_4695[(i_4438 * size_4005 +
                                                          i_4439) * 4];
    }
    
    float arg_4441 = 0.0F - not_curried_4440;
    float res_4442 = fpow32(2.718280076980591F, arg_4441);
    float y_4443 = 1.0F + res_4442;
    float res_4444 = 1.0F / y_4443;
    
    // write kernel result
    {
        *(__global float *) &mem_4698[(i_4438 * m_3943 + i_4439) * 4] =
            res_4444;
    }
}
__kernel void map_kernel_4795(int32_t n_3945, __global unsigned char *mem_4666,
                              int32_t m_3943, __global unsigned char *mem_4701)
{
    const uint global_thread_index_4795 = get_global_id(0);
    
    if (global_thread_index_4795 >= n_3945 * m_3943)
        return;
    
    int32_t i_4796;
    int32_t j_4797;
    float input_4798;
    
    // compute thread index
    {
        i_4796 = squot32(global_thread_index_4795, m_3943);
        j_4797 = global_thread_index_4795 - squot32(global_thread_index_4795,
                                                    m_3943) * m_3943;
    }
    // read kernel parameters
    {
        input_4798 = *(__global float *) &mem_4666[j_4797 * 4];
    }
    // write kernel result
    {
        *(__global float *) &mem_4701[(i_4796 * m_3943 + j_4797) * 4] =
            input_4798;
    }
}
__kernel void map_kernel_4804(int32_t n_3945, __global unsigned char *mem_4671,
                              int32_t m_3943, __global unsigned char *mem_4704)
{
    const uint global_thread_index_4804 = get_global_id(0);
    
    if (global_thread_index_4804 >= m_3943 * n_3945)
        return;
    
    int32_t i_4805;
    int32_t j_4806;
    float input_4807;
    
    // compute thread index
    {
        i_4805 = squot32(global_thread_index_4804, n_3945);
        j_4806 = global_thread_index_4804 - squot32(global_thread_index_4804,
                                                    n_3945) * n_3945;
    }
    // read kernel parameters
    {
        input_4807 = *(__global float *) &mem_4671[j_4806 * 4];
    }
    // write kernel result
    {
        *(__global float *) &mem_4704[(i_4805 * n_3945 + j_4806) * 4] =
            input_4807;
    }
}
__kernel void map_kernel_4457(__global unsigned char *mem_4704, int32_t n_3945,
                              __global unsigned char *mem_4701, int32_t m_3943,
                              __global unsigned char *mem_4707)
{
    const uint kernel_thread_index_4457 = get_global_id(0);
    
    if (kernel_thread_index_4457 >= n_3945 * m_3943)
        return;
    
    int32_t i_4458;
    int32_t i_4459;
    float x_4460;
    float y_4461;
    
    // compute thread index
    {
        i_4458 = squot32(kernel_thread_index_4457, m_3943);
        i_4459 = kernel_thread_index_4457 - squot32(kernel_thread_index_4457,
                                                    m_3943) * m_3943;
    }
    // read kernel parameters
    {
        x_4460 = *(__global float *) &mem_4701[(i_4458 * m_3943 + i_4459) * 4];
        y_4461 = *(__global float *) &mem_4704[(i_4458 * n_3945 + i_4459) * 4];
    }
    
    float res_4462 = x_4460 + y_4461;
    
    // write kernel result
    {
        *(__global float *) &mem_4707[(i_4458 * m_3943 + i_4459) * 4] =
            res_4462;
    }
}
__kernel void map_kernel_4480(__global unsigned char *b_fg_mem_4652,
                              int32_t size_4005, __global
                              unsigned char *mem_4707, int32_t m_3943, __global
                              unsigned char *mem_4710)
{
    const uint kernel_thread_index_4480 = get_global_id(0);
    
    if (kernel_thread_index_4480 >= m_3943 * size_4005)
        return;
    
    int32_t i_4481;
    int32_t i_4482;
    float x_4483;
    float y_4484;
    
    // compute thread index
    {
        i_4481 = squot32(kernel_thread_index_4480, size_4005);
        i_4482 = kernel_thread_index_4480 - squot32(kernel_thread_index_4480,
                                                    size_4005) * size_4005;
    }
    // read kernel parameters
    {
        x_4483 = *(__global float *) &mem_4707[(i_4481 * m_3943 + i_4482) * 4];
        y_4484 = *(__global float *) &b_fg_mem_4652[i_4481 * 4];
    }
    
    float res_4485 = x_4483 + y_4484;
    
    // write kernel result
    {
        *(__global float *) &mem_4710[(i_4481 * size_4005 + i_4482) * 4] =
            res_4485;
    }
}
__kernel void map_kernel_4497(int32_t n_3945, int32_t size_4005, __global
                              unsigned char *mem_4710, int32_t m_3943, __global
                              unsigned char *mem_4713)
{
    const uint kernel_thread_index_4497 = get_global_id(0);
    
    if (kernel_thread_index_4497 >= n_3945 * m_3943)
        return;
    
    int32_t i_4498;
    int32_t i_4499;
    float not_curried_4500;
    
    // compute thread index
    {
        i_4498 = squot32(kernel_thread_index_4497, m_3943);
        i_4499 = kernel_thread_index_4497 - squot32(kernel_thread_index_4497,
                                                    m_3943) * m_3943;
    }
    // read kernel parameters
    {
        not_curried_4500 = *(__global float *) &mem_4710[(i_4498 * size_4005 +
                                                          i_4499) * 4];
    }
    
    float arg_4501 = 0.0F - not_curried_4500;
    float res_4502 = fpow32(2.718280076980591F, arg_4501);
    float y_4503 = 1.0F + res_4502;
    float res_4504 = 1.0F / y_4503;
    
    // write kernel result
    {
        *(__global float *) &mem_4713[(i_4498 * m_3943 + i_4499) * 4] =
            res_4504;
    }
}
__kernel void map_kernel_4819(int32_t n_3945, __global unsigned char *mem_4666,
                              int32_t m_3943, __global unsigned char *mem_4716)
{
    const uint global_thread_index_4819 = get_global_id(0);
    
    if (global_thread_index_4819 >= n_3945 * m_3943)
        return;
    
    int32_t i_4820;
    int32_t j_4821;
    float input_4822;
    
    // compute thread index
    {
        i_4820 = squot32(global_thread_index_4819, m_3943);
        j_4821 = global_thread_index_4819 - squot32(global_thread_index_4819,
                                                    m_3943) * m_3943;
    }
    // read kernel parameters
    {
        input_4822 = *(__global float *) &mem_4666[j_4821 * 4];
    }
    // write kernel result
    {
        *(__global float *) &mem_4716[(i_4820 * m_3943 + j_4821) * 4] =
            input_4822;
    }
}
__kernel void map_kernel_4828(int32_t n_3945, __global unsigned char *mem_4671,
                              int32_t m_3943, __global unsigned char *mem_4719)
{
    const uint global_thread_index_4828 = get_global_id(0);
    
    if (global_thread_index_4828 >= m_3943 * n_3945)
        return;
    
    int32_t i_4829;
    int32_t j_4830;
    float input_4831;
    
    // compute thread index
    {
        i_4829 = squot32(global_thread_index_4828, n_3945);
        j_4830 = global_thread_index_4828 - squot32(global_thread_index_4828,
                                                    n_3945) * n_3945;
    }
    // read kernel parameters
    {
        input_4831 = *(__global float *) &mem_4671[j_4830 * 4];
    }
    // write kernel result
    {
        *(__global float *) &mem_4719[(i_4829 * n_3945 + j_4830) * 4] =
            input_4831;
    }
}
__kernel void map_kernel_4517(__global unsigned char *mem_4716, int32_t n_3945,
                              __global unsigned char *mem_4719, int32_t m_3943,
                              __global unsigned char *mem_4722)
{
    const uint kernel_thread_index_4517 = get_global_id(0);
    
    if (kernel_thread_index_4517 >= n_3945 * m_3943)
        return;
    
    int32_t i_4518;
    int32_t i_4519;
    float x_4520;
    float y_4521;
    
    // compute thread index
    {
        i_4518 = squot32(kernel_thread_index_4517, m_3943);
        i_4519 = kernel_thread_index_4517 - squot32(kernel_thread_index_4517,
                                                    m_3943) * m_3943;
    }
    // read kernel parameters
    {
        x_4520 = *(__global float *) &mem_4716[(i_4518 * m_3943 + i_4519) * 4];
        y_4521 = *(__global float *) &mem_4719[(i_4518 * n_3945 + i_4519) * 4];
    }
    
    float res_4522 = x_4520 + y_4521;
    
    // write kernel result
    {
        *(__global float *) &mem_4722[(i_4518 * m_3943 + i_4519) * 4] =
            res_4522;
    }
}
__kernel void map_kernel_4540(int32_t size_4005, __global
                              unsigned char *mem_4722, __global
                              unsigned char *b_og_mem_4658, int32_t m_3943,
                              __global unsigned char *mem_4725)
{
    const uint kernel_thread_index_4540 = get_global_id(0);
    
    if (kernel_thread_index_4540 >= m_3943 * size_4005)
        return;
    
    int32_t i_4541;
    int32_t i_4542;
    float x_4543;
    float y_4544;
    
    // compute thread index
    {
        i_4541 = squot32(kernel_thread_index_4540, size_4005);
        i_4542 = kernel_thread_index_4540 - squot32(kernel_thread_index_4540,
                                                    size_4005) * size_4005;
    }
    // read kernel parameters
    {
        x_4543 = *(__global float *) &mem_4722[(i_4541 * m_3943 + i_4542) * 4];
        y_4544 = *(__global float *) &b_og_mem_4658[i_4541 * 4];
    }
    
    float res_4545 = x_4543 + y_4544;
    
    // write kernel result
    {
        *(__global float *) &mem_4725[(i_4541 * size_4005 + i_4542) * 4] =
            res_4545;
    }
}
__kernel void map_kernel_4557(int32_t n_3945, int32_t size_4005, __global
                              unsigned char *mem_4725, int32_t m_3943, __global
                              unsigned char *mem_4728)
{
    const uint kernel_thread_index_4557 = get_global_id(0);
    
    if (kernel_thread_index_4557 >= n_3945 * m_3943)
        return;
    
    int32_t i_4558;
    int32_t i_4559;
    float not_curried_4560;
    
    // compute thread index
    {
        i_4558 = squot32(kernel_thread_index_4557, m_3943);
        i_4559 = kernel_thread_index_4557 - squot32(kernel_thread_index_4557,
                                                    m_3943) * m_3943;
    }
    // read kernel parameters
    {
        not_curried_4560 = *(__global float *) &mem_4725[(i_4558 * size_4005 +
                                                          i_4559) * 4];
    }
    
    float arg_4561 = 0.0F - not_curried_4560;
    float res_4562 = fpow32(2.718280076980591F, arg_4561);
    float y_4563 = 1.0F + res_4562;
    float res_4564 = 1.0F / y_4563;
    
    // write kernel result
    {
        *(__global float *) &mem_4728[(i_4558 * m_3943 + i_4559) * 4] =
            res_4564;
    }
}
__kernel void map_kernel_4601(__global unsigned char *prev_cell_mem_4664,
                              __global unsigned char *mem_4713, int32_t n_3945,
                              int32_t size_4005, __global
                              unsigned char *mem_4698, __global
                              unsigned char *mem_4683, int32_t m_3943, __global
                              unsigned char *mem_4731)
{
    const uint kernel_thread_index_4601 = get_global_id(0);
    
    if (kernel_thread_index_4601 >= n_3945 * size_4005)
        return;
    
    int32_t i_4602;
    int32_t i_4603;
    float x_4604;
    float x_4605;
    float y_4606;
    float y_4607;
    
    // compute thread index
    {
        i_4602 = squot32(kernel_thread_index_4601, size_4005);
        i_4603 = kernel_thread_index_4601 - squot32(kernel_thread_index_4601,
                                                    size_4005) * size_4005;
    }
    // read kernel parameters
    {
        x_4604 = *(__global float *) &prev_cell_mem_4664[(i_4602 * m_3943 +
                                                          i_4603) * 4];
        x_4605 = *(__global float *) &mem_4683[(i_4602 * m_3943 + i_4603) * 4];
        y_4606 = *(__global float *) &mem_4713[(i_4602 * m_3943 + i_4603) * 4];
        y_4607 = *(__global float *) &mem_4698[(i_4602 * m_3943 + i_4603) * 4];
    }
    
    float res_4608 = x_4605 * y_4607;
    float res_4609 = x_4604 * y_4606;
    float res_4610 = res_4608 + res_4609;
    
    // write kernel result
    {
        *(__global float *) &mem_4731[(i_4602 * size_4005 + i_4603) * 4] =
            res_4610;
    }
}
__kernel void map_kernel_4577(__global unsigned char *mem_4728, int32_t n_3945,
                              int32_t size_4005, __global
                              unsigned char *mem_4731, int32_t m_3943, __global
                              unsigned char *mem_4734)
{
    const uint kernel_thread_index_4577 = get_global_id(0);
    
    if (kernel_thread_index_4577 >= n_3945 * m_3943)
        return;
    
    int32_t i_4578;
    int32_t i_4579;
    float not_curried_4580;
    float x_4581;
    
    // compute thread index
    {
        i_4578 = squot32(kernel_thread_index_4577, m_3943);
        i_4579 = kernel_thread_index_4577 - squot32(kernel_thread_index_4577,
                                                    m_3943) * m_3943;
    }
    // read kernel parameters
    {
        not_curried_4580 = *(__global float *) &mem_4731[(i_4578 * size_4005 +
                                                          i_4579) * 4];
        x_4581 = *(__global float *) &mem_4728[(i_4578 * m_3943 + i_4579) * 4];
    }
    
    float arg_4582 = 0.0F - not_curried_4580;
    float res_4583 = fpow32(2.718280076980591F, arg_4582);
    float y_4584 = 1.0F + res_4583;
    float res_4585 = 1.0F / y_4584;
    float res_4586 = x_4581 * res_4585;
    
    // write kernel result
    {
        *(__global float *) &mem_4734[(i_4578 * m_3943 + i_4579) * 4] =
            res_4586;
    }
}
);
static cl_kernel map_kernel_4739;
static int map_kernel_4739total_runtime = 0;
static int map_kernel_4739runs = 0;
static cl_kernel map_kernel_4743;
static int map_kernel_4743total_runtime = 0;
static int map_kernel_4743runs = 0;
static cl_kernel map_kernel_4752;
static int map_kernel_4752total_runtime = 0;
static int map_kernel_4752runs = 0;
static cl_kernel map_kernel_4756;
static int map_kernel_4756total_runtime = 0;
static int map_kernel_4756runs = 0;
static cl_kernel map_kernel_4337;
static int map_kernel_4337total_runtime = 0;
static int map_kernel_4337runs = 0;
static cl_kernel map_kernel_4360;
static int map_kernel_4360total_runtime = 0;
static int map_kernel_4360runs = 0;
static cl_kernel map_kernel_4377;
static int map_kernel_4377total_runtime = 0;
static int map_kernel_4377runs = 0;
static cl_kernel map_kernel_4771;
static int map_kernel_4771total_runtime = 0;
static int map_kernel_4771runs = 0;
static cl_kernel map_kernel_4780;
static int map_kernel_4780total_runtime = 0;
static int map_kernel_4780runs = 0;
static cl_kernel map_kernel_4397;
static int map_kernel_4397total_runtime = 0;
static int map_kernel_4397runs = 0;
static cl_kernel map_kernel_4420;
static int map_kernel_4420total_runtime = 0;
static int map_kernel_4420runs = 0;
static cl_kernel map_kernel_4437;
static int map_kernel_4437total_runtime = 0;
static int map_kernel_4437runs = 0;
static cl_kernel map_kernel_4795;
static int map_kernel_4795total_runtime = 0;
static int map_kernel_4795runs = 0;
static cl_kernel map_kernel_4804;
static int map_kernel_4804total_runtime = 0;
static int map_kernel_4804runs = 0;
static cl_kernel map_kernel_4457;
static int map_kernel_4457total_runtime = 0;
static int map_kernel_4457runs = 0;
static cl_kernel map_kernel_4480;
static int map_kernel_4480total_runtime = 0;
static int map_kernel_4480runs = 0;
static cl_kernel map_kernel_4497;
static int map_kernel_4497total_runtime = 0;
static int map_kernel_4497runs = 0;
static cl_kernel map_kernel_4819;
static int map_kernel_4819total_runtime = 0;
static int map_kernel_4819runs = 0;
static cl_kernel map_kernel_4828;
static int map_kernel_4828total_runtime = 0;
static int map_kernel_4828runs = 0;
static cl_kernel map_kernel_4517;
static int map_kernel_4517total_runtime = 0;
static int map_kernel_4517runs = 0;
static cl_kernel map_kernel_4540;
static int map_kernel_4540total_runtime = 0;
static int map_kernel_4540runs = 0;
static cl_kernel map_kernel_4557;
static int map_kernel_4557total_runtime = 0;
static int map_kernel_4557runs = 0;
static cl_kernel map_kernel_4601;
static int map_kernel_4601total_runtime = 0;
static int map_kernel_4601runs = 0;
static cl_kernel map_kernel_4577;
static int map_kernel_4577total_runtime = 0;
static int map_kernel_4577runs = 0;
void setup_opencl_and_load_kernels()

{
    cl_int error;
    cl_program prog = setup_opencl(fut_opencl_prelude, fut_opencl_program);
    
    {
        map_kernel_4739 = clCreateKernel(prog, "map_kernel_4739", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4739");
    }
    {
        map_kernel_4743 = clCreateKernel(prog, "map_kernel_4743", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4743");
    }
    {
        map_kernel_4752 = clCreateKernel(prog, "map_kernel_4752", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4752");
    }
    {
        map_kernel_4756 = clCreateKernel(prog, "map_kernel_4756", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4756");
    }
    {
        map_kernel_4337 = clCreateKernel(prog, "map_kernel_4337", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4337");
    }
    {
        map_kernel_4360 = clCreateKernel(prog, "map_kernel_4360", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4360");
    }
    {
        map_kernel_4377 = clCreateKernel(prog, "map_kernel_4377", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4377");
    }
    {
        map_kernel_4771 = clCreateKernel(prog, "map_kernel_4771", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4771");
    }
    {
        map_kernel_4780 = clCreateKernel(prog, "map_kernel_4780", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4780");
    }
    {
        map_kernel_4397 = clCreateKernel(prog, "map_kernel_4397", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4397");
    }
    {
        map_kernel_4420 = clCreateKernel(prog, "map_kernel_4420", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4420");
    }
    {
        map_kernel_4437 = clCreateKernel(prog, "map_kernel_4437", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4437");
    }
    {
        map_kernel_4795 = clCreateKernel(prog, "map_kernel_4795", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4795");
    }
    {
        map_kernel_4804 = clCreateKernel(prog, "map_kernel_4804", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4804");
    }
    {
        map_kernel_4457 = clCreateKernel(prog, "map_kernel_4457", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4457");
    }
    {
        map_kernel_4480 = clCreateKernel(prog, "map_kernel_4480", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4480");
    }
    {
        map_kernel_4497 = clCreateKernel(prog, "map_kernel_4497", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4497");
    }
    {
        map_kernel_4819 = clCreateKernel(prog, "map_kernel_4819", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4819");
    }
    {
        map_kernel_4828 = clCreateKernel(prog, "map_kernel_4828", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4828");
    }
    {
        map_kernel_4517 = clCreateKernel(prog, "map_kernel_4517", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4517");
    }
    {
        map_kernel_4540 = clCreateKernel(prog, "map_kernel_4540", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4540");
    }
    {
        map_kernel_4557 = clCreateKernel(prog, "map_kernel_4557", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4557");
    }
    {
        map_kernel_4601 = clCreateKernel(prog, "map_kernel_4601", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4601");
    }
    {
        map_kernel_4577 = clCreateKernel(prog, "map_kernel_4577", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_4577");
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
    
    cl_int clCreateBuffer_succeeded_5016;
    
    block->mem = clCreateBuffer(fut_cl_context, CL_MEM_READ_WRITE, size >
                                0 ? size : 1, NULL,
                                &clCreateBuffer_succeeded_5016);
    OPENCL_SUCCEED(clCreateBuffer_succeeded_5016);
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
static struct tuple_int32_t_device_mem_int32_t_device_mem
futhark_main(int32_t W_bi_mem_size_4635, int32_t U_bi_mem_size_4637, int32_t b_bi_mem_size_4639, int32_t W_ig_mem_size_4641, int32_t U_ig_mem_size_4643, int32_t b_ig_mem_size_4645, int32_t W_fg_mem_size_4647, int32_t U_fg_mem_size_4649, int32_t b_fg_mem_size_4651, int32_t W_og_mem_size_4653, int32_t U_og_mem_size_4655, int32_t b_og_mem_size_4657, int32_t input_mem_size_4659, int32_t prev_output_mem_size_4661, int32_t prev_cell_mem_size_4663, struct memblock_device W_bi_mem_4636, struct memblock_device U_bi_mem_4638, struct memblock_device b_bi_mem_4640, struct memblock_device W_ig_mem_4642, struct memblock_device U_ig_mem_4644, struct memblock_device b_ig_mem_4646, struct memblock_device W_fg_mem_4648, struct memblock_device U_fg_mem_4650, struct memblock_device b_fg_mem_4652, struct memblock_device W_og_mem_4654, struct memblock_device U_og_mem_4656, struct memblock_device b_og_mem_4658, struct memblock_device input_mem_4660, struct memblock_device prev_output_mem_4662, struct memblock_device prev_cell_mem_4664, int32_t m_3943, int32_t o_3944, int32_t n_3945);
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
static inline float futhark_atan2_32(float x, float y)
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
struct tuple_int32_t_device_mem_int32_t_device_mem futhark_main(int32_t W_bi_mem_size_4635,
                                                                int32_t U_bi_mem_size_4637,
                                                                int32_t b_bi_mem_size_4639,
                                                                int32_t W_ig_mem_size_4641,
                                                                int32_t U_ig_mem_size_4643,
                                                                int32_t b_ig_mem_size_4645,
                                                                int32_t W_fg_mem_size_4647,
                                                                int32_t U_fg_mem_size_4649,
                                                                int32_t b_fg_mem_size_4651,
                                                                int32_t W_og_mem_size_4653,
                                                                int32_t U_og_mem_size_4655,
                                                                int32_t b_og_mem_size_4657,
                                                                int32_t input_mem_size_4659,
                                                                int32_t prev_output_mem_size_4661,
                                                                int32_t prev_cell_mem_size_4663,
                                                                struct memblock_device W_bi_mem_4636,
                                                                struct memblock_device U_bi_mem_4638,
                                                                struct memblock_device b_bi_mem_4640,
                                                                struct memblock_device W_ig_mem_4642,
                                                                struct memblock_device U_ig_mem_4644,
                                                                struct memblock_device b_ig_mem_4646,
                                                                struct memblock_device W_fg_mem_4648,
                                                                struct memblock_device U_fg_mem_4650,
                                                                struct memblock_device b_fg_mem_4652,
                                                                struct memblock_device W_og_mem_4654,
                                                                struct memblock_device U_og_mem_4656,
                                                                struct memblock_device b_og_mem_4658,
                                                                struct memblock_device input_mem_4660,
                                                                struct memblock_device prev_output_mem_4662,
                                                                struct memblock_device prev_cell_mem_4664,
                                                                int32_t m_3943,
                                                                int32_t o_3944,
                                                                int32_t n_3945)
{
    int32_t out_memsize_4736;
    struct memblock_device out_mem_4735;
    
    out_mem_4735.references = NULL;
    
    int32_t out_memsize_4738;
    struct memblock_device out_mem_4737;
    
    out_mem_4737.references = NULL;
    
    int32_t bytes_4665 = 4 * m_3943;
    struct memblock_device mem_4666;
    
    mem_4666.references = NULL;
    memblock_alloc_device(&mem_4666, bytes_4665);
    
    int32_t group_size_4741;
    int32_t num_groups_4742;
    
    group_size_4741 = cl_group_size;
    num_groups_4742 = squot32(m_3943 + group_size_4741 - 1, group_size_4741);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4739, 0, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4739, 1, sizeof(mem_4666.mem),
                                  &mem_4666.mem));
    if (1 * (num_groups_4742 * group_size_4741) != 0) {
        const size_t global_work_size_4848[1] = {num_groups_4742 *
                     group_size_4741};
        const size_t local_work_size_4852[1] = {group_size_4741};
        int64_t time_start_4849, time_end_4850;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4739");
            fprintf(stderr, "%zu", global_work_size_4848[0]);
            fprintf(stderr, "].\n");
            time_start_4849 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4739, 1,
                                              NULL, global_work_size_4848,
                                              local_work_size_4852, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4850 = get_wall_time();
            
            long time_diff_4851 = time_end_4850 - time_start_4849;
            
            if (detail_timing) {
                map_kernel_4739total_runtime += time_diff_4851;
                map_kernel_4739runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4739",
                        (int) time_diff_4851);
            }
        }
    }
    
    int32_t x_4668 = 4 * n_3945;
    int32_t bytes_4667 = x_4668 * m_3943;
    struct memblock_device mem_4669;
    
    mem_4669.references = NULL;
    memblock_alloc_device(&mem_4669, bytes_4667);
    
    int32_t group_size_4747;
    int32_t num_groups_4748;
    
    group_size_4747 = cl_group_size;
    num_groups_4748 = squot32(n_3945 * m_3943 + group_size_4747 - 1,
                              group_size_4747);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4743, 0, sizeof(n_3945), &n_3945));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4743, 1, sizeof(mem_4666.mem),
                                  &mem_4666.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4743, 2, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4743, 3, sizeof(mem_4669.mem),
                                  &mem_4669.mem));
    if (1 * (num_groups_4748 * group_size_4747) != 0) {
        const size_t global_work_size_4853[1] = {num_groups_4748 *
                     group_size_4747};
        const size_t local_work_size_4857[1] = {group_size_4747};
        int64_t time_start_4854, time_end_4855;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4743");
            fprintf(stderr, "%zu", global_work_size_4853[0]);
            fprintf(stderr, "].\n");
            time_start_4854 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4743, 1,
                                              NULL, global_work_size_4853,
                                              local_work_size_4857, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4855 = get_wall_time();
            
            long time_diff_4856 = time_end_4855 - time_start_4854;
            
            if (detail_timing) {
                map_kernel_4743total_runtime += time_diff_4856;
                map_kernel_4743runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4743",
                        (int) time_diff_4856);
            }
        }
    }
    for (int i_3964 = 0; i_3964 < m_3943; i_3964++) {
        char y_3965 = slt32(i_3964, n_3945);
        char bounds_check_3966;
        
        if (!y_3965) {
            fprintf(stderr, "Assertion %s at %s failed.\n", "y_3965",
                    "lstm.fut:12:16-12:16");
            abort();
        }
        for (int j_3968 = 0; j_3968 < n_3945; j_3968++) {
            float res_3975;
            float res_3969 = 0.0F;
            
            for (int k_3970 = 0; k_3970 < o_3944; k_3970++) {
                float read_res_4858;
                
                OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                                   W_bi_mem_4636.mem, CL_TRUE,
                                                   (i_3964 * o_3944 + k_3970) *
                                                   4, sizeof(float),
                                                   &read_res_4858, 0, NULL,
                                                   NULL));
                
                float x_3971 = read_res_4858;
                float read_res_4859;
                
                OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                                   input_mem_4660.mem, CL_TRUE,
                                                   (k_3970 * n_3945 + j_3968) *
                                                   4, sizeof(float),
                                                   &read_res_4859, 0, NULL,
                                                   NULL));
                
                float y_3972 = read_res_4859;
                float y_3973 = x_3971 * y_3972;
                float res_3974 = res_3969 + y_3973;
                float res_tmp_4751 = res_3974;
                
                res_3969 = res_tmp_4751;
            }
            res_3975 = res_3969;
            
            char y_3976 = slt32(j_3968, m_3943);
            char bounds_check_3977;
            
            if (!y_3976) {
                fprintf(stderr, "Assertion %s at %s failed.\n", "y_3976",
                        "lstm.fut:12:16-12:16");
                abort();
            }
            
            float write_tmp_4860 = res_3975;
            
            OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue, mem_4669.mem,
                                                CL_TRUE, (i_3964 * m_3943 +
                                                          j_3968) * 4,
                                                sizeof(float), &write_tmp_4860,
                                                0, NULL, NULL));
        }
    }
    
    struct memblock_device mem_4671;
    
    mem_4671.references = NULL;
    memblock_alloc_device(&mem_4671, x_4668);
    
    int32_t group_size_4754;
    int32_t num_groups_4755;
    
    group_size_4754 = cl_group_size;
    num_groups_4755 = squot32(n_3945 + group_size_4754 - 1, group_size_4754);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4752, 0, sizeof(n_3945), &n_3945));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4752, 1, sizeof(mem_4671.mem),
                                  &mem_4671.mem));
    if (1 * (num_groups_4755 * group_size_4754) != 0) {
        const size_t global_work_size_4861[1] = {num_groups_4755 *
                     group_size_4754};
        const size_t local_work_size_4865[1] = {group_size_4754};
        int64_t time_start_4862, time_end_4863;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4752");
            fprintf(stderr, "%zu", global_work_size_4861[0]);
            fprintf(stderr, "].\n");
            time_start_4862 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4752, 1,
                                              NULL, global_work_size_4861,
                                              local_work_size_4865, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4863 = get_wall_time();
            
            long time_diff_4864 = time_end_4863 - time_start_4862;
            
            if (detail_timing) {
                map_kernel_4752total_runtime += time_diff_4864;
                map_kernel_4752runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4752",
                        (int) time_diff_4864);
            }
        }
    }
    
    int32_t bytes_4672 = bytes_4665 * n_3945;
    struct memblock_device mem_4674;
    
    mem_4674.references = NULL;
    memblock_alloc_device(&mem_4674, bytes_4672);
    
    int32_t group_size_4760;
    int32_t num_groups_4761;
    
    group_size_4760 = cl_group_size;
    num_groups_4761 = squot32(m_3943 * n_3945 + group_size_4760 - 1,
                              group_size_4760);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4756, 0, sizeof(n_3945), &n_3945));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4756, 1, sizeof(mem_4671.mem),
                                  &mem_4671.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4756, 2, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4756, 3, sizeof(mem_4674.mem),
                                  &mem_4674.mem));
    if (1 * (num_groups_4761 * group_size_4760) != 0) {
        const size_t global_work_size_4866[1] = {num_groups_4761 *
                     group_size_4760};
        const size_t local_work_size_4870[1] = {group_size_4760};
        int64_t time_start_4867, time_end_4868;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4756");
            fprintf(stderr, "%zu", global_work_size_4866[0]);
            fprintf(stderr, "].\n");
            time_start_4867 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4756, 1,
                                              NULL, global_work_size_4866,
                                              local_work_size_4870, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4868 = get_wall_time();
            
            long time_diff_4869 = time_end_4868 - time_start_4867;
            
            if (detail_timing) {
                map_kernel_4756total_runtime += time_diff_4869;
                map_kernel_4756runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4756",
                        (int) time_diff_4869);
            }
        }
    }
    for (int i_3984 = 0; i_3984 < n_3945; i_3984++) {
        char y_3985 = slt32(i_3984, m_3943);
        char bounds_check_3986;
        
        if (!y_3985) {
            fprintf(stderr, "Assertion %s at %s failed.\n", "y_3985",
                    "lstm.fut:12:16-12:16");
            abort();
        }
        for (int j_3988 = 0; j_3988 < m_3943; j_3988++) {
            float res_3995;
            float res_3989 = 0.0F;
            
            for (int k_3990 = 0; k_3990 < n_3945; k_3990++) {
                float read_res_4871;
                
                OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                                   U_bi_mem_4638.mem, CL_TRUE,
                                                   (i_3984 * n_3945 + k_3990) *
                                                   4, sizeof(float),
                                                   &read_res_4871, 0, NULL,
                                                   NULL));
                
                float x_3991 = read_res_4871;
                float read_res_4872;
                
                OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                                   prev_output_mem_4662.mem,
                                                   CL_TRUE, (k_3990 * m_3943 +
                                                             j_3988) * 4,
                                                   sizeof(float),
                                                   &read_res_4872, 0, NULL,
                                                   NULL));
                
                float y_3992 = read_res_4872;
                float y_3993 = x_3991 * y_3992;
                float res_3994 = res_3989 + y_3993;
                float res_tmp_4764 = res_3994;
                
                res_3989 = res_tmp_4764;
            }
            res_3995 = res_3989;
            
            char y_3996 = slt32(j_3988, n_3945);
            char bounds_check_3997;
            
            if (!y_3996) {
                fprintf(stderr, "Assertion %s at %s failed.\n", "y_3996",
                        "lstm.fut:12:16-12:16");
                abort();
            }
            
            float write_tmp_4873 = res_3995;
            
            OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue, mem_4674.mem,
                                                CL_TRUE, (i_3984 * n_3945 +
                                                          j_3988) * 4,
                                                sizeof(float), &write_tmp_4873,
                                                0, NULL, NULL));
        }
    }
    
    char zip_cmp_4001 = n_3945 == m_3943;
    char zip_assert_4002;
    
    if (!zip_cmp_4001) {
        fprintf(stderr, "Assertion %s at %s failed.\n", "zip_cmp_4001",
                "lstm.fut:18:9-18:9");
        abort();
    }
    
    char cond_4004 = n_3945 == 0;
    int32_t size_4005;
    
    if (cond_4004) {
        size_4005 = 0;
    } else {
        size_4005 = m_3943;
    }
    
    char zip_cmp_4006 = m_3943 == n_3945;
    char zip_assert_4007;
    
    if (!zip_cmp_4006) {
        fprintf(stderr, "Assertion %s at %s failed.\n", "zip_cmp_4006",
                "lstm.fut:19:33-19:33");
        abort();
    }
    
    char eq_x_y_4008 = m_3943 == 0;
    char p_and_eq_x_y_4009 = cond_4004 && eq_x_y_4008;
    char not_p_4010 = !cond_4004;
    char assert_arg_4011 = p_and_eq_x_y_4009 || not_p_4010;
    char shape_cert_4012;
    
    if (!assert_arg_4011) {
        fprintf(stderr, "Assertion %s at %s failed.\n", "assert_arg_4011",
                "lstm.fut:18:17-18:17");
        abort();
    }
    
    int32_t nesting_size_4335 = m_3943 * n_3945;
    struct memblock_device mem_4677;
    
    mem_4677.references = NULL;
    memblock_alloc_device(&mem_4677, bytes_4667);
    
    int32_t group_size_4765;
    int32_t num_groups_4766;
    
    group_size_4765 = cl_group_size;
    num_groups_4766 = squot32(n_3945 * m_3943 + group_size_4765 - 1,
                              group_size_4765);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4337, 0, sizeof(mem_4669.mem),
                                  &mem_4669.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4337, 1, sizeof(n_3945), &n_3945));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4337, 2, sizeof(mem_4674.mem),
                                  &mem_4674.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4337, 3, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4337, 4, sizeof(mem_4677.mem),
                                  &mem_4677.mem));
    if (1 * (num_groups_4766 * group_size_4765) != 0) {
        const size_t global_work_size_4874[1] = {num_groups_4766 *
                     group_size_4765};
        const size_t local_work_size_4878[1] = {group_size_4765};
        int64_t time_start_4875, time_end_4876;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4337");
            fprintf(stderr, "%zu", global_work_size_4874[0]);
            fprintf(stderr, "].\n");
            time_start_4875 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4337, 1,
                                              NULL, global_work_size_4874,
                                              local_work_size_4878, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4876 = get_wall_time();
            
            long time_diff_4877 = time_end_4876 - time_start_4875;
            
            if (detail_timing) {
                map_kernel_4337total_runtime += time_diff_4877;
                map_kernel_4337runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4337",
                        (int) time_diff_4877);
            }
        }
    }
    
    char eq_x_z_4026 = 0 == m_3943;
    char p_and_eq_x_y_4027 = not_p_4010 && eq_x_z_4026;
    char eq_x_y_4028 = cond_4004 || p_and_eq_x_y_4027;
    char p_and_eq_x_y_4029 = eq_x_y_4008 && eq_x_y_4028;
    char not_p_4030 = !eq_x_y_4008;
    char assert_arg_4031 = p_and_eq_x_y_4029 || not_p_4030;
    char shape_cert_4032;
    
    if (!assert_arg_4031) {
        fprintf(stderr, "Assertion %s at %s failed.\n", "assert_arg_4031",
                "lstm.fut:18:17-18:17");
        abort();
    }
    
    int32_t nesting_size_4358 = size_4005 * m_3943;
    int32_t bytes_4678 = bytes_4665 * size_4005;
    struct memblock_device mem_4680;
    
    mem_4680.references = NULL;
    memblock_alloc_device(&mem_4680, bytes_4678);
    
    int32_t group_size_4767;
    int32_t num_groups_4768;
    
    group_size_4767 = cl_group_size;
    num_groups_4768 = squot32(m_3943 * size_4005 + group_size_4767 - 1,
                              group_size_4767);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4360, 0, sizeof(b_bi_mem_4640.mem),
                                  &b_bi_mem_4640.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4360, 1, sizeof(mem_4677.mem),
                                  &mem_4677.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4360, 2, sizeof(size_4005),
                                  &size_4005));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4360, 3, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4360, 4, sizeof(mem_4680.mem),
                                  &mem_4680.mem));
    if (1 * (num_groups_4768 * group_size_4767) != 0) {
        const size_t global_work_size_4879[1] = {num_groups_4768 *
                     group_size_4767};
        const size_t local_work_size_4883[1] = {group_size_4767};
        int64_t time_start_4880, time_end_4881;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4360");
            fprintf(stderr, "%zu", global_work_size_4879[0]);
            fprintf(stderr, "].\n");
            time_start_4880 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4360, 1,
                                              NULL, global_work_size_4879,
                                              local_work_size_4883, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4881 = get_wall_time();
            
            long time_diff_4882 = time_end_4881 - time_start_4880;
            
            if (detail_timing) {
                map_kernel_4360total_runtime += time_diff_4882;
                map_kernel_4360runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4360",
                        (int) time_diff_4882);
            }
        }
    }
    
    char p_and_eq_x_y_4041 = not_p_4030 && eq_x_y_4028;
    char eq_x_y_4042 = eq_x_y_4008 || p_and_eq_x_y_4041;
    char p_and_eq_x_y_4043 = eq_x_y_4008 && eq_x_y_4008;
    char p_and_eq_x_y_4044 = not_p_4030 && assert_arg_4011;
    char eq_x_z_4045 = p_and_eq_x_y_4043 || p_and_eq_x_y_4044;
    char p_and_eq_x_y_4046 = cond_4004 && eq_x_y_4042;
    char p_and_eq_x_y_4047 = not_p_4010 && eq_x_z_4045;
    char assert_arg_4048 = p_and_eq_x_y_4046 || p_and_eq_x_y_4047;
    char shape_cert_4049;
    
    if (!assert_arg_4048) {
        fprintf(stderr, "Assertion %s at %s failed.\n", "assert_arg_4048",
                "lstm.fut:31:17-31:17");
        abort();
    }
    
    struct memblock_device mem_4683;
    
    mem_4683.references = NULL;
    memblock_alloc_device(&mem_4683, bytes_4667);
    
    int32_t group_size_4769;
    int32_t num_groups_4770;
    
    group_size_4769 = cl_group_size;
    num_groups_4770 = squot32(n_3945 * m_3943 + group_size_4769 - 1,
                              group_size_4769);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4377, 0, sizeof(mem_4680.mem),
                                  &mem_4680.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4377, 1, sizeof(n_3945), &n_3945));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4377, 2, sizeof(size_4005),
                                  &size_4005));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4377, 3, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4377, 4, sizeof(mem_4683.mem),
                                  &mem_4683.mem));
    if (1 * (num_groups_4770 * group_size_4769) != 0) {
        const size_t global_work_size_4884[1] = {num_groups_4770 *
                     group_size_4769};
        const size_t local_work_size_4888[1] = {group_size_4769};
        int64_t time_start_4885, time_end_4886;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4377");
            fprintf(stderr, "%zu", global_work_size_4884[0]);
            fprintf(stderr, "].\n");
            time_start_4885 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4377, 1,
                                              NULL, global_work_size_4884,
                                              local_work_size_4888, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4886 = get_wall_time();
            
            long time_diff_4887 = time_end_4886 - time_start_4885;
            
            if (detail_timing) {
                map_kernel_4377total_runtime += time_diff_4887;
                map_kernel_4377runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4377",
                        (int) time_diff_4887);
            }
        }
    }
    
    struct memblock_device mem_4686;
    
    mem_4686.references = NULL;
    memblock_alloc_device(&mem_4686, bytes_4667);
    
    int32_t group_size_4775;
    int32_t num_groups_4776;
    
    group_size_4775 = cl_group_size;
    num_groups_4776 = squot32(n_3945 * m_3943 + group_size_4775 - 1,
                              group_size_4775);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4771, 0, sizeof(n_3945), &n_3945));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4771, 1, sizeof(mem_4666.mem),
                                  &mem_4666.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4771, 2, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4771, 3, sizeof(mem_4686.mem),
                                  &mem_4686.mem));
    if (1 * (num_groups_4776 * group_size_4775) != 0) {
        const size_t global_work_size_4889[1] = {num_groups_4776 *
                     group_size_4775};
        const size_t local_work_size_4893[1] = {group_size_4775};
        int64_t time_start_4890, time_end_4891;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4771");
            fprintf(stderr, "%zu", global_work_size_4889[0]);
            fprintf(stderr, "].\n");
            time_start_4890 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4771, 1,
                                              NULL, global_work_size_4889,
                                              local_work_size_4893, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4891 = get_wall_time();
            
            long time_diff_4892 = time_end_4891 - time_start_4890;
            
            if (detail_timing) {
                map_kernel_4771total_runtime += time_diff_4892;
                map_kernel_4771runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4771",
                        (int) time_diff_4892);
            }
        }
    }
    for (int i_4062 = 0; i_4062 < m_3943; i_4062++) {
        char y_4063 = slt32(i_4062, n_3945);
        char bounds_check_4064;
        
        if (!y_4063) {
            fprintf(stderr, "Assertion %s at %s failed.\n", "y_4063",
                    "lstm.fut:12:16-12:16");
            abort();
        }
        for (int j_4066 = 0; j_4066 < n_3945; j_4066++) {
            float res_4073;
            float res_4067 = 0.0F;
            
            for (int k_4068 = 0; k_4068 < o_3944; k_4068++) {
                float read_res_4894;
                
                OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                                   W_ig_mem_4642.mem, CL_TRUE,
                                                   (i_4062 * o_3944 + k_4068) *
                                                   4, sizeof(float),
                                                   &read_res_4894, 0, NULL,
                                                   NULL));
                
                float x_4069 = read_res_4894;
                float read_res_4895;
                
                OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                                   input_mem_4660.mem, CL_TRUE,
                                                   (k_4068 * n_3945 + j_4066) *
                                                   4, sizeof(float),
                                                   &read_res_4895, 0, NULL,
                                                   NULL));
                
                float y_4070 = read_res_4895;
                float y_4071 = x_4069 * y_4070;
                float res_4072 = res_4067 + y_4071;
                float res_tmp_4779 = res_4072;
                
                res_4067 = res_tmp_4779;
            }
            res_4073 = res_4067;
            
            char y_4074 = slt32(j_4066, m_3943);
            char bounds_check_4075;
            
            if (!y_4074) {
                fprintf(stderr, "Assertion %s at %s failed.\n", "y_4074",
                        "lstm.fut:12:16-12:16");
                abort();
            }
            
            float write_tmp_4896 = res_4073;
            
            OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue, mem_4686.mem,
                                                CL_TRUE, (i_4062 * m_3943 +
                                                          j_4066) * 4,
                                                sizeof(float), &write_tmp_4896,
                                                0, NULL, NULL));
        }
    }
    
    struct memblock_device mem_4689;
    
    mem_4689.references = NULL;
    memblock_alloc_device(&mem_4689, bytes_4672);
    
    int32_t group_size_4784;
    int32_t num_groups_4785;
    
    group_size_4784 = cl_group_size;
    num_groups_4785 = squot32(m_3943 * n_3945 + group_size_4784 - 1,
                              group_size_4784);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4780, 0, sizeof(n_3945), &n_3945));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4780, 1, sizeof(mem_4671.mem),
                                  &mem_4671.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4780, 2, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4780, 3, sizeof(mem_4689.mem),
                                  &mem_4689.mem));
    if (1 * (num_groups_4785 * group_size_4784) != 0) {
        const size_t global_work_size_4897[1] = {num_groups_4785 *
                     group_size_4784};
        const size_t local_work_size_4901[1] = {group_size_4784};
        int64_t time_start_4898, time_end_4899;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4780");
            fprintf(stderr, "%zu", global_work_size_4897[0]);
            fprintf(stderr, "].\n");
            time_start_4898 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4780, 1,
                                              NULL, global_work_size_4897,
                                              local_work_size_4901, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4899 = get_wall_time();
            
            long time_diff_4900 = time_end_4899 - time_start_4898;
            
            if (detail_timing) {
                map_kernel_4780total_runtime += time_diff_4900;
                map_kernel_4780runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4780",
                        (int) time_diff_4900);
            }
        }
    }
    for (int i_4081 = 0; i_4081 < n_3945; i_4081++) {
        char y_4082 = slt32(i_4081, m_3943);
        char bounds_check_4083;
        
        if (!y_4082) {
            fprintf(stderr, "Assertion %s at %s failed.\n", "y_4082",
                    "lstm.fut:12:16-12:16");
            abort();
        }
        for (int j_4085 = 0; j_4085 < m_3943; j_4085++) {
            float res_4092;
            float res_4086 = 0.0F;
            
            for (int k_4087 = 0; k_4087 < n_3945; k_4087++) {
                float read_res_4902;
                
                OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                                   U_ig_mem_4644.mem, CL_TRUE,
                                                   (i_4081 * n_3945 + k_4087) *
                                                   4, sizeof(float),
                                                   &read_res_4902, 0, NULL,
                                                   NULL));
                
                float x_4088 = read_res_4902;
                float read_res_4903;
                
                OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                                   prev_output_mem_4662.mem,
                                                   CL_TRUE, (k_4087 * m_3943 +
                                                             j_4085) * 4,
                                                   sizeof(float),
                                                   &read_res_4903, 0, NULL,
                                                   NULL));
                
                float y_4089 = read_res_4903;
                float y_4090 = x_4088 * y_4089;
                float res_4091 = res_4086 + y_4090;
                float res_tmp_4788 = res_4091;
                
                res_4086 = res_tmp_4788;
            }
            res_4092 = res_4086;
            
            char y_4093 = slt32(j_4085, n_3945);
            char bounds_check_4094;
            
            if (!y_4093) {
                fprintf(stderr, "Assertion %s at %s failed.\n", "y_4093",
                        "lstm.fut:12:16-12:16");
                abort();
            }
            
            float write_tmp_4904 = res_4092;
            
            OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue, mem_4689.mem,
                                                CL_TRUE, (i_4081 * n_3945 +
                                                          j_4085) * 4,
                                                sizeof(float), &write_tmp_4904,
                                                0, NULL, NULL));
        }
    }
    
    struct memblock_device mem_4692;
    
    mem_4692.references = NULL;
    memblock_alloc_device(&mem_4692, bytes_4667);
    
    int32_t group_size_4789;
    int32_t num_groups_4790;
    
    group_size_4789 = cl_group_size;
    num_groups_4790 = squot32(n_3945 * m_3943 + group_size_4789 - 1,
                              group_size_4789);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4397, 0, sizeof(n_3945), &n_3945));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4397, 1, sizeof(mem_4689.mem),
                                  &mem_4689.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4397, 2, sizeof(mem_4686.mem),
                                  &mem_4686.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4397, 3, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4397, 4, sizeof(mem_4692.mem),
                                  &mem_4692.mem));
    if (1 * (num_groups_4790 * group_size_4789) != 0) {
        const size_t global_work_size_4905[1] = {num_groups_4790 *
                     group_size_4789};
        const size_t local_work_size_4909[1] = {group_size_4789};
        int64_t time_start_4906, time_end_4907;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4397");
            fprintf(stderr, "%zu", global_work_size_4905[0]);
            fprintf(stderr, "].\n");
            time_start_4906 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4397, 1,
                                              NULL, global_work_size_4905,
                                              local_work_size_4909, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4907 = get_wall_time();
            
            long time_diff_4908 = time_end_4907 - time_start_4906;
            
            if (detail_timing) {
                map_kernel_4397total_runtime += time_diff_4908;
                map_kernel_4397runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4397",
                        (int) time_diff_4908);
            }
        }
    }
    
    struct memblock_device mem_4695;
    
    mem_4695.references = NULL;
    memblock_alloc_device(&mem_4695, bytes_4678);
    
    int32_t group_size_4791;
    int32_t num_groups_4792;
    
    group_size_4791 = cl_group_size;
    num_groups_4792 = squot32(m_3943 * size_4005 + group_size_4791 - 1,
                              group_size_4791);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4420, 0, sizeof(mem_4692.mem),
                                  &mem_4692.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4420, 1, sizeof(size_4005),
                                  &size_4005));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4420, 2, sizeof(b_ig_mem_4646.mem),
                                  &b_ig_mem_4646.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4420, 3, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4420, 4, sizeof(mem_4695.mem),
                                  &mem_4695.mem));
    if (1 * (num_groups_4792 * group_size_4791) != 0) {
        const size_t global_work_size_4910[1] = {num_groups_4792 *
                     group_size_4791};
        const size_t local_work_size_4914[1] = {group_size_4791};
        int64_t time_start_4911, time_end_4912;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4420");
            fprintf(stderr, "%zu", global_work_size_4910[0]);
            fprintf(stderr, "].\n");
            time_start_4911 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4420, 1,
                                              NULL, global_work_size_4910,
                                              local_work_size_4914, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4912 = get_wall_time();
            
            long time_diff_4913 = time_end_4912 - time_start_4911;
            
            if (detail_timing) {
                map_kernel_4420total_runtime += time_diff_4913;
                map_kernel_4420runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4420",
                        (int) time_diff_4913);
            }
        }
    }
    
    struct memblock_device mem_4698;
    
    mem_4698.references = NULL;
    memblock_alloc_device(&mem_4698, bytes_4667);
    
    int32_t group_size_4793;
    int32_t num_groups_4794;
    
    group_size_4793 = cl_group_size;
    num_groups_4794 = squot32(n_3945 * m_3943 + group_size_4793 - 1,
                              group_size_4793);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4437, 0, sizeof(n_3945), &n_3945));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4437, 1, sizeof(size_4005),
                                  &size_4005));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4437, 2, sizeof(mem_4695.mem),
                                  &mem_4695.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4437, 3, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4437, 4, sizeof(mem_4698.mem),
                                  &mem_4698.mem));
    if (1 * (num_groups_4794 * group_size_4793) != 0) {
        const size_t global_work_size_4915[1] = {num_groups_4794 *
                     group_size_4793};
        const size_t local_work_size_4919[1] = {group_size_4793};
        int64_t time_start_4916, time_end_4917;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4437");
            fprintf(stderr, "%zu", global_work_size_4915[0]);
            fprintf(stderr, "].\n");
            time_start_4916 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4437, 1,
                                              NULL, global_work_size_4915,
                                              local_work_size_4919, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4917 = get_wall_time();
            
            long time_diff_4918 = time_end_4917 - time_start_4916;
            
            if (detail_timing) {
                map_kernel_4437total_runtime += time_diff_4918;
                map_kernel_4437runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4437",
                        (int) time_diff_4918);
            }
        }
    }
    
    struct memblock_device mem_4701;
    
    mem_4701.references = NULL;
    memblock_alloc_device(&mem_4701, bytes_4667);
    
    int32_t group_size_4799;
    int32_t num_groups_4800;
    
    group_size_4799 = cl_group_size;
    num_groups_4800 = squot32(n_3945 * m_3943 + group_size_4799 - 1,
                              group_size_4799);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4795, 0, sizeof(n_3945), &n_3945));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4795, 1, sizeof(mem_4666.mem),
                                  &mem_4666.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4795, 2, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4795, 3, sizeof(mem_4701.mem),
                                  &mem_4701.mem));
    if (1 * (num_groups_4800 * group_size_4799) != 0) {
        const size_t global_work_size_4920[1] = {num_groups_4800 *
                     group_size_4799};
        const size_t local_work_size_4924[1] = {group_size_4799};
        int64_t time_start_4921, time_end_4922;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4795");
            fprintf(stderr, "%zu", global_work_size_4920[0]);
            fprintf(stderr, "].\n");
            time_start_4921 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4795, 1,
                                              NULL, global_work_size_4920,
                                              local_work_size_4924, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4922 = get_wall_time();
            
            long time_diff_4923 = time_end_4922 - time_start_4921;
            
            if (detail_timing) {
                map_kernel_4795total_runtime += time_diff_4923;
                map_kernel_4795runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4795",
                        (int) time_diff_4923);
            }
        }
    }
    for (int i_4131 = 0; i_4131 < m_3943; i_4131++) {
        char y_4132 = slt32(i_4131, n_3945);
        char bounds_check_4133;
        
        if (!y_4132) {
            fprintf(stderr, "Assertion %s at %s failed.\n", "y_4132",
                    "lstm.fut:12:16-12:16");
            abort();
        }
        for (int j_4135 = 0; j_4135 < n_3945; j_4135++) {
            float res_4142;
            float res_4136 = 0.0F;
            
            for (int k_4137 = 0; k_4137 < o_3944; k_4137++) {
                float read_res_4925;
                
                OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                                   W_fg_mem_4648.mem, CL_TRUE,
                                                   (i_4131 * o_3944 + k_4137) *
                                                   4, sizeof(float),
                                                   &read_res_4925, 0, NULL,
                                                   NULL));
                
                float x_4138 = read_res_4925;
                float read_res_4926;
                
                OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                                   input_mem_4660.mem, CL_TRUE,
                                                   (k_4137 * n_3945 + j_4135) *
                                                   4, sizeof(float),
                                                   &read_res_4926, 0, NULL,
                                                   NULL));
                
                float y_4139 = read_res_4926;
                float y_4140 = x_4138 * y_4139;
                float res_4141 = res_4136 + y_4140;
                float res_tmp_4803 = res_4141;
                
                res_4136 = res_tmp_4803;
            }
            res_4142 = res_4136;
            
            char y_4143 = slt32(j_4135, m_3943);
            char bounds_check_4144;
            
            if (!y_4143) {
                fprintf(stderr, "Assertion %s at %s failed.\n", "y_4143",
                        "lstm.fut:12:16-12:16");
                abort();
            }
            
            float write_tmp_4927 = res_4142;
            
            OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue, mem_4701.mem,
                                                CL_TRUE, (i_4131 * m_3943 +
                                                          j_4135) * 4,
                                                sizeof(float), &write_tmp_4927,
                                                0, NULL, NULL));
        }
    }
    
    struct memblock_device mem_4704;
    
    mem_4704.references = NULL;
    memblock_alloc_device(&mem_4704, bytes_4672);
    
    int32_t group_size_4808;
    int32_t num_groups_4809;
    
    group_size_4808 = cl_group_size;
    num_groups_4809 = squot32(m_3943 * n_3945 + group_size_4808 - 1,
                              group_size_4808);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4804, 0, sizeof(n_3945), &n_3945));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4804, 1, sizeof(mem_4671.mem),
                                  &mem_4671.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4804, 2, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4804, 3, sizeof(mem_4704.mem),
                                  &mem_4704.mem));
    if (1 * (num_groups_4809 * group_size_4808) != 0) {
        const size_t global_work_size_4928[1] = {num_groups_4809 *
                     group_size_4808};
        const size_t local_work_size_4932[1] = {group_size_4808};
        int64_t time_start_4929, time_end_4930;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4804");
            fprintf(stderr, "%zu", global_work_size_4928[0]);
            fprintf(stderr, "].\n");
            time_start_4929 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4804, 1,
                                              NULL, global_work_size_4928,
                                              local_work_size_4932, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4930 = get_wall_time();
            
            long time_diff_4931 = time_end_4930 - time_start_4929;
            
            if (detail_timing) {
                map_kernel_4804total_runtime += time_diff_4931;
                map_kernel_4804runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4804",
                        (int) time_diff_4931);
            }
        }
    }
    for (int i_4150 = 0; i_4150 < n_3945; i_4150++) {
        char y_4151 = slt32(i_4150, m_3943);
        char bounds_check_4152;
        
        if (!y_4151) {
            fprintf(stderr, "Assertion %s at %s failed.\n", "y_4151",
                    "lstm.fut:12:16-12:16");
            abort();
        }
        for (int j_4154 = 0; j_4154 < m_3943; j_4154++) {
            float res_4161;
            float res_4155 = 0.0F;
            
            for (int k_4156 = 0; k_4156 < n_3945; k_4156++) {
                float read_res_4933;
                
                OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                                   U_fg_mem_4650.mem, CL_TRUE,
                                                   (i_4150 * n_3945 + k_4156) *
                                                   4, sizeof(float),
                                                   &read_res_4933, 0, NULL,
                                                   NULL));
                
                float x_4157 = read_res_4933;
                float read_res_4934;
                
                OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                                   prev_output_mem_4662.mem,
                                                   CL_TRUE, (k_4156 * m_3943 +
                                                             j_4154) * 4,
                                                   sizeof(float),
                                                   &read_res_4934, 0, NULL,
                                                   NULL));
                
                float y_4158 = read_res_4934;
                float y_4159 = x_4157 * y_4158;
                float res_4160 = res_4155 + y_4159;
                float res_tmp_4812 = res_4160;
                
                res_4155 = res_tmp_4812;
            }
            res_4161 = res_4155;
            
            char y_4162 = slt32(j_4154, n_3945);
            char bounds_check_4163;
            
            if (!y_4162) {
                fprintf(stderr, "Assertion %s at %s failed.\n", "y_4162",
                        "lstm.fut:12:16-12:16");
                abort();
            }
            
            float write_tmp_4935 = res_4161;
            
            OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue, mem_4704.mem,
                                                CL_TRUE, (i_4150 * n_3945 +
                                                          j_4154) * 4,
                                                sizeof(float), &write_tmp_4935,
                                                0, NULL, NULL));
        }
    }
    
    struct memblock_device mem_4707;
    
    mem_4707.references = NULL;
    memblock_alloc_device(&mem_4707, bytes_4667);
    
    int32_t group_size_4813;
    int32_t num_groups_4814;
    
    group_size_4813 = cl_group_size;
    num_groups_4814 = squot32(n_3945 * m_3943 + group_size_4813 - 1,
                              group_size_4813);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4457, 0, sizeof(mem_4704.mem),
                                  &mem_4704.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4457, 1, sizeof(n_3945), &n_3945));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4457, 2, sizeof(mem_4701.mem),
                                  &mem_4701.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4457, 3, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4457, 4, sizeof(mem_4707.mem),
                                  &mem_4707.mem));
    if (1 * (num_groups_4814 * group_size_4813) != 0) {
        const size_t global_work_size_4936[1] = {num_groups_4814 *
                     group_size_4813};
        const size_t local_work_size_4940[1] = {group_size_4813};
        int64_t time_start_4937, time_end_4938;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4457");
            fprintf(stderr, "%zu", global_work_size_4936[0]);
            fprintf(stderr, "].\n");
            time_start_4937 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4457, 1,
                                              NULL, global_work_size_4936,
                                              local_work_size_4940, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4938 = get_wall_time();
            
            long time_diff_4939 = time_end_4938 - time_start_4937;
            
            if (detail_timing) {
                map_kernel_4457total_runtime += time_diff_4939;
                map_kernel_4457runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4457",
                        (int) time_diff_4939);
            }
        }
    }
    
    struct memblock_device mem_4710;
    
    mem_4710.references = NULL;
    memblock_alloc_device(&mem_4710, bytes_4678);
    
    int32_t group_size_4815;
    int32_t num_groups_4816;
    
    group_size_4815 = cl_group_size;
    num_groups_4816 = squot32(m_3943 * size_4005 + group_size_4815 - 1,
                              group_size_4815);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4480, 0, sizeof(b_fg_mem_4652.mem),
                                  &b_fg_mem_4652.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4480, 1, sizeof(size_4005),
                                  &size_4005));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4480, 2, sizeof(mem_4707.mem),
                                  &mem_4707.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4480, 3, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4480, 4, sizeof(mem_4710.mem),
                                  &mem_4710.mem));
    if (1 * (num_groups_4816 * group_size_4815) != 0) {
        const size_t global_work_size_4941[1] = {num_groups_4816 *
                     group_size_4815};
        const size_t local_work_size_4945[1] = {group_size_4815};
        int64_t time_start_4942, time_end_4943;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4480");
            fprintf(stderr, "%zu", global_work_size_4941[0]);
            fprintf(stderr, "].\n");
            time_start_4942 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4480, 1,
                                              NULL, global_work_size_4941,
                                              local_work_size_4945, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4943 = get_wall_time();
            
            long time_diff_4944 = time_end_4943 - time_start_4942;
            
            if (detail_timing) {
                map_kernel_4480total_runtime += time_diff_4944;
                map_kernel_4480runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4480",
                        (int) time_diff_4944);
            }
        }
    }
    
    struct memblock_device mem_4713;
    
    mem_4713.references = NULL;
    memblock_alloc_device(&mem_4713, bytes_4667);
    
    int32_t group_size_4817;
    int32_t num_groups_4818;
    
    group_size_4817 = cl_group_size;
    num_groups_4818 = squot32(n_3945 * m_3943 + group_size_4817 - 1,
                              group_size_4817);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4497, 0, sizeof(n_3945), &n_3945));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4497, 1, sizeof(size_4005),
                                  &size_4005));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4497, 2, sizeof(mem_4710.mem),
                                  &mem_4710.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4497, 3, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4497, 4, sizeof(mem_4713.mem),
                                  &mem_4713.mem));
    if (1 * (num_groups_4818 * group_size_4817) != 0) {
        const size_t global_work_size_4946[1] = {num_groups_4818 *
                     group_size_4817};
        const size_t local_work_size_4950[1] = {group_size_4817};
        int64_t time_start_4947, time_end_4948;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4497");
            fprintf(stderr, "%zu", global_work_size_4946[0]);
            fprintf(stderr, "].\n");
            time_start_4947 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4497, 1,
                                              NULL, global_work_size_4946,
                                              local_work_size_4950, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4948 = get_wall_time();
            
            long time_diff_4949 = time_end_4948 - time_start_4947;
            
            if (detail_timing) {
                map_kernel_4497total_runtime += time_diff_4949;
                map_kernel_4497runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4497",
                        (int) time_diff_4949);
            }
        }
    }
    
    struct memblock_device mem_4716;
    
    mem_4716.references = NULL;
    memblock_alloc_device(&mem_4716, bytes_4667);
    
    int32_t group_size_4823;
    int32_t num_groups_4824;
    
    group_size_4823 = cl_group_size;
    num_groups_4824 = squot32(n_3945 * m_3943 + group_size_4823 - 1,
                              group_size_4823);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4819, 0, sizeof(n_3945), &n_3945));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4819, 1, sizeof(mem_4666.mem),
                                  &mem_4666.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4819, 2, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4819, 3, sizeof(mem_4716.mem),
                                  &mem_4716.mem));
    if (1 * (num_groups_4824 * group_size_4823) != 0) {
        const size_t global_work_size_4951[1] = {num_groups_4824 *
                     group_size_4823};
        const size_t local_work_size_4955[1] = {group_size_4823};
        int64_t time_start_4952, time_end_4953;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4819");
            fprintf(stderr, "%zu", global_work_size_4951[0]);
            fprintf(stderr, "].\n");
            time_start_4952 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4819, 1,
                                              NULL, global_work_size_4951,
                                              local_work_size_4955, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4953 = get_wall_time();
            
            long time_diff_4954 = time_end_4953 - time_start_4952;
            
            if (detail_timing) {
                map_kernel_4819total_runtime += time_diff_4954;
                map_kernel_4819runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4819",
                        (int) time_diff_4954);
            }
        }
    }
    for (int i_4200 = 0; i_4200 < m_3943; i_4200++) {
        char y_4201 = slt32(i_4200, n_3945);
        char bounds_check_4202;
        
        if (!y_4201) {
            fprintf(stderr, "Assertion %s at %s failed.\n", "y_4201",
                    "lstm.fut:12:16-12:16");
            abort();
        }
        for (int j_4204 = 0; j_4204 < n_3945; j_4204++) {
            float res_4211;
            float res_4205 = 0.0F;
            
            for (int k_4206 = 0; k_4206 < o_3944; k_4206++) {
                float read_res_4956;
                
                OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                                   W_og_mem_4654.mem, CL_TRUE,
                                                   (i_4200 * o_3944 + k_4206) *
                                                   4, sizeof(float),
                                                   &read_res_4956, 0, NULL,
                                                   NULL));
                
                float x_4207 = read_res_4956;
                float read_res_4957;
                
                OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                                   input_mem_4660.mem, CL_TRUE,
                                                   (k_4206 * n_3945 + j_4204) *
                                                   4, sizeof(float),
                                                   &read_res_4957, 0, NULL,
                                                   NULL));
                
                float y_4208 = read_res_4957;
                float y_4209 = x_4207 * y_4208;
                float res_4210 = res_4205 + y_4209;
                float res_tmp_4827 = res_4210;
                
                res_4205 = res_tmp_4827;
            }
            res_4211 = res_4205;
            
            char y_4212 = slt32(j_4204, m_3943);
            char bounds_check_4213;
            
            if (!y_4212) {
                fprintf(stderr, "Assertion %s at %s failed.\n", "y_4212",
                        "lstm.fut:12:16-12:16");
                abort();
            }
            
            float write_tmp_4958 = res_4211;
            
            OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue, mem_4716.mem,
                                                CL_TRUE, (i_4200 * m_3943 +
                                                          j_4204) * 4,
                                                sizeof(float), &write_tmp_4958,
                                                0, NULL, NULL));
        }
    }
    
    struct memblock_device mem_4719;
    
    mem_4719.references = NULL;
    memblock_alloc_device(&mem_4719, bytes_4672);
    
    int32_t group_size_4832;
    int32_t num_groups_4833;
    
    group_size_4832 = cl_group_size;
    num_groups_4833 = squot32(m_3943 * n_3945 + group_size_4832 - 1,
                              group_size_4832);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4828, 0, sizeof(n_3945), &n_3945));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4828, 1, sizeof(mem_4671.mem),
                                  &mem_4671.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4828, 2, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4828, 3, sizeof(mem_4719.mem),
                                  &mem_4719.mem));
    if (1 * (num_groups_4833 * group_size_4832) != 0) {
        const size_t global_work_size_4959[1] = {num_groups_4833 *
                     group_size_4832};
        const size_t local_work_size_4963[1] = {group_size_4832};
        int64_t time_start_4960, time_end_4961;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4828");
            fprintf(stderr, "%zu", global_work_size_4959[0]);
            fprintf(stderr, "].\n");
            time_start_4960 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4828, 1,
                                              NULL, global_work_size_4959,
                                              local_work_size_4963, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4961 = get_wall_time();
            
            long time_diff_4962 = time_end_4961 - time_start_4960;
            
            if (detail_timing) {
                map_kernel_4828total_runtime += time_diff_4962;
                map_kernel_4828runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4828",
                        (int) time_diff_4962);
            }
        }
    }
    for (int i_4219 = 0; i_4219 < n_3945; i_4219++) {
        char y_4220 = slt32(i_4219, m_3943);
        char bounds_check_4221;
        
        if (!y_4220) {
            fprintf(stderr, "Assertion %s at %s failed.\n", "y_4220",
                    "lstm.fut:12:16-12:16");
            abort();
        }
        for (int j_4223 = 0; j_4223 < m_3943; j_4223++) {
            float res_4230;
            float res_4224 = 0.0F;
            
            for (int k_4225 = 0; k_4225 < n_3945; k_4225++) {
                float read_res_4964;
                
                OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                                   U_og_mem_4656.mem, CL_TRUE,
                                                   (i_4219 * n_3945 + k_4225) *
                                                   4, sizeof(float),
                                                   &read_res_4964, 0, NULL,
                                                   NULL));
                
                float x_4226 = read_res_4964;
                float read_res_4965;
                
                OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                                   prev_output_mem_4662.mem,
                                                   CL_TRUE, (k_4225 * m_3943 +
                                                             j_4223) * 4,
                                                   sizeof(float),
                                                   &read_res_4965, 0, NULL,
                                                   NULL));
                
                float y_4227 = read_res_4965;
                float y_4228 = x_4226 * y_4227;
                float res_4229 = res_4224 + y_4228;
                float res_tmp_4836 = res_4229;
                
                res_4224 = res_tmp_4836;
            }
            res_4230 = res_4224;
            
            char y_4231 = slt32(j_4223, n_3945);
            char bounds_check_4232;
            
            if (!y_4231) {
                fprintf(stderr, "Assertion %s at %s failed.\n", "y_4231",
                        "lstm.fut:12:16-12:16");
                abort();
            }
            
            float write_tmp_4966 = res_4230;
            
            OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue, mem_4719.mem,
                                                CL_TRUE, (i_4219 * n_3945 +
                                                          j_4223) * 4,
                                                sizeof(float), &write_tmp_4966,
                                                0, NULL, NULL));
        }
    }
    
    struct memblock_device mem_4722;
    
    mem_4722.references = NULL;
    memblock_alloc_device(&mem_4722, bytes_4667);
    
    int32_t group_size_4837;
    int32_t num_groups_4838;
    
    group_size_4837 = cl_group_size;
    num_groups_4838 = squot32(n_3945 * m_3943 + group_size_4837 - 1,
                              group_size_4837);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4517, 0, sizeof(mem_4716.mem),
                                  &mem_4716.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4517, 1, sizeof(n_3945), &n_3945));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4517, 2, sizeof(mem_4719.mem),
                                  &mem_4719.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4517, 3, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4517, 4, sizeof(mem_4722.mem),
                                  &mem_4722.mem));
    if (1 * (num_groups_4838 * group_size_4837) != 0) {
        const size_t global_work_size_4967[1] = {num_groups_4838 *
                     group_size_4837};
        const size_t local_work_size_4971[1] = {group_size_4837};
        int64_t time_start_4968, time_end_4969;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4517");
            fprintf(stderr, "%zu", global_work_size_4967[0]);
            fprintf(stderr, "].\n");
            time_start_4968 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4517, 1,
                                              NULL, global_work_size_4967,
                                              local_work_size_4971, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4969 = get_wall_time();
            
            long time_diff_4970 = time_end_4969 - time_start_4968;
            
            if (detail_timing) {
                map_kernel_4517total_runtime += time_diff_4970;
                map_kernel_4517runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4517",
                        (int) time_diff_4970);
            }
        }
    }
    
    struct memblock_device mem_4725;
    
    mem_4725.references = NULL;
    memblock_alloc_device(&mem_4725, bytes_4678);
    
    int32_t group_size_4839;
    int32_t num_groups_4840;
    
    group_size_4839 = cl_group_size;
    num_groups_4840 = squot32(m_3943 * size_4005 + group_size_4839 - 1,
                              group_size_4839);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4540, 0, sizeof(size_4005),
                                  &size_4005));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4540, 1, sizeof(mem_4722.mem),
                                  &mem_4722.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4540, 2, sizeof(b_og_mem_4658.mem),
                                  &b_og_mem_4658.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4540, 3, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4540, 4, sizeof(mem_4725.mem),
                                  &mem_4725.mem));
    if (1 * (num_groups_4840 * group_size_4839) != 0) {
        const size_t global_work_size_4972[1] = {num_groups_4840 *
                     group_size_4839};
        const size_t local_work_size_4976[1] = {group_size_4839};
        int64_t time_start_4973, time_end_4974;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4540");
            fprintf(stderr, "%zu", global_work_size_4972[0]);
            fprintf(stderr, "].\n");
            time_start_4973 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4540, 1,
                                              NULL, global_work_size_4972,
                                              local_work_size_4976, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4974 = get_wall_time();
            
            long time_diff_4975 = time_end_4974 - time_start_4973;
            
            if (detail_timing) {
                map_kernel_4540total_runtime += time_diff_4975;
                map_kernel_4540runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4540",
                        (int) time_diff_4975);
            }
        }
    }
    
    struct memblock_device mem_4728;
    
    mem_4728.references = NULL;
    memblock_alloc_device(&mem_4728, bytes_4667);
    
    int32_t group_size_4841;
    int32_t num_groups_4842;
    
    group_size_4841 = cl_group_size;
    num_groups_4842 = squot32(n_3945 * m_3943 + group_size_4841 - 1,
                              group_size_4841);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4557, 0, sizeof(n_3945), &n_3945));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4557, 1, sizeof(size_4005),
                                  &size_4005));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4557, 2, sizeof(mem_4725.mem),
                                  &mem_4725.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4557, 3, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4557, 4, sizeof(mem_4728.mem),
                                  &mem_4728.mem));
    if (1 * (num_groups_4842 * group_size_4841) != 0) {
        const size_t global_work_size_4977[1] = {num_groups_4842 *
                     group_size_4841};
        const size_t local_work_size_4981[1] = {group_size_4841};
        int64_t time_start_4978, time_end_4979;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4557");
            fprintf(stderr, "%zu", global_work_size_4977[0]);
            fprintf(stderr, "].\n");
            time_start_4978 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4557, 1,
                                              NULL, global_work_size_4977,
                                              local_work_size_4981, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4979 = get_wall_time();
            
            long time_diff_4980 = time_end_4979 - time_start_4978;
            
            if (detail_timing) {
                map_kernel_4557total_runtime += time_diff_4980;
                map_kernel_4557runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4557",
                        (int) time_diff_4980);
            }
        }
    }
    
    int32_t nesting_size_4599 = size_4005 * n_3945;
    int32_t bytes_4729 = x_4668 * size_4005;
    struct memblock_device mem_4731;
    
    mem_4731.references = NULL;
    memblock_alloc_device(&mem_4731, bytes_4729);
    
    int32_t group_size_4843;
    int32_t num_groups_4844;
    
    group_size_4843 = cl_group_size;
    num_groups_4844 = squot32(n_3945 * size_4005 + group_size_4843 - 1,
                              group_size_4843);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4601, 0,
                                  sizeof(prev_cell_mem_4664.mem),
                                  &prev_cell_mem_4664.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4601, 1, sizeof(mem_4713.mem),
                                  &mem_4713.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4601, 2, sizeof(n_3945), &n_3945));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4601, 3, sizeof(size_4005),
                                  &size_4005));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4601, 4, sizeof(mem_4698.mem),
                                  &mem_4698.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4601, 5, sizeof(mem_4683.mem),
                                  &mem_4683.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4601, 6, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4601, 7, sizeof(mem_4731.mem),
                                  &mem_4731.mem));
    if (1 * (num_groups_4844 * group_size_4843) != 0) {
        const size_t global_work_size_4982[1] = {num_groups_4844 *
                     group_size_4843};
        const size_t local_work_size_4986[1] = {group_size_4843};
        int64_t time_start_4983, time_end_4984;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4601");
            fprintf(stderr, "%zu", global_work_size_4982[0]);
            fprintf(stderr, "].\n");
            time_start_4983 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4601, 1,
                                              NULL, global_work_size_4982,
                                              local_work_size_4986, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4984 = get_wall_time();
            
            long time_diff_4985 = time_end_4984 - time_start_4983;
            
            if (detail_timing) {
                map_kernel_4601total_runtime += time_diff_4985;
                map_kernel_4601runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4601",
                        (int) time_diff_4985);
            }
        }
    }
    
    struct memblock_device mem_4734;
    
    mem_4734.references = NULL;
    memblock_alloc_device(&mem_4734, bytes_4667);
    
    int32_t group_size_4845;
    int32_t num_groups_4846;
    
    group_size_4845 = cl_group_size;
    num_groups_4846 = squot32(n_3945 * m_3943 + group_size_4845 - 1,
                              group_size_4845);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4577, 0, sizeof(mem_4728.mem),
                                  &mem_4728.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4577, 1, sizeof(n_3945), &n_3945));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4577, 2, sizeof(size_4005),
                                  &size_4005));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4577, 3, sizeof(mem_4731.mem),
                                  &mem_4731.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4577, 4, sizeof(m_3943), &m_3943));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_4577, 5, sizeof(mem_4734.mem),
                                  &mem_4734.mem));
    if (1 * (num_groups_4846 * group_size_4845) != 0) {
        const size_t global_work_size_4987[1] = {num_groups_4846 *
                     group_size_4845};
        const size_t local_work_size_4991[1] = {group_size_4845};
        int64_t time_start_4988, time_end_4989;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_4577");
            fprintf(stderr, "%zu", global_work_size_4987[0]);
            fprintf(stderr, "].\n");
            time_start_4988 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_4577, 1,
                                              NULL, global_work_size_4987,
                                              local_work_size_4991, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_4989 = get_wall_time();
            
            long time_diff_4990 = time_end_4989 - time_start_4988;
            
            if (detail_timing) {
                map_kernel_4577total_runtime += time_diff_4990;
                map_kernel_4577runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_4577",
                        (int) time_diff_4990);
            }
        }
    }
    memblock_set_device(&out_mem_4735, &mem_4734);
    out_memsize_4736 = bytes_4667;
    memblock_set_device(&out_mem_4737, &mem_4731);
    out_memsize_4738 = bytes_4729;
    
    struct tuple_int32_t_device_mem_int32_t_device_mem retval_4847;
    
    retval_4847.elem_0 = out_memsize_4736;
    retval_4847.elem_1.references = NULL;
    memblock_set_device(&retval_4847.elem_1, &out_mem_4735);
    retval_4847.elem_2 = out_memsize_4738;
    retval_4847.elem_3.references = NULL;
    memblock_set_device(&retval_4847.elem_3, &out_mem_4737);
    memblock_unref_device(&out_mem_4735);
    memblock_unref_device(&out_mem_4737);
    memblock_unref_device(&mem_4666);
    memblock_unref_device(&mem_4669);
    memblock_unref_device(&mem_4671);
    memblock_unref_device(&mem_4674);
    memblock_unref_device(&mem_4677);
    memblock_unref_device(&mem_4680);
    memblock_unref_device(&mem_4683);
    memblock_unref_device(&mem_4686);
    memblock_unref_device(&mem_4689);
    memblock_unref_device(&mem_4692);
    memblock_unref_device(&mem_4695);
    memblock_unref_device(&mem_4698);
    memblock_unref_device(&mem_4701);
    memblock_unref_device(&mem_4704);
    memblock_unref_device(&mem_4707);
    memblock_unref_device(&mem_4710);
    memblock_unref_device(&mem_4713);
    memblock_unref_device(&mem_4716);
    memblock_unref_device(&mem_4719);
    memblock_unref_device(&mem_4722);
    memblock_unref_device(&mem_4725);
    memblock_unref_device(&mem_4728);
    memblock_unref_device(&mem_4731);
    memblock_unref_device(&mem_4734);
    return retval_4847;
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
    
    int32_t W_bi_mem_size_4635;
    int32_t U_bi_mem_size_4637;
    int32_t b_bi_mem_size_4639;
    int32_t W_ig_mem_size_4641;
    int32_t U_ig_mem_size_4643;
    int32_t b_ig_mem_size_4645;
    int32_t W_fg_mem_size_4647;
    int32_t U_fg_mem_size_4649;
    int32_t b_fg_mem_size_4651;
    int32_t W_og_mem_size_4653;
    int32_t U_og_mem_size_4655;
    int32_t b_og_mem_size_4657;
    int32_t input_mem_size_4659;
    int32_t prev_output_mem_size_4661;
    int32_t prev_cell_mem_size_4663;
    struct memblock W_bi_mem_4636;
    
    W_bi_mem_4636.references = NULL;
    memblock_alloc(&W_bi_mem_4636, 0);
    
    struct memblock U_bi_mem_4638;
    
    U_bi_mem_4638.references = NULL;
    memblock_alloc(&U_bi_mem_4638, 0);
    
    struct memblock b_bi_mem_4640;
    
    b_bi_mem_4640.references = NULL;
    memblock_alloc(&b_bi_mem_4640, 0);
    
    struct memblock W_ig_mem_4642;
    
    W_ig_mem_4642.references = NULL;
    memblock_alloc(&W_ig_mem_4642, 0);
    
    struct memblock U_ig_mem_4644;
    
    U_ig_mem_4644.references = NULL;
    memblock_alloc(&U_ig_mem_4644, 0);
    
    struct memblock b_ig_mem_4646;
    
    b_ig_mem_4646.references = NULL;
    memblock_alloc(&b_ig_mem_4646, 0);
    
    struct memblock W_fg_mem_4648;
    
    W_fg_mem_4648.references = NULL;
    memblock_alloc(&W_fg_mem_4648, 0);
    
    struct memblock U_fg_mem_4650;
    
    U_fg_mem_4650.references = NULL;
    memblock_alloc(&U_fg_mem_4650, 0);
    
    struct memblock b_fg_mem_4652;
    
    b_fg_mem_4652.references = NULL;
    memblock_alloc(&b_fg_mem_4652, 0);
    
    struct memblock W_og_mem_4654;
    
    W_og_mem_4654.references = NULL;
    memblock_alloc(&W_og_mem_4654, 0);
    
    struct memblock U_og_mem_4656;
    
    U_og_mem_4656.references = NULL;
    memblock_alloc(&U_og_mem_4656, 0);
    
    struct memblock b_og_mem_4658;
    
    b_og_mem_4658.references = NULL;
    memblock_alloc(&b_og_mem_4658, 0);
    
    struct memblock input_mem_4660;
    
    input_mem_4660.references = NULL;
    memblock_alloc(&input_mem_4660, 0);
    
    struct memblock prev_output_mem_4662;
    
    prev_output_mem_4662.references = NULL;
    memblock_alloc(&prev_output_mem_4662, 0);
    
    struct memblock prev_cell_mem_4664;
    
    prev_cell_mem_4664.references = NULL;
    memblock_alloc(&prev_cell_mem_4664, 0);
    
    int32_t m_3943;
    int32_t o_3944;
    int32_t n_3945;
    struct tuple_int32_t_device_mem_int32_t_device_mem main_ret_4992;
    
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float, (void **) &W_bi_mem_4636.mem,
                       shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        m_3943 = shape[0];
        o_3944 = shape[1];
        W_bi_mem_size_4635 = sizeof(float) * shape[0] * shape[1];
    }
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float, (void **) &U_bi_mem_4638.mem,
                       shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        n_3945 = shape[0];
        n_3945 = shape[1];
        U_bi_mem_size_4637 = sizeof(float) * shape[0] * shape[1];
    }
    {
        int64_t shape[1];
        
        if (read_array(sizeof(float), read_float, (void **) &b_bi_mem_4640.mem,
                       shape, 1) != 0)
            panic(1, "Syntax error when reading %s.\n", "[f32]");
        m_3943 = shape[0];
        b_bi_mem_size_4639 = sizeof(float) * shape[0];
    }
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float, (void **) &W_ig_mem_4642.mem,
                       shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        m_3943 = shape[0];
        o_3944 = shape[1];
        W_ig_mem_size_4641 = sizeof(float) * shape[0] * shape[1];
    }
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float, (void **) &U_ig_mem_4644.mem,
                       shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        n_3945 = shape[0];
        n_3945 = shape[1];
        U_ig_mem_size_4643 = sizeof(float) * shape[0] * shape[1];
    }
    {
        int64_t shape[1];
        
        if (read_array(sizeof(float), read_float, (void **) &b_ig_mem_4646.mem,
                       shape, 1) != 0)
            panic(1, "Syntax error when reading %s.\n", "[f32]");
        m_3943 = shape[0];
        b_ig_mem_size_4645 = sizeof(float) * shape[0];
    }
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float, (void **) &W_fg_mem_4648.mem,
                       shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        m_3943 = shape[0];
        o_3944 = shape[1];
        W_fg_mem_size_4647 = sizeof(float) * shape[0] * shape[1];
    }
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float, (void **) &U_fg_mem_4650.mem,
                       shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        n_3945 = shape[0];
        n_3945 = shape[1];
        U_fg_mem_size_4649 = sizeof(float) * shape[0] * shape[1];
    }
    {
        int64_t shape[1];
        
        if (read_array(sizeof(float), read_float, (void **) &b_fg_mem_4652.mem,
                       shape, 1) != 0)
            panic(1, "Syntax error when reading %s.\n", "[f32]");
        m_3943 = shape[0];
        b_fg_mem_size_4651 = sizeof(float) * shape[0];
    }
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float, (void **) &W_og_mem_4654.mem,
                       shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        m_3943 = shape[0];
        o_3944 = shape[1];
        W_og_mem_size_4653 = sizeof(float) * shape[0] * shape[1];
    }
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float, (void **) &U_og_mem_4656.mem,
                       shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        n_3945 = shape[0];
        n_3945 = shape[1];
        U_og_mem_size_4655 = sizeof(float) * shape[0] * shape[1];
    }
    {
        int64_t shape[1];
        
        if (read_array(sizeof(float), read_float, (void **) &b_og_mem_4658.mem,
                       shape, 1) != 0)
            panic(1, "Syntax error when reading %s.\n", "[f32]");
        m_3943 = shape[0];
        b_og_mem_size_4657 = sizeof(float) * shape[0];
    }
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float, (void **) &input_mem_4660.mem,
                       shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        o_3944 = shape[0];
        n_3945 = shape[1];
        input_mem_size_4659 = sizeof(float) * shape[0] * shape[1];
    }
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float,
                       (void **) &prev_output_mem_4662.mem, shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        n_3945 = shape[0];
        m_3943 = shape[1];
        prev_output_mem_size_4661 = sizeof(float) * shape[0] * shape[1];
    }
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float,
                       (void **) &prev_cell_mem_4664.mem, shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        n_3945 = shape[0];
        m_3943 = shape[1];
        prev_cell_mem_size_4663 = sizeof(float) * shape[0] * shape[1];
    }
    
    struct memblock_device W_bi_mem_device_4993;
    
    W_bi_mem_device_4993.references = NULL;
    memblock_alloc_device(&W_bi_mem_device_4993, W_bi_mem_size_4635);
    if (W_bi_mem_size_4635 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            W_bi_mem_device_4993.mem, CL_TRUE,
                                            0, W_bi_mem_size_4635,
                                            W_bi_mem_4636.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device U_bi_mem_device_4994;
    
    U_bi_mem_device_4994.references = NULL;
    memblock_alloc_device(&U_bi_mem_device_4994, U_bi_mem_size_4637);
    if (U_bi_mem_size_4637 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            U_bi_mem_device_4994.mem, CL_TRUE,
                                            0, U_bi_mem_size_4637,
                                            U_bi_mem_4638.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device b_bi_mem_device_4995;
    
    b_bi_mem_device_4995.references = NULL;
    memblock_alloc_device(&b_bi_mem_device_4995, b_bi_mem_size_4639);
    if (b_bi_mem_size_4639 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            b_bi_mem_device_4995.mem, CL_TRUE,
                                            0, b_bi_mem_size_4639,
                                            b_bi_mem_4640.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device W_ig_mem_device_4996;
    
    W_ig_mem_device_4996.references = NULL;
    memblock_alloc_device(&W_ig_mem_device_4996, W_ig_mem_size_4641);
    if (W_ig_mem_size_4641 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            W_ig_mem_device_4996.mem, CL_TRUE,
                                            0, W_ig_mem_size_4641,
                                            W_ig_mem_4642.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device U_ig_mem_device_4997;
    
    U_ig_mem_device_4997.references = NULL;
    memblock_alloc_device(&U_ig_mem_device_4997, U_ig_mem_size_4643);
    if (U_ig_mem_size_4643 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            U_ig_mem_device_4997.mem, CL_TRUE,
                                            0, U_ig_mem_size_4643,
                                            U_ig_mem_4644.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device b_ig_mem_device_4998;
    
    b_ig_mem_device_4998.references = NULL;
    memblock_alloc_device(&b_ig_mem_device_4998, b_ig_mem_size_4645);
    if (b_ig_mem_size_4645 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            b_ig_mem_device_4998.mem, CL_TRUE,
                                            0, b_ig_mem_size_4645,
                                            b_ig_mem_4646.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device W_fg_mem_device_4999;
    
    W_fg_mem_device_4999.references = NULL;
    memblock_alloc_device(&W_fg_mem_device_4999, W_fg_mem_size_4647);
    if (W_fg_mem_size_4647 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            W_fg_mem_device_4999.mem, CL_TRUE,
                                            0, W_fg_mem_size_4647,
                                            W_fg_mem_4648.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device U_fg_mem_device_5000;
    
    U_fg_mem_device_5000.references = NULL;
    memblock_alloc_device(&U_fg_mem_device_5000, U_fg_mem_size_4649);
    if (U_fg_mem_size_4649 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            U_fg_mem_device_5000.mem, CL_TRUE,
                                            0, U_fg_mem_size_4649,
                                            U_fg_mem_4650.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device b_fg_mem_device_5001;
    
    b_fg_mem_device_5001.references = NULL;
    memblock_alloc_device(&b_fg_mem_device_5001, b_fg_mem_size_4651);
    if (b_fg_mem_size_4651 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            b_fg_mem_device_5001.mem, CL_TRUE,
                                            0, b_fg_mem_size_4651,
                                            b_fg_mem_4652.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device W_og_mem_device_5002;
    
    W_og_mem_device_5002.references = NULL;
    memblock_alloc_device(&W_og_mem_device_5002, W_og_mem_size_4653);
    if (W_og_mem_size_4653 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            W_og_mem_device_5002.mem, CL_TRUE,
                                            0, W_og_mem_size_4653,
                                            W_og_mem_4654.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device U_og_mem_device_5003;
    
    U_og_mem_device_5003.references = NULL;
    memblock_alloc_device(&U_og_mem_device_5003, U_og_mem_size_4655);
    if (U_og_mem_size_4655 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            U_og_mem_device_5003.mem, CL_TRUE,
                                            0, U_og_mem_size_4655,
                                            U_og_mem_4656.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device b_og_mem_device_5004;
    
    b_og_mem_device_5004.references = NULL;
    memblock_alloc_device(&b_og_mem_device_5004, b_og_mem_size_4657);
    if (b_og_mem_size_4657 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            b_og_mem_device_5004.mem, CL_TRUE,
                                            0, b_og_mem_size_4657,
                                            b_og_mem_4658.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device input_mem_device_5005;
    
    input_mem_device_5005.references = NULL;
    memblock_alloc_device(&input_mem_device_5005, input_mem_size_4659);
    if (input_mem_size_4659 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            input_mem_device_5005.mem, CL_TRUE,
                                            0, input_mem_size_4659,
                                            input_mem_4660.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device prev_output_mem_device_5006;
    
    prev_output_mem_device_5006.references = NULL;
    memblock_alloc_device(&prev_output_mem_device_5006,
                          prev_output_mem_size_4661);
    if (prev_output_mem_size_4661 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            prev_output_mem_device_5006.mem,
                                            CL_TRUE, 0,
                                            prev_output_mem_size_4661,
                                            prev_output_mem_4662.mem + 0, 0,
                                            NULL, NULL));
    
    struct memblock_device prev_cell_mem_device_5007;
    
    prev_cell_mem_device_5007.references = NULL;
    memblock_alloc_device(&prev_cell_mem_device_5007, prev_cell_mem_size_4663);
    if (prev_cell_mem_size_4663 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            prev_cell_mem_device_5007.mem,
                                            CL_TRUE, 0, prev_cell_mem_size_4663,
                                            prev_cell_mem_4664.mem + 0, 0, NULL,
                                            NULL));
    
    int32_t out_memsize_4736;
    struct memblock out_mem_4735;
    
    out_mem_4735.references = NULL;
    
    int32_t out_memsize_4738;
    struct memblock out_mem_4737;
    
    out_mem_4737.references = NULL;
    if (perform_warmup) {
        time_runs = 0;
        t_start = get_wall_time();
        main_ret_4992 = futhark_main(W_bi_mem_size_4635, U_bi_mem_size_4637,
                                     b_bi_mem_size_4639, W_ig_mem_size_4641,
                                     U_ig_mem_size_4643, b_ig_mem_size_4645,
                                     W_fg_mem_size_4647, U_fg_mem_size_4649,
                                     b_fg_mem_size_4651, W_og_mem_size_4653,
                                     U_og_mem_size_4655, b_og_mem_size_4657,
                                     input_mem_size_4659,
                                     prev_output_mem_size_4661,
                                     prev_cell_mem_size_4663,
                                     W_bi_mem_device_4993, U_bi_mem_device_4994,
                                     b_bi_mem_device_4995, W_ig_mem_device_4996,
                                     U_ig_mem_device_4997, b_ig_mem_device_4998,
                                     W_fg_mem_device_4999, U_fg_mem_device_5000,
                                     b_fg_mem_device_5001, W_og_mem_device_5002,
                                     U_og_mem_device_5003, b_og_mem_device_5004,
                                     input_mem_device_5005,
                                     prev_output_mem_device_5006,
                                     prev_cell_mem_device_5007, m_3943, o_3944,
                                     n_3945);
        OPENCL_SUCCEED(clFinish(fut_cl_queue));
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%ld\n", elapsed_usec);
        memblock_unref_device(&main_ret_4992.elem_1);
        memblock_unref_device(&main_ret_4992.elem_3);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        if (run == num_runs - 1)
            detail_timing = 1;
        t_start = get_wall_time();
        main_ret_4992 = futhark_main(W_bi_mem_size_4635, U_bi_mem_size_4637,
                                     b_bi_mem_size_4639, W_ig_mem_size_4641,
                                     U_ig_mem_size_4643, b_ig_mem_size_4645,
                                     W_fg_mem_size_4647, U_fg_mem_size_4649,
                                     b_fg_mem_size_4651, W_og_mem_size_4653,
                                     U_og_mem_size_4655, b_og_mem_size_4657,
                                     input_mem_size_4659,
                                     prev_output_mem_size_4661,
                                     prev_cell_mem_size_4663,
                                     W_bi_mem_device_4993, U_bi_mem_device_4994,
                                     b_bi_mem_device_4995, W_ig_mem_device_4996,
                                     U_ig_mem_device_4997, b_ig_mem_device_4998,
                                     W_fg_mem_device_4999, U_fg_mem_device_5000,
                                     b_fg_mem_device_5001, W_og_mem_device_5002,
                                     U_og_mem_device_5003, b_og_mem_device_5004,
                                     input_mem_device_5005,
                                     prev_output_mem_device_5006,
                                     prev_cell_mem_device_5007, m_3943, o_3944,
                                     n_3945);
        OPENCL_SUCCEED(clFinish(fut_cl_queue));
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%ld\n", elapsed_usec);
        if (run < num_runs - 1) {
            memblock_unref_device(&main_ret_4992.elem_1);
            memblock_unref_device(&main_ret_4992.elem_3);
        }
    }
    memblock_unref(&W_bi_mem_4636);
    memblock_unref(&U_bi_mem_4638);
    memblock_unref(&b_bi_mem_4640);
    memblock_unref(&W_ig_mem_4642);
    memblock_unref(&U_ig_mem_4644);
    memblock_unref(&b_ig_mem_4646);
    memblock_unref(&W_fg_mem_4648);
    memblock_unref(&U_fg_mem_4650);
    memblock_unref(&b_fg_mem_4652);
    memblock_unref(&W_og_mem_4654);
    memblock_unref(&U_og_mem_4656);
    memblock_unref(&b_og_mem_4658);
    memblock_unref(&input_mem_4660);
    memblock_unref(&prev_output_mem_4662);
    memblock_unref(&prev_cell_mem_4664);
    out_memsize_4736 = main_ret_4992.elem_0;
    memblock_alloc(&out_mem_4735, out_memsize_4736);
    if (out_memsize_4736 > 0)
        OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                           main_ret_4992.elem_1.mem, CL_TRUE, 0,
                                           out_memsize_4736, out_mem_4735.mem +
                                           0, 0, NULL, NULL));
    out_memsize_4738 = main_ret_4992.elem_2;
    memblock_alloc(&out_mem_4737, out_memsize_4738);
    if (out_memsize_4738 > 0)
        OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                           main_ret_4992.elem_3.mem, CL_TRUE, 0,
                                           out_memsize_4738, out_mem_4737.mem +
                                           0, 0, NULL, NULL));
    if (n_3945 == 0)
        printf("empty(%s)", "[f32]");
    else {
        int print_i_5008;
        
        putchar('[');
        for (print_i_5008 = 0; print_i_5008 < n_3945; print_i_5008++) {
            float *print_elem_5009 = (float *) out_mem_4735.mem + print_i_5008 *
                  m_3943;
            
            if (m_3943 == 0)
                printf("empty(%s)", "f32");
            else {
                int print_i_5010;
                
                putchar('[');
                for (print_i_5010 = 0; print_i_5010 < m_3943; print_i_5010++) {
                    float *print_elem_5011 = (float *) print_elem_5009 +
                          print_i_5010 * 1;
                    
                    printf("%.6ff32", *print_elem_5011);
                    if (print_i_5010 != m_3943 - 1)
                        printf(", ");
                }
                putchar(']');
            }
            if (print_i_5008 != n_3945 - 1)
                printf(", ");
        }
        putchar(']');
    }
    printf("\n");
    if (n_3945 == 0)
        printf("empty(%s)", "[f32]");
    else {
        int print_i_5012;
        
        putchar('[');
        for (print_i_5012 = 0; print_i_5012 < n_3945; print_i_5012++) {
            float *print_elem_5013 = (float *) out_mem_4737.mem + print_i_5012 *
                  m_3943;
            
            if (m_3943 == 0)
                printf("empty(%s)", "f32");
            else {
                int print_i_5014;
                
                putchar('[');
                for (print_i_5014 = 0; print_i_5014 < m_3943; print_i_5014++) {
                    float *print_elem_5015 = (float *) print_elem_5013 +
                          print_i_5014 * 1;
                    
                    printf("%.6ff32", *print_elem_5015);
                    if (print_i_5014 != m_3943 - 1)
                        printf(", ");
                }
                putchar(']');
            }
            if (print_i_5012 != n_3945 - 1)
                printf(", ");
        }
        putchar(']');
    }
    printf("\n");
    
    int total_runtime = 0;
    int total_runs = 0;
    
    if (cl_debug) {
        fprintf(stderr,
                "Kernel map_kernel_4739 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4739runs, (long) map_kernel_4739total_runtime /
                (map_kernel_4739runs != 0 ? map_kernel_4739runs : 1),
                (long) map_kernel_4739total_runtime);
        total_runtime += map_kernel_4739total_runtime;
        total_runs += map_kernel_4739runs;
        fprintf(stderr,
                "Kernel map_kernel_4743 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4743runs, (long) map_kernel_4743total_runtime /
                (map_kernel_4743runs != 0 ? map_kernel_4743runs : 1),
                (long) map_kernel_4743total_runtime);
        total_runtime += map_kernel_4743total_runtime;
        total_runs += map_kernel_4743runs;
        fprintf(stderr,
                "Kernel map_kernel_4752 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4752runs, (long) map_kernel_4752total_runtime /
                (map_kernel_4752runs != 0 ? map_kernel_4752runs : 1),
                (long) map_kernel_4752total_runtime);
        total_runtime += map_kernel_4752total_runtime;
        total_runs += map_kernel_4752runs;
        fprintf(stderr,
                "Kernel map_kernel_4756 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4756runs, (long) map_kernel_4756total_runtime /
                (map_kernel_4756runs != 0 ? map_kernel_4756runs : 1),
                (long) map_kernel_4756total_runtime);
        total_runtime += map_kernel_4756total_runtime;
        total_runs += map_kernel_4756runs;
        fprintf(stderr,
                "Kernel map_kernel_4337 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4337runs, (long) map_kernel_4337total_runtime /
                (map_kernel_4337runs != 0 ? map_kernel_4337runs : 1),
                (long) map_kernel_4337total_runtime);
        total_runtime += map_kernel_4337total_runtime;
        total_runs += map_kernel_4337runs;
        fprintf(stderr,
                "Kernel map_kernel_4360 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4360runs, (long) map_kernel_4360total_runtime /
                (map_kernel_4360runs != 0 ? map_kernel_4360runs : 1),
                (long) map_kernel_4360total_runtime);
        total_runtime += map_kernel_4360total_runtime;
        total_runs += map_kernel_4360runs;
        fprintf(stderr,
                "Kernel map_kernel_4377 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4377runs, (long) map_kernel_4377total_runtime /
                (map_kernel_4377runs != 0 ? map_kernel_4377runs : 1),
                (long) map_kernel_4377total_runtime);
        total_runtime += map_kernel_4377total_runtime;
        total_runs += map_kernel_4377runs;
        fprintf(stderr,
                "Kernel map_kernel_4771 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4771runs, (long) map_kernel_4771total_runtime /
                (map_kernel_4771runs != 0 ? map_kernel_4771runs : 1),
                (long) map_kernel_4771total_runtime);
        total_runtime += map_kernel_4771total_runtime;
        total_runs += map_kernel_4771runs;
        fprintf(stderr,
                "Kernel map_kernel_4780 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4780runs, (long) map_kernel_4780total_runtime /
                (map_kernel_4780runs != 0 ? map_kernel_4780runs : 1),
                (long) map_kernel_4780total_runtime);
        total_runtime += map_kernel_4780total_runtime;
        total_runs += map_kernel_4780runs;
        fprintf(stderr,
                "Kernel map_kernel_4397 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4397runs, (long) map_kernel_4397total_runtime /
                (map_kernel_4397runs != 0 ? map_kernel_4397runs : 1),
                (long) map_kernel_4397total_runtime);
        total_runtime += map_kernel_4397total_runtime;
        total_runs += map_kernel_4397runs;
        fprintf(stderr,
                "Kernel map_kernel_4420 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4420runs, (long) map_kernel_4420total_runtime /
                (map_kernel_4420runs != 0 ? map_kernel_4420runs : 1),
                (long) map_kernel_4420total_runtime);
        total_runtime += map_kernel_4420total_runtime;
        total_runs += map_kernel_4420runs;
        fprintf(stderr,
                "Kernel map_kernel_4437 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4437runs, (long) map_kernel_4437total_runtime /
                (map_kernel_4437runs != 0 ? map_kernel_4437runs : 1),
                (long) map_kernel_4437total_runtime);
        total_runtime += map_kernel_4437total_runtime;
        total_runs += map_kernel_4437runs;
        fprintf(stderr,
                "Kernel map_kernel_4795 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4795runs, (long) map_kernel_4795total_runtime /
                (map_kernel_4795runs != 0 ? map_kernel_4795runs : 1),
                (long) map_kernel_4795total_runtime);
        total_runtime += map_kernel_4795total_runtime;
        total_runs += map_kernel_4795runs;
        fprintf(stderr,
                "Kernel map_kernel_4804 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4804runs, (long) map_kernel_4804total_runtime /
                (map_kernel_4804runs != 0 ? map_kernel_4804runs : 1),
                (long) map_kernel_4804total_runtime);
        total_runtime += map_kernel_4804total_runtime;
        total_runs += map_kernel_4804runs;
        fprintf(stderr,
                "Kernel map_kernel_4457 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4457runs, (long) map_kernel_4457total_runtime /
                (map_kernel_4457runs != 0 ? map_kernel_4457runs : 1),
                (long) map_kernel_4457total_runtime);
        total_runtime += map_kernel_4457total_runtime;
        total_runs += map_kernel_4457runs;
        fprintf(stderr,
                "Kernel map_kernel_4480 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4480runs, (long) map_kernel_4480total_runtime /
                (map_kernel_4480runs != 0 ? map_kernel_4480runs : 1),
                (long) map_kernel_4480total_runtime);
        total_runtime += map_kernel_4480total_runtime;
        total_runs += map_kernel_4480runs;
        fprintf(stderr,
                "Kernel map_kernel_4497 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4497runs, (long) map_kernel_4497total_runtime /
                (map_kernel_4497runs != 0 ? map_kernel_4497runs : 1),
                (long) map_kernel_4497total_runtime);
        total_runtime += map_kernel_4497total_runtime;
        total_runs += map_kernel_4497runs;
        fprintf(stderr,
                "Kernel map_kernel_4819 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4819runs, (long) map_kernel_4819total_runtime /
                (map_kernel_4819runs != 0 ? map_kernel_4819runs : 1),
                (long) map_kernel_4819total_runtime);
        total_runtime += map_kernel_4819total_runtime;
        total_runs += map_kernel_4819runs;
        fprintf(stderr,
                "Kernel map_kernel_4828 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4828runs, (long) map_kernel_4828total_runtime /
                (map_kernel_4828runs != 0 ? map_kernel_4828runs : 1),
                (long) map_kernel_4828total_runtime);
        total_runtime += map_kernel_4828total_runtime;
        total_runs += map_kernel_4828runs;
        fprintf(stderr,
                "Kernel map_kernel_4517 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4517runs, (long) map_kernel_4517total_runtime /
                (map_kernel_4517runs != 0 ? map_kernel_4517runs : 1),
                (long) map_kernel_4517total_runtime);
        total_runtime += map_kernel_4517total_runtime;
        total_runs += map_kernel_4517runs;
        fprintf(stderr,
                "Kernel map_kernel_4540 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4540runs, (long) map_kernel_4540total_runtime /
                (map_kernel_4540runs != 0 ? map_kernel_4540runs : 1),
                (long) map_kernel_4540total_runtime);
        total_runtime += map_kernel_4540total_runtime;
        total_runs += map_kernel_4540runs;
        fprintf(stderr,
                "Kernel map_kernel_4557 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4557runs, (long) map_kernel_4557total_runtime /
                (map_kernel_4557runs != 0 ? map_kernel_4557runs : 1),
                (long) map_kernel_4557total_runtime);
        total_runtime += map_kernel_4557total_runtime;
        total_runs += map_kernel_4557runs;
        fprintf(stderr,
                "Kernel map_kernel_4601 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4601runs, (long) map_kernel_4601total_runtime /
                (map_kernel_4601runs != 0 ? map_kernel_4601runs : 1),
                (long) map_kernel_4601total_runtime);
        total_runtime += map_kernel_4601total_runtime;
        total_runs += map_kernel_4601runs;
        fprintf(stderr,
                "Kernel map_kernel_4577 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_4577runs, (long) map_kernel_4577total_runtime /
                (map_kernel_4577runs != 0 ? map_kernel_4577runs : 1),
                (long) map_kernel_4577total_runtime);
        total_runtime += map_kernel_4577total_runtime;
        total_runs += map_kernel_4577runs;
    }
    if (cl_debug)
        fprintf(stderr, "Ran %d kernels with cumulative runtime: %6ldus\n",
                total_runs, total_runtime);
    memblock_unref_device(&main_ret_4992.elem_1);
    memblock_unref_device(&main_ret_4992.elem_3);
    if (runtime_file != NULL)
        fclose(runtime_file);
    return 0;
}
