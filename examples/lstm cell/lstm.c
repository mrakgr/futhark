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
);
static cl_kernel map_kernel_5053;
static int map_kernel_5053total_runtime = 0;
static int map_kernel_5053runs = 0;
static cl_kernel map_kernel_5098;
static int map_kernel_5098total_runtime = 0;
static int map_kernel_5098runs = 0;
static cl_kernel map_kernel_5122;
static int map_kernel_5122total_runtime = 0;
static int map_kernel_5122runs = 0;
static cl_kernel map_kernel_5167;
static int map_kernel_5167total_runtime = 0;
static int map_kernel_5167runs = 0;
static cl_kernel map_kernel_5191;
static int map_kernel_5191total_runtime = 0;
static int map_kernel_5191runs = 0;
static cl_kernel map_kernel_5236;
static int map_kernel_5236total_runtime = 0;
static int map_kernel_5236runs = 0;
static cl_kernel map_kernel_5260;
static int map_kernel_5260total_runtime = 0;
static int map_kernel_5260runs = 0;
static cl_kernel map_kernel_5305;
static int map_kernel_5305total_runtime = 0;
static int map_kernel_5305runs = 0;
static cl_kernel map_kernel_5349;
static int map_kernel_5349total_runtime = 0;
static int map_kernel_5349runs = 0;
static cl_kernel map_kernel_5325;
static int map_kernel_5325total_runtime = 0;
static int map_kernel_5325runs = 0;
void setup_opencl_and_load_kernels()

{
    cl_int error;
    cl_program prog = setup_opencl(fut_opencl_prelude, fut_opencl_program);
    
    {
        map_kernel_5053 = clCreateKernel(prog, "map_kernel_5053", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_5053");
    }
    {
        map_kernel_5098 = clCreateKernel(prog, "map_kernel_5098", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_5098");
    }
    {
        map_kernel_5122 = clCreateKernel(prog, "map_kernel_5122", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_5122");
    }
    {
        map_kernel_5167 = clCreateKernel(prog, "map_kernel_5167", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_5167");
    }
    {
        map_kernel_5191 = clCreateKernel(prog, "map_kernel_5191", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_5191");
    }
    {
        map_kernel_5236 = clCreateKernel(prog, "map_kernel_5236", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_5236");
    }
    {
        map_kernel_5260 = clCreateKernel(prog, "map_kernel_5260", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_5260");
    }
    {
        map_kernel_5305 = clCreateKernel(prog, "map_kernel_5305", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_5305");
    }
    {
        map_kernel_5349 = clCreateKernel(prog, "map_kernel_5349", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_5349");
    }
    {
        map_kernel_5325 = clCreateKernel(prog, "map_kernel_5325", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_5325");
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
    
    cl_int clCreateBuffer_succeeded_5580;
    
    block->mem = clCreateBuffer(fut_cl_context, CL_MEM_READ_WRITE, size >
                                0 ? size : 1, NULL,
                                &clCreateBuffer_succeeded_5580);
    OPENCL_SUCCEED(clCreateBuffer_succeeded_5580);
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
futhark_main(int32_t W_bi_mem_size_5407, int32_t U_bi_mem_size_5409, int32_t b_bi_mem_size_5411, int32_t W_ig_mem_size_5413, int32_t U_ig_mem_size_5415, int32_t b_ig_mem_size_5417, int32_t W_fg_mem_size_5419, int32_t U_fg_mem_size_5421, int32_t b_fg_mem_size_5423, int32_t W_og_mem_size_5425, int32_t U_og_mem_size_5427, int32_t b_og_mem_size_5429, int32_t input_mem_size_5431, int32_t prev_output_mem_size_5433, int32_t prev_cell_mem_size_5435, struct memblock_device W_bi_mem_5408, struct memblock_device U_bi_mem_5410, struct memblock_device b_bi_mem_5412, struct memblock_device W_ig_mem_5414, struct memblock_device U_ig_mem_5416, struct memblock_device b_ig_mem_5418, struct memblock_device W_fg_mem_5420, struct memblock_device U_fg_mem_5422, struct memblock_device b_fg_mem_5424, struct memblock_device W_og_mem_5426, struct memblock_device U_og_mem_5428, struct memblock_device b_og_mem_5430, struct memblock_device input_mem_5432, struct memblock_device prev_output_mem_5434, struct memblock_device prev_cell_mem_5436, int32_t m_4676, int32_t o_4677, int32_t n_4678);
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
struct tuple_int32_t_device_mem_int32_t_device_mem futhark_main(int32_t W_bi_mem_size_5407,
                                                                int32_t U_bi_mem_size_5409,
                                                                int32_t b_bi_mem_size_5411,
                                                                int32_t W_ig_mem_size_5413,
                                                                int32_t U_ig_mem_size_5415,
                                                                int32_t b_ig_mem_size_5417,
                                                                int32_t W_fg_mem_size_5419,
                                                                int32_t U_fg_mem_size_5421,
                                                                int32_t b_fg_mem_size_5423,
                                                                int32_t W_og_mem_size_5425,
                                                                int32_t U_og_mem_size_5427,
                                                                int32_t b_og_mem_size_5429,
                                                                int32_t input_mem_size_5431,
                                                                int32_t prev_output_mem_size_5433,
                                                                int32_t prev_cell_mem_size_5435,
                                                                struct memblock_device W_bi_mem_5408,
                                                                struct memblock_device U_bi_mem_5410,
                                                                struct memblock_device b_bi_mem_5412,
                                                                struct memblock_device W_ig_mem_5414,
                                                                struct memblock_device U_ig_mem_5416,
                                                                struct memblock_device b_ig_mem_5418,
                                                                struct memblock_device W_fg_mem_5420,
                                                                struct memblock_device U_fg_mem_5422,
                                                                struct memblock_device b_fg_mem_5424,
                                                                struct memblock_device W_og_mem_5426,
                                                                struct memblock_device U_og_mem_5428,
                                                                struct memblock_device b_og_mem_5430,
                                                                struct memblock_device input_mem_5432,
                                                                struct memblock_device prev_output_mem_5434,
                                                                struct memblock_device prev_cell_mem_5436,
                                                                int32_t m_4676,
                                                                int32_t o_4677,
                                                                int32_t n_4678)
{
    int32_t out_memsize_5474;
    struct memblock_device out_mem_5473;
    
    out_mem_5473.references = NULL;
    
    int32_t out_memsize_5476;
    struct memblock_device out_mem_5475;
    
    out_mem_5475.references = NULL;
    
    char cond_4696 = m_4676 == 0;
    int32_t size_4697;
    
    if (cond_4696) {
        size_4697 = 0;
    } else {
        size_4697 = n_4678;
    }
    
    char eq_x_y_4698 = n_4678 == 0;
    char p_and_eq_x_y_4699 = cond_4696 && eq_x_y_4698;
    char not_p_4700 = !cond_4696;
    char assert_arg_4701 = p_and_eq_x_y_4699 || not_p_4700;
    char shape_cert_4702;
    
    if (!assert_arg_4701) {
        fprintf(stderr, "Assertion %s at %s failed.\n", "assert_arg_4701",
                "lstm.fut:31:17-31:17");
        abort();
    }
    
    int32_t nesting_size_5051 = size_4697 * m_4676;
    int32_t x_5438 = 4 * m_4676;
    int32_t bytes_5437 = x_5438 * n_4678;
    struct memblock_device mem_5439;
    
    mem_5439.references = NULL;
    memblock_alloc_device(&mem_5439, bytes_5437);
    if (m_4676 * n_4678 * sizeof(float) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(fut_cl_queue,
                                           prev_output_mem_5434.mem,
                                           mem_5439.mem, 0, 0, m_4676 * n_4678 *
                                           sizeof(float), 0, NULL, NULL));
        if (cl_debug)
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
    }
    
    int32_t x_5441 = 4 * o_4677;
    int32_t bytes_5440 = x_5441 * n_4678;
    struct memblock_device mem_5442;
    
    mem_5442.references = NULL;
    memblock_alloc_device(&mem_5442, bytes_5440);
    if (o_4677 * n_4678 * sizeof(float) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(fut_cl_queue, input_mem_5432.mem,
                                           mem_5442.mem, 0, 0, o_4677 * n_4678 *
                                           sizeof(float), 0, NULL, NULL));
        if (cl_debug)
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
    }
    
    int32_t bytes_5443 = x_5438 * size_4697;
    struct memblock_device mem_5445;
    
    mem_5445.references = NULL;
    memblock_alloc_device(&mem_5445, bytes_5443);
    
    int32_t group_size_5479;
    int32_t num_groups_5480;
    
    group_size_5479 = cl_group_size;
    num_groups_5480 = squot32(m_4676 * size_4697 + group_size_5479 - 1,
                              group_size_5479);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5053, 0, sizeof(b_bi_mem_5412.mem),
                                  &b_bi_mem_5412.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5053, 1, sizeof(W_bi_mem_5408.mem),
                                  &W_bi_mem_5408.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5053, 2, sizeof(m_4676), &m_4676));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5053, 3, sizeof(size_4697),
                                  &size_4697));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5053, 4, sizeof(o_4677), &o_4677));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5053, 5, sizeof(mem_5442.mem),
                                  &mem_5442.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5053, 6, sizeof(U_bi_mem_5410.mem),
                                  &U_bi_mem_5410.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5053, 7, sizeof(n_4678), &n_4678));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5053, 8, sizeof(mem_5439.mem),
                                  &mem_5439.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5053, 9, sizeof(mem_5445.mem),
                                  &mem_5445.mem));
    if (1 * (num_groups_5480 * group_size_5479) != 0) {
        const size_t global_work_size_5506[1] = {num_groups_5480 *
                     group_size_5479};
        const size_t local_work_size_5510[1] = {group_size_5479};
        int64_t time_start_5507, time_end_5508;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_5053");
            fprintf(stderr, "%zu", global_work_size_5506[0]);
            fprintf(stderr, "].\n");
            time_start_5507 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_5053, 1,
                                              NULL, global_work_size_5506,
                                              local_work_size_5510, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_5508 = get_wall_time();
            
            long time_diff_5509 = time_end_5508 - time_start_5507;
            
            if (detail_timing) {
                map_kernel_5053total_runtime += time_diff_5509;
                map_kernel_5053runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_5053",
                        (int) time_diff_5509);
            }
        }
    }
    
    int32_t nesting_size_5096 = n_4678 * m_4676;
    struct memblock_device mem_5448;
    
    mem_5448.references = NULL;
    memblock_alloc_device(&mem_5448, bytes_5437);
    
    int32_t group_size_5481;
    int32_t num_groups_5482;
    
    group_size_5481 = cl_group_size;
    num_groups_5482 = squot32(m_4676 * n_4678 + group_size_5481 - 1,
                              group_size_5481);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5098, 0, sizeof(m_4676), &m_4676));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5098, 1, sizeof(size_4697),
                                  &size_4697));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5098, 2, sizeof(mem_5445.mem),
                                  &mem_5445.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5098, 3, sizeof(n_4678), &n_4678));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5098, 4, sizeof(mem_5448.mem),
                                  &mem_5448.mem));
    if (1 * (num_groups_5482 * group_size_5481) != 0) {
        const size_t global_work_size_5511[1] = {num_groups_5482 *
                     group_size_5481};
        const size_t local_work_size_5515[1] = {group_size_5481};
        int64_t time_start_5512, time_end_5513;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_5098");
            fprintf(stderr, "%zu", global_work_size_5511[0]);
            fprintf(stderr, "].\n");
            time_start_5512 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_5098, 1,
                                              NULL, global_work_size_5511,
                                              local_work_size_5515, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_5513 = get_wall_time();
            
            long time_diff_5514 = time_end_5513 - time_start_5512;
            
            if (detail_timing) {
                map_kernel_5098total_runtime += time_diff_5514;
                map_kernel_5098runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_5098",
                        (int) time_diff_5514);
            }
        }
    }
    
    struct memblock_device mem_5451;
    
    mem_5451.references = NULL;
    memblock_alloc_device(&mem_5451, bytes_5443);
    
    int32_t group_size_5485;
    int32_t num_groups_5486;
    
    group_size_5485 = cl_group_size;
    num_groups_5486 = squot32(m_4676 * size_4697 + group_size_5485 - 1,
                              group_size_5485);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5122, 0, sizeof(m_4676), &m_4676));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5122, 1, sizeof(U_ig_mem_5416.mem),
                                  &U_ig_mem_5416.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5122, 2, sizeof(size_4697),
                                  &size_4697));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5122, 3, sizeof(o_4677), &o_4677));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5122, 4, sizeof(mem_5442.mem),
                                  &mem_5442.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5122, 5, sizeof(W_ig_mem_5414.mem),
                                  &W_ig_mem_5414.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5122, 6, sizeof(n_4678), &n_4678));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5122, 7, sizeof(b_ig_mem_5418.mem),
                                  &b_ig_mem_5418.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5122, 8, sizeof(mem_5439.mem),
                                  &mem_5439.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5122, 9, sizeof(mem_5451.mem),
                                  &mem_5451.mem));
    if (1 * (num_groups_5486 * group_size_5485) != 0) {
        const size_t global_work_size_5516[1] = {num_groups_5486 *
                     group_size_5485};
        const size_t local_work_size_5520[1] = {group_size_5485};
        int64_t time_start_5517, time_end_5518;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_5122");
            fprintf(stderr, "%zu", global_work_size_5516[0]);
            fprintf(stderr, "].\n");
            time_start_5517 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_5122, 1,
                                              NULL, global_work_size_5516,
                                              local_work_size_5520, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_5518 = get_wall_time();
            
            long time_diff_5519 = time_end_5518 - time_start_5517;
            
            if (detail_timing) {
                map_kernel_5122total_runtime += time_diff_5519;
                map_kernel_5122runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_5122",
                        (int) time_diff_5519);
            }
        }
    }
    
    struct memblock_device mem_5454;
    
    mem_5454.references = NULL;
    memblock_alloc_device(&mem_5454, bytes_5437);
    
    int32_t group_size_5487;
    int32_t num_groups_5488;
    
    group_size_5487 = cl_group_size;
    num_groups_5488 = squot32(m_4676 * n_4678 + group_size_5487 - 1,
                              group_size_5487);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5167, 0, sizeof(m_4676), &m_4676));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5167, 1, sizeof(size_4697),
                                  &size_4697));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5167, 2, sizeof(n_4678), &n_4678));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5167, 3, sizeof(mem_5451.mem),
                                  &mem_5451.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5167, 4, sizeof(mem_5454.mem),
                                  &mem_5454.mem));
    if (1 * (num_groups_5488 * group_size_5487) != 0) {
        const size_t global_work_size_5521[1] = {num_groups_5488 *
                     group_size_5487};
        const size_t local_work_size_5525[1] = {group_size_5487};
        int64_t time_start_5522, time_end_5523;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_5167");
            fprintf(stderr, "%zu", global_work_size_5521[0]);
            fprintf(stderr, "].\n");
            time_start_5522 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_5167, 1,
                                              NULL, global_work_size_5521,
                                              local_work_size_5525, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_5523 = get_wall_time();
            
            long time_diff_5524 = time_end_5523 - time_start_5522;
            
            if (detail_timing) {
                map_kernel_5167total_runtime += time_diff_5524;
                map_kernel_5167runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_5167",
                        (int) time_diff_5524);
            }
        }
    }
    
    struct memblock_device mem_5457;
    
    mem_5457.references = NULL;
    memblock_alloc_device(&mem_5457, bytes_5443);
    
    int32_t group_size_5491;
    int32_t num_groups_5492;
    
    group_size_5491 = cl_group_size;
    num_groups_5492 = squot32(m_4676 * size_4697 + group_size_5491 - 1,
                              group_size_5491);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5191, 0, sizeof(W_fg_mem_5420.mem),
                                  &W_fg_mem_5420.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5191, 1, sizeof(m_4676), &m_4676));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5191, 2, sizeof(b_fg_mem_5424.mem),
                                  &b_fg_mem_5424.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5191, 3, sizeof(size_4697),
                                  &size_4697));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5191, 4, sizeof(o_4677), &o_4677));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5191, 5, sizeof(mem_5442.mem),
                                  &mem_5442.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5191, 6, sizeof(U_fg_mem_5422.mem),
                                  &U_fg_mem_5422.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5191, 7, sizeof(n_4678), &n_4678));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5191, 8, sizeof(mem_5439.mem),
                                  &mem_5439.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5191, 9, sizeof(mem_5457.mem),
                                  &mem_5457.mem));
    if (1 * (num_groups_5492 * group_size_5491) != 0) {
        const size_t global_work_size_5526[1] = {num_groups_5492 *
                     group_size_5491};
        const size_t local_work_size_5530[1] = {group_size_5491};
        int64_t time_start_5527, time_end_5528;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_5191");
            fprintf(stderr, "%zu", global_work_size_5526[0]);
            fprintf(stderr, "].\n");
            time_start_5527 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_5191, 1,
                                              NULL, global_work_size_5526,
                                              local_work_size_5530, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_5528 = get_wall_time();
            
            long time_diff_5529 = time_end_5528 - time_start_5527;
            
            if (detail_timing) {
                map_kernel_5191total_runtime += time_diff_5529;
                map_kernel_5191runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_5191",
                        (int) time_diff_5529);
            }
        }
    }
    
    struct memblock_device mem_5460;
    
    mem_5460.references = NULL;
    memblock_alloc_device(&mem_5460, bytes_5437);
    
    int32_t group_size_5493;
    int32_t num_groups_5494;
    
    group_size_5493 = cl_group_size;
    num_groups_5494 = squot32(m_4676 * n_4678 + group_size_5493 - 1,
                              group_size_5493);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5236, 0, sizeof(m_4676), &m_4676));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5236, 1, sizeof(size_4697),
                                  &size_4697));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5236, 2, sizeof(mem_5457.mem),
                                  &mem_5457.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5236, 3, sizeof(n_4678), &n_4678));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5236, 4, sizeof(mem_5460.mem),
                                  &mem_5460.mem));
    if (1 * (num_groups_5494 * group_size_5493) != 0) {
        const size_t global_work_size_5531[1] = {num_groups_5494 *
                     group_size_5493};
        const size_t local_work_size_5535[1] = {group_size_5493};
        int64_t time_start_5532, time_end_5533;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_5236");
            fprintf(stderr, "%zu", global_work_size_5531[0]);
            fprintf(stderr, "].\n");
            time_start_5532 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_5236, 1,
                                              NULL, global_work_size_5531,
                                              local_work_size_5535, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_5533 = get_wall_time();
            
            long time_diff_5534 = time_end_5533 - time_start_5532;
            
            if (detail_timing) {
                map_kernel_5236total_runtime += time_diff_5534;
                map_kernel_5236runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_5236",
                        (int) time_diff_5534);
            }
        }
    }
    
    struct memblock_device mem_5463;
    
    mem_5463.references = NULL;
    memblock_alloc_device(&mem_5463, bytes_5443);
    
    int32_t group_size_5497;
    int32_t num_groups_5498;
    
    group_size_5497 = cl_group_size;
    num_groups_5498 = squot32(m_4676 * size_4697 + group_size_5497 - 1,
                              group_size_5497);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5260, 0, sizeof(U_og_mem_5428.mem),
                                  &U_og_mem_5428.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5260, 1, sizeof(m_4676), &m_4676));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5260, 2, sizeof(size_4697),
                                  &size_4697));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5260, 3, sizeof(o_4677), &o_4677));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5260, 4, sizeof(W_og_mem_5426.mem),
                                  &W_og_mem_5426.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5260, 5, sizeof(mem_5442.mem),
                                  &mem_5442.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5260, 6, sizeof(b_og_mem_5430.mem),
                                  &b_og_mem_5430.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5260, 7, sizeof(n_4678), &n_4678));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5260, 8, sizeof(mem_5439.mem),
                                  &mem_5439.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5260, 9, sizeof(mem_5463.mem),
                                  &mem_5463.mem));
    if (1 * (num_groups_5498 * group_size_5497) != 0) {
        const size_t global_work_size_5536[1] = {num_groups_5498 *
                     group_size_5497};
        const size_t local_work_size_5540[1] = {group_size_5497};
        int64_t time_start_5537, time_end_5538;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_5260");
            fprintf(stderr, "%zu", global_work_size_5536[0]);
            fprintf(stderr, "].\n");
            time_start_5537 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_5260, 1,
                                              NULL, global_work_size_5536,
                                              local_work_size_5540, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_5538 = get_wall_time();
            
            long time_diff_5539 = time_end_5538 - time_start_5537;
            
            if (detail_timing) {
                map_kernel_5260total_runtime += time_diff_5539;
                map_kernel_5260runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_5260",
                        (int) time_diff_5539);
            }
        }
    }
    
    struct memblock_device mem_5466;
    
    mem_5466.references = NULL;
    memblock_alloc_device(&mem_5466, bytes_5437);
    
    int32_t group_size_5499;
    int32_t num_groups_5500;
    
    group_size_5499 = cl_group_size;
    num_groups_5500 = squot32(m_4676 * n_4678 + group_size_5499 - 1,
                              group_size_5499);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5305, 0, sizeof(m_4676), &m_4676));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5305, 1, sizeof(size_4697),
                                  &size_4697));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5305, 2, sizeof(n_4678), &n_4678));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5305, 3, sizeof(mem_5463.mem),
                                  &mem_5463.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5305, 4, sizeof(mem_5466.mem),
                                  &mem_5466.mem));
    if (1 * (num_groups_5500 * group_size_5499) != 0) {
        const size_t global_work_size_5541[1] = {num_groups_5500 *
                     group_size_5499};
        const size_t local_work_size_5545[1] = {group_size_5499};
        int64_t time_start_5542, time_end_5543;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_5305");
            fprintf(stderr, "%zu", global_work_size_5541[0]);
            fprintf(stderr, "].\n");
            time_start_5542 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_5305, 1,
                                              NULL, global_work_size_5541,
                                              local_work_size_5545, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_5543 = get_wall_time();
            
            long time_diff_5544 = time_end_5543 - time_start_5542;
            
            if (detail_timing) {
                map_kernel_5305total_runtime += time_diff_5544;
                map_kernel_5305runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_5305",
                        (int) time_diff_5544);
            }
        }
    }
    
    struct memblock_device mem_5469;
    
    mem_5469.references = NULL;
    memblock_alloc_device(&mem_5469, bytes_5443);
    
    int32_t group_size_5501;
    int32_t num_groups_5502;
    
    group_size_5501 = cl_group_size;
    num_groups_5502 = squot32(m_4676 * size_4697 + group_size_5501 - 1,
                              group_size_5501);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5349, 0, sizeof(mem_5448.mem),
                                  &mem_5448.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5349, 1,
                                  sizeof(prev_cell_mem_5436.mem),
                                  &prev_cell_mem_5436.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5349, 2, sizeof(m_4676), &m_4676));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5349, 3, sizeof(mem_5460.mem),
                                  &mem_5460.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5349, 4, sizeof(size_4697),
                                  &size_4697));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5349, 5, sizeof(n_4678), &n_4678));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5349, 6, sizeof(mem_5454.mem),
                                  &mem_5454.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5349, 7, sizeof(mem_5469.mem),
                                  &mem_5469.mem));
    if (1 * (num_groups_5502 * group_size_5501) != 0) {
        const size_t global_work_size_5546[1] = {num_groups_5502 *
                     group_size_5501};
        const size_t local_work_size_5550[1] = {group_size_5501};
        int64_t time_start_5547, time_end_5548;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_5349");
            fprintf(stderr, "%zu", global_work_size_5546[0]);
            fprintf(stderr, "].\n");
            time_start_5547 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_5349, 1,
                                              NULL, global_work_size_5546,
                                              local_work_size_5550, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_5548 = get_wall_time();
            
            long time_diff_5549 = time_end_5548 - time_start_5547;
            
            if (detail_timing) {
                map_kernel_5349total_runtime += time_diff_5549;
                map_kernel_5349runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_5349",
                        (int) time_diff_5549);
            }
        }
    }
    
    struct memblock_device mem_5472;
    
    mem_5472.references = NULL;
    memblock_alloc_device(&mem_5472, bytes_5437);
    
    int32_t group_size_5503;
    int32_t num_groups_5504;
    
    group_size_5503 = cl_group_size;
    num_groups_5504 = squot32(m_4676 * n_4678 + group_size_5503 - 1,
                              group_size_5503);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5325, 0, sizeof(m_4676), &m_4676));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5325, 1, sizeof(size_4697),
                                  &size_4697));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5325, 2, sizeof(mem_5469.mem),
                                  &mem_5469.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5325, 3, sizeof(mem_5466.mem),
                                  &mem_5466.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5325, 4, sizeof(n_4678), &n_4678));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_5325, 5, sizeof(mem_5472.mem),
                                  &mem_5472.mem));
    if (1 * (num_groups_5504 * group_size_5503) != 0) {
        const size_t global_work_size_5551[1] = {num_groups_5504 *
                     group_size_5503};
        const size_t local_work_size_5555[1] = {group_size_5503};
        int64_t time_start_5552, time_end_5553;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_5325");
            fprintf(stderr, "%zu", global_work_size_5551[0]);
            fprintf(stderr, "].\n");
            time_start_5552 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_5325, 1,
                                              NULL, global_work_size_5551,
                                              local_work_size_5555, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_5553 = get_wall_time();
            
            long time_diff_5554 = time_end_5553 - time_start_5552;
            
            if (detail_timing) {
                map_kernel_5325total_runtime += time_diff_5554;
                map_kernel_5325runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_5325",
                        (int) time_diff_5554);
            }
        }
    }
    memblock_set_device(&out_mem_5473, &mem_5472);
    out_memsize_5474 = bytes_5437;
    memblock_set_device(&out_mem_5475, &mem_5469);
    out_memsize_5476 = bytes_5443;
    
    struct tuple_int32_t_device_mem_int32_t_device_mem retval_5505;
    
    retval_5505.elem_0 = out_memsize_5474;
    retval_5505.elem_1.references = NULL;
    memblock_set_device(&retval_5505.elem_1, &out_mem_5473);
    retval_5505.elem_2 = out_memsize_5476;
    retval_5505.elem_3.references = NULL;
    memblock_set_device(&retval_5505.elem_3, &out_mem_5475);
    memblock_unref_device(&out_mem_5473);
    memblock_unref_device(&out_mem_5475);
    memblock_unref_device(&mem_5439);
    memblock_unref_device(&mem_5442);
    memblock_unref_device(&mem_5445);
    memblock_unref_device(&mem_5448);
    memblock_unref_device(&mem_5451);
    memblock_unref_device(&mem_5454);
    memblock_unref_device(&mem_5457);
    memblock_unref_device(&mem_5460);
    memblock_unref_device(&mem_5463);
    memblock_unref_device(&mem_5466);
    memblock_unref_device(&mem_5469);
    memblock_unref_device(&mem_5472);
    return retval_5505;
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
    
    int32_t W_bi_mem_size_5407;
    int32_t U_bi_mem_size_5409;
    int32_t b_bi_mem_size_5411;
    int32_t W_ig_mem_size_5413;
    int32_t U_ig_mem_size_5415;
    int32_t b_ig_mem_size_5417;
    int32_t W_fg_mem_size_5419;
    int32_t U_fg_mem_size_5421;
    int32_t b_fg_mem_size_5423;
    int32_t W_og_mem_size_5425;
    int32_t U_og_mem_size_5427;
    int32_t b_og_mem_size_5429;
    int32_t input_mem_size_5431;
    int32_t prev_output_mem_size_5433;
    int32_t prev_cell_mem_size_5435;
    struct memblock W_bi_mem_5408;
    
    W_bi_mem_5408.references = NULL;
    memblock_alloc(&W_bi_mem_5408, 0);
    
    struct memblock U_bi_mem_5410;
    
    U_bi_mem_5410.references = NULL;
    memblock_alloc(&U_bi_mem_5410, 0);
    
    struct memblock b_bi_mem_5412;
    
    b_bi_mem_5412.references = NULL;
    memblock_alloc(&b_bi_mem_5412, 0);
    
    struct memblock W_ig_mem_5414;
    
    W_ig_mem_5414.references = NULL;
    memblock_alloc(&W_ig_mem_5414, 0);
    
    struct memblock U_ig_mem_5416;
    
    U_ig_mem_5416.references = NULL;
    memblock_alloc(&U_ig_mem_5416, 0);
    
    struct memblock b_ig_mem_5418;
    
    b_ig_mem_5418.references = NULL;
    memblock_alloc(&b_ig_mem_5418, 0);
    
    struct memblock W_fg_mem_5420;
    
    W_fg_mem_5420.references = NULL;
    memblock_alloc(&W_fg_mem_5420, 0);
    
    struct memblock U_fg_mem_5422;
    
    U_fg_mem_5422.references = NULL;
    memblock_alloc(&U_fg_mem_5422, 0);
    
    struct memblock b_fg_mem_5424;
    
    b_fg_mem_5424.references = NULL;
    memblock_alloc(&b_fg_mem_5424, 0);
    
    struct memblock W_og_mem_5426;
    
    W_og_mem_5426.references = NULL;
    memblock_alloc(&W_og_mem_5426, 0);
    
    struct memblock U_og_mem_5428;
    
    U_og_mem_5428.references = NULL;
    memblock_alloc(&U_og_mem_5428, 0);
    
    struct memblock b_og_mem_5430;
    
    b_og_mem_5430.references = NULL;
    memblock_alloc(&b_og_mem_5430, 0);
    
    struct memblock input_mem_5432;
    
    input_mem_5432.references = NULL;
    memblock_alloc(&input_mem_5432, 0);
    
    struct memblock prev_output_mem_5434;
    
    prev_output_mem_5434.references = NULL;
    memblock_alloc(&prev_output_mem_5434, 0);
    
    struct memblock prev_cell_mem_5436;
    
    prev_cell_mem_5436.references = NULL;
    memblock_alloc(&prev_cell_mem_5436, 0);
    
    int32_t m_4676;
    int32_t o_4677;
    int32_t n_4678;
    struct tuple_int32_t_device_mem_int32_t_device_mem main_ret_5556;
    
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float, (void **) &W_bi_mem_5408.mem,
                       shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        m_4676 = shape[0];
        o_4677 = shape[1];
        W_bi_mem_size_5407 = sizeof(float) * shape[0] * shape[1];
    }
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float, (void **) &U_bi_mem_5410.mem,
                       shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        m_4676 = shape[0];
        m_4676 = shape[1];
        U_bi_mem_size_5409 = sizeof(float) * shape[0] * shape[1];
    }
    {
        int64_t shape[1];
        
        if (read_array(sizeof(float), read_float, (void **) &b_bi_mem_5412.mem,
                       shape, 1) != 0)
            panic(1, "Syntax error when reading %s.\n", "[f32]");
        m_4676 = shape[0];
        b_bi_mem_size_5411 = sizeof(float) * shape[0];
    }
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float, (void **) &W_ig_mem_5414.mem,
                       shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        m_4676 = shape[0];
        o_4677 = shape[1];
        W_ig_mem_size_5413 = sizeof(float) * shape[0] * shape[1];
    }
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float, (void **) &U_ig_mem_5416.mem,
                       shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        m_4676 = shape[0];
        m_4676 = shape[1];
        U_ig_mem_size_5415 = sizeof(float) * shape[0] * shape[1];
    }
    {
        int64_t shape[1];
        
        if (read_array(sizeof(float), read_float, (void **) &b_ig_mem_5418.mem,
                       shape, 1) != 0)
            panic(1, "Syntax error when reading %s.\n", "[f32]");
        m_4676 = shape[0];
        b_ig_mem_size_5417 = sizeof(float) * shape[0];
    }
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float, (void **) &W_fg_mem_5420.mem,
                       shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        m_4676 = shape[0];
        o_4677 = shape[1];
        W_fg_mem_size_5419 = sizeof(float) * shape[0] * shape[1];
    }
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float, (void **) &U_fg_mem_5422.mem,
                       shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        m_4676 = shape[0];
        m_4676 = shape[1];
        U_fg_mem_size_5421 = sizeof(float) * shape[0] * shape[1];
    }
    {
        int64_t shape[1];
        
        if (read_array(sizeof(float), read_float, (void **) &b_fg_mem_5424.mem,
                       shape, 1) != 0)
            panic(1, "Syntax error when reading %s.\n", "[f32]");
        m_4676 = shape[0];
        b_fg_mem_size_5423 = sizeof(float) * shape[0];
    }
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float, (void **) &W_og_mem_5426.mem,
                       shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        m_4676 = shape[0];
        o_4677 = shape[1];
        W_og_mem_size_5425 = sizeof(float) * shape[0] * shape[1];
    }
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float, (void **) &U_og_mem_5428.mem,
                       shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        m_4676 = shape[0];
        m_4676 = shape[1];
        U_og_mem_size_5427 = sizeof(float) * shape[0] * shape[1];
    }
    {
        int64_t shape[1];
        
        if (read_array(sizeof(float), read_float, (void **) &b_og_mem_5430.mem,
                       shape, 1) != 0)
            panic(1, "Syntax error when reading %s.\n", "[f32]");
        m_4676 = shape[0];
        b_og_mem_size_5429 = sizeof(float) * shape[0];
    }
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float, (void **) &input_mem_5432.mem,
                       shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        o_4677 = shape[0];
        n_4678 = shape[1];
        input_mem_size_5431 = sizeof(float) * shape[0] * shape[1];
    }
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float,
                       (void **) &prev_output_mem_5434.mem, shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        m_4676 = shape[0];
        n_4678 = shape[1];
        prev_output_mem_size_5433 = sizeof(float) * shape[0] * shape[1];
    }
    {
        int64_t shape[2];
        
        if (read_array(sizeof(float), read_float,
                       (void **) &prev_cell_mem_5436.mem, shape, 2) != 0)
            panic(1, "Syntax error when reading %s.\n", "[[f32]]");
        m_4676 = shape[0];
        n_4678 = shape[1];
        prev_cell_mem_size_5435 = sizeof(float) * shape[0] * shape[1];
    }
    
    struct memblock_device W_bi_mem_device_5557;
    
    W_bi_mem_device_5557.references = NULL;
    memblock_alloc_device(&W_bi_mem_device_5557, W_bi_mem_size_5407);
    if (W_bi_mem_size_5407 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            W_bi_mem_device_5557.mem, CL_TRUE,
                                            0, W_bi_mem_size_5407,
                                            W_bi_mem_5408.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device U_bi_mem_device_5558;
    
    U_bi_mem_device_5558.references = NULL;
    memblock_alloc_device(&U_bi_mem_device_5558, U_bi_mem_size_5409);
    if (U_bi_mem_size_5409 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            U_bi_mem_device_5558.mem, CL_TRUE,
                                            0, U_bi_mem_size_5409,
                                            U_bi_mem_5410.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device b_bi_mem_device_5559;
    
    b_bi_mem_device_5559.references = NULL;
    memblock_alloc_device(&b_bi_mem_device_5559, b_bi_mem_size_5411);
    if (b_bi_mem_size_5411 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            b_bi_mem_device_5559.mem, CL_TRUE,
                                            0, b_bi_mem_size_5411,
                                            b_bi_mem_5412.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device W_ig_mem_device_5560;
    
    W_ig_mem_device_5560.references = NULL;
    memblock_alloc_device(&W_ig_mem_device_5560, W_ig_mem_size_5413);
    if (W_ig_mem_size_5413 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            W_ig_mem_device_5560.mem, CL_TRUE,
                                            0, W_ig_mem_size_5413,
                                            W_ig_mem_5414.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device U_ig_mem_device_5561;
    
    U_ig_mem_device_5561.references = NULL;
    memblock_alloc_device(&U_ig_mem_device_5561, U_ig_mem_size_5415);
    if (U_ig_mem_size_5415 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            U_ig_mem_device_5561.mem, CL_TRUE,
                                            0, U_ig_mem_size_5415,
                                            U_ig_mem_5416.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device b_ig_mem_device_5562;
    
    b_ig_mem_device_5562.references = NULL;
    memblock_alloc_device(&b_ig_mem_device_5562, b_ig_mem_size_5417);
    if (b_ig_mem_size_5417 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            b_ig_mem_device_5562.mem, CL_TRUE,
                                            0, b_ig_mem_size_5417,
                                            b_ig_mem_5418.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device W_fg_mem_device_5563;
    
    W_fg_mem_device_5563.references = NULL;
    memblock_alloc_device(&W_fg_mem_device_5563, W_fg_mem_size_5419);
    if (W_fg_mem_size_5419 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            W_fg_mem_device_5563.mem, CL_TRUE,
                                            0, W_fg_mem_size_5419,
                                            W_fg_mem_5420.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device U_fg_mem_device_5564;
    
    U_fg_mem_device_5564.references = NULL;
    memblock_alloc_device(&U_fg_mem_device_5564, U_fg_mem_size_5421);
    if (U_fg_mem_size_5421 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            U_fg_mem_device_5564.mem, CL_TRUE,
                                            0, U_fg_mem_size_5421,
                                            U_fg_mem_5422.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device b_fg_mem_device_5565;
    
    b_fg_mem_device_5565.references = NULL;
    memblock_alloc_device(&b_fg_mem_device_5565, b_fg_mem_size_5423);
    if (b_fg_mem_size_5423 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            b_fg_mem_device_5565.mem, CL_TRUE,
                                            0, b_fg_mem_size_5423,
                                            b_fg_mem_5424.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device W_og_mem_device_5566;
    
    W_og_mem_device_5566.references = NULL;
    memblock_alloc_device(&W_og_mem_device_5566, W_og_mem_size_5425);
    if (W_og_mem_size_5425 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            W_og_mem_device_5566.mem, CL_TRUE,
                                            0, W_og_mem_size_5425,
                                            W_og_mem_5426.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device U_og_mem_device_5567;
    
    U_og_mem_device_5567.references = NULL;
    memblock_alloc_device(&U_og_mem_device_5567, U_og_mem_size_5427);
    if (U_og_mem_size_5427 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            U_og_mem_device_5567.mem, CL_TRUE,
                                            0, U_og_mem_size_5427,
                                            U_og_mem_5428.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device b_og_mem_device_5568;
    
    b_og_mem_device_5568.references = NULL;
    memblock_alloc_device(&b_og_mem_device_5568, b_og_mem_size_5429);
    if (b_og_mem_size_5429 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            b_og_mem_device_5568.mem, CL_TRUE,
                                            0, b_og_mem_size_5429,
                                            b_og_mem_5430.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device input_mem_device_5569;
    
    input_mem_device_5569.references = NULL;
    memblock_alloc_device(&input_mem_device_5569, input_mem_size_5431);
    if (input_mem_size_5431 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            input_mem_device_5569.mem, CL_TRUE,
                                            0, input_mem_size_5431,
                                            input_mem_5432.mem + 0, 0, NULL,
                                            NULL));
    
    struct memblock_device prev_output_mem_device_5570;
    
    prev_output_mem_device_5570.references = NULL;
    memblock_alloc_device(&prev_output_mem_device_5570,
                          prev_output_mem_size_5433);
    if (prev_output_mem_size_5433 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            prev_output_mem_device_5570.mem,
                                            CL_TRUE, 0,
                                            prev_output_mem_size_5433,
                                            prev_output_mem_5434.mem + 0, 0,
                                            NULL, NULL));
    
    struct memblock_device prev_cell_mem_device_5571;
    
    prev_cell_mem_device_5571.references = NULL;
    memblock_alloc_device(&prev_cell_mem_device_5571, prev_cell_mem_size_5435);
    if (prev_cell_mem_size_5435 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            prev_cell_mem_device_5571.mem,
                                            CL_TRUE, 0, prev_cell_mem_size_5435,
                                            prev_cell_mem_5436.mem + 0, 0, NULL,
                                            NULL));
    
    int32_t out_memsize_5474;
    struct memblock out_mem_5473;
    
    out_mem_5473.references = NULL;
    
    int32_t out_memsize_5476;
    struct memblock out_mem_5475;
    
    out_mem_5475.references = NULL;
    if (perform_warmup) {
        time_runs = 0;
        t_start = get_wall_time();
        main_ret_5556 = futhark_main(W_bi_mem_size_5407, U_bi_mem_size_5409,
                                     b_bi_mem_size_5411, W_ig_mem_size_5413,
                                     U_ig_mem_size_5415, b_ig_mem_size_5417,
                                     W_fg_mem_size_5419, U_fg_mem_size_5421,
                                     b_fg_mem_size_5423, W_og_mem_size_5425,
                                     U_og_mem_size_5427, b_og_mem_size_5429,
                                     input_mem_size_5431,
                                     prev_output_mem_size_5433,
                                     prev_cell_mem_size_5435,
                                     W_bi_mem_device_5557, U_bi_mem_device_5558,
                                     b_bi_mem_device_5559, W_ig_mem_device_5560,
                                     U_ig_mem_device_5561, b_ig_mem_device_5562,
                                     W_fg_mem_device_5563, U_fg_mem_device_5564,
                                     b_fg_mem_device_5565, W_og_mem_device_5566,
                                     U_og_mem_device_5567, b_og_mem_device_5568,
                                     input_mem_device_5569,
                                     prev_output_mem_device_5570,
                                     prev_cell_mem_device_5571, m_4676, o_4677,
                                     n_4678);
        OPENCL_SUCCEED(clFinish(fut_cl_queue));
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%ld\n", elapsed_usec);
        memblock_unref_device(&main_ret_5556.elem_1);
        memblock_unref_device(&main_ret_5556.elem_3);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        if (run == num_runs - 1)
            detail_timing = 1;
        t_start = get_wall_time();
        main_ret_5556 = futhark_main(W_bi_mem_size_5407, U_bi_mem_size_5409,
                                     b_bi_mem_size_5411, W_ig_mem_size_5413,
                                     U_ig_mem_size_5415, b_ig_mem_size_5417,
                                     W_fg_mem_size_5419, U_fg_mem_size_5421,
                                     b_fg_mem_size_5423, W_og_mem_size_5425,
                                     U_og_mem_size_5427, b_og_mem_size_5429,
                                     input_mem_size_5431,
                                     prev_output_mem_size_5433,
                                     prev_cell_mem_size_5435,
                                     W_bi_mem_device_5557, U_bi_mem_device_5558,
                                     b_bi_mem_device_5559, W_ig_mem_device_5560,
                                     U_ig_mem_device_5561, b_ig_mem_device_5562,
                                     W_fg_mem_device_5563, U_fg_mem_device_5564,
                                     b_fg_mem_device_5565, W_og_mem_device_5566,
                                     U_og_mem_device_5567, b_og_mem_device_5568,
                                     input_mem_device_5569,
                                     prev_output_mem_device_5570,
                                     prev_cell_mem_device_5571, m_4676, o_4677,
                                     n_4678);
        OPENCL_SUCCEED(clFinish(fut_cl_queue));
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%ld\n", elapsed_usec);
        if (run < num_runs - 1) {
            memblock_unref_device(&main_ret_5556.elem_1);
            memblock_unref_device(&main_ret_5556.elem_3);
        }
    }
    memblock_unref(&W_bi_mem_5408);
    memblock_unref(&U_bi_mem_5410);
    memblock_unref(&b_bi_mem_5412);
    memblock_unref(&W_ig_mem_5414);
    memblock_unref(&U_ig_mem_5416);
    memblock_unref(&b_ig_mem_5418);
    memblock_unref(&W_fg_mem_5420);
    memblock_unref(&U_fg_mem_5422);
    memblock_unref(&b_fg_mem_5424);
    memblock_unref(&W_og_mem_5426);
    memblock_unref(&U_og_mem_5428);
    memblock_unref(&b_og_mem_5430);
    memblock_unref(&input_mem_5432);
    memblock_unref(&prev_output_mem_5434);
    memblock_unref(&prev_cell_mem_5436);
    out_memsize_5474 = main_ret_5556.elem_0;
    memblock_alloc(&out_mem_5473, out_memsize_5474);
    if (out_memsize_5474 > 0)
        OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                           main_ret_5556.elem_1.mem, CL_TRUE, 0,
                                           out_memsize_5474, out_mem_5473.mem +
                                           0, 0, NULL, NULL));
    out_memsize_5476 = main_ret_5556.elem_2;
    memblock_alloc(&out_mem_5475, out_memsize_5476);
    if (out_memsize_5476 > 0)
        OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                           main_ret_5556.elem_3.mem, CL_TRUE, 0,
                                           out_memsize_5476, out_mem_5475.mem +
                                           0, 0, NULL, NULL));
    if (m_4676 == 0)
        printf("empty(%s)", "[f32]");
    else {
        int print_i_5572;
        
        putchar('[');
        for (print_i_5572 = 0; print_i_5572 < m_4676; print_i_5572++) {
            float *print_elem_5573 = (float *) out_mem_5473.mem + print_i_5572 *
                  n_4678;
            
            if (n_4678 == 0)
                printf("empty(%s)", "f32");
            else {
                int print_i_5574;
                
                putchar('[');
                for (print_i_5574 = 0; print_i_5574 < n_4678; print_i_5574++) {
                    float *print_elem_5575 = (float *) print_elem_5573 +
                          print_i_5574 * 1;
                    
                    printf("%.6ff32", *print_elem_5575);
                    if (print_i_5574 != n_4678 - 1)
                        printf(", ");
                }
                putchar(']');
            }
            if (print_i_5572 != m_4676 - 1)
                printf(", ");
        }
        putchar(']');
    }
    printf("\n");
    if (m_4676 == 0)
        printf("empty(%s)", "[f32]");
    else {
        int print_i_5576;
        
        putchar('[');
        for (print_i_5576 = 0; print_i_5576 < m_4676; print_i_5576++) {
            float *print_elem_5577 = (float *) out_mem_5475.mem + print_i_5576 *
                  n_4678;
            
            if (n_4678 == 0)
                printf("empty(%s)", "f32");
            else {
                int print_i_5578;
                
                putchar('[');
                for (print_i_5578 = 0; print_i_5578 < n_4678; print_i_5578++) {
                    float *print_elem_5579 = (float *) print_elem_5577 +
                          print_i_5578 * 1;
                    
                    printf("%.6ff32", *print_elem_5579);
                    if (print_i_5578 != n_4678 - 1)
                        printf(", ");
                }
                putchar(']');
            }
            if (print_i_5576 != m_4676 - 1)
                printf(", ");
        }
        putchar(']');
    }
    printf("\n");
    
    int total_runtime = 0;
    int total_runs = 0;
    
    if (cl_debug) {
        fprintf(stderr,
                "Kernel map_kernel_5053 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_5053runs, (long) map_kernel_5053total_runtime /
                (map_kernel_5053runs != 0 ? map_kernel_5053runs : 1),
                (long) map_kernel_5053total_runtime);
        total_runtime += map_kernel_5053total_runtime;
        total_runs += map_kernel_5053runs;
        fprintf(stderr,
                "Kernel map_kernel_5098 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_5098runs, (long) map_kernel_5098total_runtime /
                (map_kernel_5098runs != 0 ? map_kernel_5098runs : 1),
                (long) map_kernel_5098total_runtime);
        total_runtime += map_kernel_5098total_runtime;
        total_runs += map_kernel_5098runs;
        fprintf(stderr,
                "Kernel map_kernel_5122 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_5122runs, (long) map_kernel_5122total_runtime /
                (map_kernel_5122runs != 0 ? map_kernel_5122runs : 1),
                (long) map_kernel_5122total_runtime);
        total_runtime += map_kernel_5122total_runtime;
        total_runs += map_kernel_5122runs;
        fprintf(stderr,
                "Kernel map_kernel_5167 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_5167runs, (long) map_kernel_5167total_runtime /
                (map_kernel_5167runs != 0 ? map_kernel_5167runs : 1),
                (long) map_kernel_5167total_runtime);
        total_runtime += map_kernel_5167total_runtime;
        total_runs += map_kernel_5167runs;
        fprintf(stderr,
                "Kernel map_kernel_5191 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_5191runs, (long) map_kernel_5191total_runtime /
                (map_kernel_5191runs != 0 ? map_kernel_5191runs : 1),
                (long) map_kernel_5191total_runtime);
        total_runtime += map_kernel_5191total_runtime;
        total_runs += map_kernel_5191runs;
        fprintf(stderr,
                "Kernel map_kernel_5236 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_5236runs, (long) map_kernel_5236total_runtime /
                (map_kernel_5236runs != 0 ? map_kernel_5236runs : 1),
                (long) map_kernel_5236total_runtime);
        total_runtime += map_kernel_5236total_runtime;
        total_runs += map_kernel_5236runs;
        fprintf(stderr,
                "Kernel map_kernel_5260 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_5260runs, (long) map_kernel_5260total_runtime /
                (map_kernel_5260runs != 0 ? map_kernel_5260runs : 1),
                (long) map_kernel_5260total_runtime);
        total_runtime += map_kernel_5260total_runtime;
        total_runs += map_kernel_5260runs;
        fprintf(stderr,
                "Kernel map_kernel_5305 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_5305runs, (long) map_kernel_5305total_runtime /
                (map_kernel_5305runs != 0 ? map_kernel_5305runs : 1),
                (long) map_kernel_5305total_runtime);
        total_runtime += map_kernel_5305total_runtime;
        total_runs += map_kernel_5305runs;
        fprintf(stderr,
                "Kernel map_kernel_5349 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_5349runs, (long) map_kernel_5349total_runtime /
                (map_kernel_5349runs != 0 ? map_kernel_5349runs : 1),
                (long) map_kernel_5349total_runtime);
        total_runtime += map_kernel_5349total_runtime;
        total_runs += map_kernel_5349runs;
        fprintf(stderr,
                "Kernel map_kernel_5325 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_5325runs, (long) map_kernel_5325total_runtime /
                (map_kernel_5325runs != 0 ? map_kernel_5325runs : 1),
                (long) map_kernel_5325total_runtime);
        total_runtime += map_kernel_5325total_runtime;
        total_runs += map_kernel_5325runs;
    }
    if (cl_debug)
        fprintf(stderr, "Ran %d kernels with cumulative runtime: %6ldus\n",
                total_runs, total_runtime);
    memblock_unref_device(&main_ret_5556.elem_1);
    memblock_unref_device(&main_ret_5556.elem_3);
    if (runtime_file != NULL)
        fclose(runtime_file);
    return 0;
}
