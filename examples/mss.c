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
__kernel void map_kernel_2324(__global unsigned char *xs_mem_2584,
                              int32_t size_634, __global
                              unsigned char *mem_2586, __global
                              unsigned char *mem_2588, __global
                              unsigned char *mem_2590)
{
    const uint kernel_thread_index_2324 = get_global_id(0);
    
    if (kernel_thread_index_2324 >= size_634)
        return;
    
    int32_t i_2325;
    int32_t not_curried_2326;
    
    // compute thread index
    {
        i_2325 = kernel_thread_index_2324;
    }
    // read kernel parameters
    {
        not_curried_2326 = *(__global int32_t *) &xs_mem_2584[i_2325 * 4];
    }
    
    char cond_2327 = slt32(not_curried_2326, 0);
    int32_t res_2328;
    
    if (cond_2327) {
        res_2328 = 0;
    } else {
        res_2328 = not_curried_2326;
    }
    // write kernel result
    {
        *(__global int32_t *) &mem_2586[i_2325 * 4] = res_2328;
        *(__global int32_t *) &mem_2588[i_2325 * 4] = res_2328;
        *(__global int32_t *) &mem_2590[i_2325 * 4] = res_2328;
    }
}
__kernel void fut_kernel_map_transpose_i32(__global int32_t *odata,
                                           uint odata_offset, __global
                                           int32_t *idata, uint idata_offset,
                                           uint width, uint height,
                                           uint total_size, __local
                                           int32_t *block)
{
    uint x_index;
    uint y_index;
    uint our_array_offset;
    
    // Adjust the input and output arrays with the basic offset.
    odata += odata_offset / sizeof(int32_t);
    idata += idata_offset / sizeof(int32_t);
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
__kernel void scan_kernel_2335(__local volatile
                               int32_t *restrict not_curried_mem_local_aligned_0,
                               __local volatile
                               int32_t *restrict not_curried_mem_local_aligned_1,
                               __local volatile
                               int32_t *restrict not_curried_mem_local_aligned_2,
                               __local volatile
                               int32_t *restrict not_curried_mem_local_aligned_3,
                               __global unsigned char *mem_2612, __global
                               unsigned char *mem_2597, int32_t size_634,
                               int32_t per_thread_elements_2334,
                               int32_t group_size_2330, __global
                               unsigned char *mem_2602, __global
                               unsigned char *mem_2607,
                               int32_t num_threads_2331, __global
                               unsigned char *mem_2614, __global
                               unsigned char *mem_2616, __global
                               unsigned char *mem_2618, __global
                               unsigned char *mem_2620, __global
                               unsigned char *mem_2623, __global
                               unsigned char *mem_2626, __global
                               unsigned char *mem_2629, __global
                               unsigned char *mem_2632)
{
    __local volatile char *restrict not_curried_mem_local_2735 =
                          not_curried_mem_local_aligned_0;
    __local volatile char *restrict not_curried_mem_local_2737 =
                          not_curried_mem_local_aligned_1;
    __local volatile char *restrict not_curried_mem_local_2739 =
                          not_curried_mem_local_aligned_2;
    __local volatile char *restrict not_curried_mem_local_2741 =
                          not_curried_mem_local_aligned_3;
    int32_t local_id_2708;
    int32_t group_id_2709;
    int32_t wave_size_2710;
    int32_t thread_chunk_size_2712;
    int32_t skip_waves_2711;
    int32_t my_index_2335;
    int32_t other_index_2336;
    int32_t not_curried_2058;
    int32_t not_curried_2059;
    int32_t not_curried_2060;
    int32_t not_curried_2061;
    int32_t not_curried_2062;
    int32_t not_curried_2063;
    int32_t not_curried_2064;
    int32_t not_curried_2065;
    int32_t my_index_2713;
    int32_t other_index_2714;
    int32_t not_curried_2715;
    int32_t not_curried_2716;
    int32_t not_curried_2717;
    int32_t not_curried_2718;
    int32_t not_curried_2719;
    int32_t not_curried_2720;
    int32_t not_curried_2721;
    int32_t not_curried_2722;
    int32_t my_index_2337;
    int32_t other_index_2338;
    int32_t not_curried_2339;
    int32_t not_curried_2340;
    int32_t not_curried_2341;
    int32_t not_curried_2342;
    int32_t not_curried_2343;
    int32_t not_curried_2344;
    int32_t not_curried_2345;
    int32_t not_curried_2346;
    
    local_id_2708 = get_local_id(0);
    group_id_2709 = get_group_id(0);
    skip_waves_2711 = get_global_id(0);
    wave_size_2710 = LOCKSTEP_WIDTH;
    my_index_2337 = skip_waves_2711 * per_thread_elements_2334;
    
    int32_t starting_point_2744 = skip_waves_2711 * per_thread_elements_2334;
    int32_t remaining_elements_2745 = size_634 - starting_point_2744;
    
    if (sle32(remaining_elements_2745, 0) || sle32(size_634,
                                                   starting_point_2744)) {
        thread_chunk_size_2712 = 0;
    } else {
        if (slt32(size_634, (skip_waves_2711 + 1) * per_thread_elements_2334)) {
            thread_chunk_size_2712 = size_634 - skip_waves_2711 *
                per_thread_elements_2334;
        } else {
            thread_chunk_size_2712 = per_thread_elements_2334;
        }
    }
    not_curried_2339 = 0;
    not_curried_2340 = 0;
    not_curried_2341 = 0;
    not_curried_2342 = 0;
    // sequentially scan a chunk
    {
        for (int elements_scanned_2743 = 0; elements_scanned_2743 <
             thread_chunk_size_2712; elements_scanned_2743++) {
            not_curried_2343 = *(__global
                                 int32_t *) &mem_2597[(elements_scanned_2743 *
                                                       num_threads_2331 +
                                                       skip_waves_2711) * 4];
            not_curried_2344 = *(__global
                                 int32_t *) &mem_2602[(elements_scanned_2743 *
                                                       num_threads_2331 +
                                                       skip_waves_2711) * 4];
            not_curried_2345 = *(__global
                                 int32_t *) &mem_2607[(elements_scanned_2743 *
                                                       num_threads_2331 +
                                                       skip_waves_2711) * 4];
            not_curried_2346 = *(__global
                                 int32_t *) &mem_2612[(elements_scanned_2743 *
                                                       num_threads_2331 +
                                                       skip_waves_2711) * 4];
            
            int32_t res_2347 = not_curried_2339 + not_curried_2343;
            int32_t arg_2348 = not_curried_2343 + not_curried_2340;
            char cond_2349 = slt32(arg_2348, not_curried_2344);
            int32_t res_2350;
            
            if (cond_2349) {
                res_2350 = not_curried_2344;
            } else {
                res_2350 = arg_2348;
            }
            
            int32_t arg_2351 = not_curried_2345 + not_curried_2339;
            char cond_2352 = slt32(arg_2351, not_curried_2341);
            int32_t res_2353;
            
            if (cond_2352) {
                res_2353 = not_curried_2341;
            } else {
                res_2353 = arg_2351;
            }
            
            int32_t arg_2354 = not_curried_2340 + not_curried_2345;
            char cond_2355 = slt32(arg_2354, not_curried_2342);
            int32_t res_2356;
            
            if (cond_2355) {
                res_2356 = not_curried_2342;
            } else {
                res_2356 = arg_2354;
            }
            
            char cond_2357 = slt32(res_2356, not_curried_2346);
            int32_t res_2358;
            
            if (cond_2357) {
                res_2358 = not_curried_2346;
            } else {
                res_2358 = res_2356;
            }
            not_curried_2339 = res_2347;
            not_curried_2340 = res_2350;
            not_curried_2341 = res_2353;
            not_curried_2342 = res_2358;
            *(__global int32_t *) &mem_2614[(elements_scanned_2743 *
                                             num_threads_2331 +
                                             skip_waves_2711) * 4] =
                not_curried_2339;
            *(__global int32_t *) &mem_2616[(elements_scanned_2743 *
                                             num_threads_2331 +
                                             skip_waves_2711) * 4] =
                not_curried_2340;
            *(__global int32_t *) &mem_2618[(elements_scanned_2743 *
                                             num_threads_2331 +
                                             skip_waves_2711) * 4] =
                not_curried_2341;
            *(__global int32_t *) &mem_2620[(elements_scanned_2743 *
                                             num_threads_2331 +
                                             skip_waves_2711) * 4] =
                not_curried_2342;
            my_index_2337 += 1;
        }
    }
    *(__local volatile int32_t *) &not_curried_mem_local_2735[local_id_2708 *
                                                              sizeof(int32_t)] =
        not_curried_2339;
    *(__local volatile int32_t *) &not_curried_mem_local_2737[local_id_2708 *
                                                              sizeof(int32_t)] =
        not_curried_2340;
    *(__local volatile int32_t *) &not_curried_mem_local_2739[local_id_2708 *
                                                              sizeof(int32_t)] =
        not_curried_2341;
    *(__local volatile int32_t *) &not_curried_mem_local_2741[local_id_2708 *
                                                              sizeof(int32_t)] =
        not_curried_2342;
    not_curried_2062 = *(__local volatile
                         int32_t *) &not_curried_mem_local_2735[local_id_2708 *
                                                                sizeof(int32_t)];
    not_curried_2063 = *(__local volatile
                         int32_t *) &not_curried_mem_local_2737[local_id_2708 *
                                                                sizeof(int32_t)];
    not_curried_2064 = *(__local volatile
                         int32_t *) &not_curried_mem_local_2739[local_id_2708 *
                                                                sizeof(int32_t)];
    not_curried_2065 = *(__local volatile
                         int32_t *) &not_curried_mem_local_2741[local_id_2708 *
                                                                sizeof(int32_t)];
    // in-wave scan (no barriers needed)
    {
        int32_t skip_threads_2746 = 1;
        
        while (slt32(skip_threads_2746, wave_size_2710)) {
            if (sle32(skip_threads_2746, local_id_2708 - squot32(local_id_2708,
                                                                 wave_size_2710) *
                      wave_size_2710)) {
                // read operands
                {
                    not_curried_2058 = *(__local volatile
                                         int32_t *) &not_curried_mem_local_2735[(local_id_2708 -
                                                                                 skip_threads_2746) *
                                                                                sizeof(int32_t)];
                    not_curried_2059 = *(__local volatile
                                         int32_t *) &not_curried_mem_local_2737[(local_id_2708 -
                                                                                 skip_threads_2746) *
                                                                                sizeof(int32_t)];
                    not_curried_2060 = *(__local volatile
                                         int32_t *) &not_curried_mem_local_2739[(local_id_2708 -
                                                                                 skip_threads_2746) *
                                                                                sizeof(int32_t)];
                    not_curried_2061 = *(__local volatile
                                         int32_t *) &not_curried_mem_local_2741[(local_id_2708 -
                                                                                 skip_threads_2746) *
                                                                                sizeof(int32_t)];
                }
                // perform operation
                {
                    int32_t res_2074 = not_curried_2058 + not_curried_2062;
                    int32_t arg_2075 = not_curried_2062 + not_curried_2059;
                    char cond_2078 = slt32(arg_2075, not_curried_2063);
                    int32_t res_2079;
                    
                    if (cond_2078) {
                        res_2079 = not_curried_2063;
                    } else {
                        res_2079 = arg_2075;
                    }
                    
                    int32_t arg_2081 = not_curried_2064 + not_curried_2058;
                    char cond_2084 = slt32(arg_2081, not_curried_2060);
                    int32_t res_2085;
                    
                    if (cond_2084) {
                        res_2085 = not_curried_2060;
                    } else {
                        res_2085 = arg_2081;
                    }
                    
                    int32_t arg_2087 = not_curried_2059 + not_curried_2064;
                    char cond_2090 = slt32(arg_2087, not_curried_2061);
                    int32_t res_2091;
                    
                    if (cond_2090) {
                        res_2091 = not_curried_2061;
                    } else {
                        res_2091 = arg_2087;
                    }
                    
                    char cond_2095 = slt32(res_2091, not_curried_2065);
                    int32_t res_2096;
                    
                    if (cond_2095) {
                        res_2096 = not_curried_2065;
                    } else {
                        res_2096 = res_2091;
                    }
                    not_curried_2062 = res_2074;
                    not_curried_2063 = res_2079;
                    not_curried_2064 = res_2085;
                    not_curried_2065 = res_2096;
                }
                // write result
                {
                    *(__local volatile
                      int32_t *) &not_curried_mem_local_2735[local_id_2708 *
                                                             sizeof(int32_t)] =
                        not_curried_2062;
                    *(__local volatile
                      int32_t *) &not_curried_mem_local_2737[local_id_2708 *
                                                             sizeof(int32_t)] =
                        not_curried_2063;
                    *(__local volatile
                      int32_t *) &not_curried_mem_local_2739[local_id_2708 *
                                                             sizeof(int32_t)] =
                        not_curried_2064;
                    *(__local volatile
                      int32_t *) &not_curried_mem_local_2741[local_id_2708 *
                                                             sizeof(int32_t)] =
                        not_curried_2065;
                }
            }
            skip_threads_2746 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of wave 'i' writes its result to offset 'i'
    {
        if ((local_id_2708 - squot32(local_id_2708, wave_size_2710) *
             wave_size_2710) == wave_size_2710 - 1) {
            *(__local volatile
              int32_t *) &not_curried_mem_local_2735[squot32(local_id_2708,
                                                             wave_size_2710) *
                                                     sizeof(int32_t)] =
                not_curried_2062;
            *(__local volatile
              int32_t *) &not_curried_mem_local_2737[squot32(local_id_2708,
                                                             wave_size_2710) *
                                                     sizeof(int32_t)] =
                not_curried_2063;
            *(__local volatile
              int32_t *) &not_curried_mem_local_2739[squot32(local_id_2708,
                                                             wave_size_2710) *
                                                     sizeof(int32_t)] =
                not_curried_2064;
            *(__local volatile
              int32_t *) &not_curried_mem_local_2741[squot32(local_id_2708,
                                                             wave_size_2710) *
                                                     sizeof(int32_t)] =
                not_curried_2065;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first wave, after which offset 'i' contains carry-in for warp 'i+1'
    {
        if (squot32(local_id_2708, wave_size_2710) == 0) {
            not_curried_2719 = *(__local volatile
                                 int32_t *) &not_curried_mem_local_2735[local_id_2708 *
                                                                        sizeof(int32_t)];
            not_curried_2720 = *(__local volatile
                                 int32_t *) &not_curried_mem_local_2737[local_id_2708 *
                                                                        sizeof(int32_t)];
            not_curried_2721 = *(__local volatile
                                 int32_t *) &not_curried_mem_local_2739[local_id_2708 *
                                                                        sizeof(int32_t)];
            not_curried_2722 = *(__local volatile
                                 int32_t *) &not_curried_mem_local_2741[local_id_2708 *
                                                                        sizeof(int32_t)];
            // in-wave scan (no barriers needed)
            {
                int32_t skip_threads_2747 = 1;
                
                while (slt32(skip_threads_2747, wave_size_2710)) {
                    if (sle32(skip_threads_2747, local_id_2708 -
                              squot32(local_id_2708, wave_size_2710) *
                              wave_size_2710)) {
                        // read operands
                        {
                            not_curried_2715 = *(__local volatile
                                                 int32_t *) &not_curried_mem_local_2735[(local_id_2708 -
                                                                                         skip_threads_2747) *
                                                                                        sizeof(int32_t)];
                            not_curried_2716 = *(__local volatile
                                                 int32_t *) &not_curried_mem_local_2737[(local_id_2708 -
                                                                                         skip_threads_2747) *
                                                                                        sizeof(int32_t)];
                            not_curried_2717 = *(__local volatile
                                                 int32_t *) &not_curried_mem_local_2739[(local_id_2708 -
                                                                                         skip_threads_2747) *
                                                                                        sizeof(int32_t)];
                            not_curried_2718 = *(__local volatile
                                                 int32_t *) &not_curried_mem_local_2741[(local_id_2708 -
                                                                                         skip_threads_2747) *
                                                                                        sizeof(int32_t)];
                        }
                        // perform operation
                        {
                            int32_t res_2723 = not_curried_2715 +
                                    not_curried_2719;
                            int32_t arg_2724 = not_curried_2719 +
                                    not_curried_2716;
                            char cond_2725 = slt32(arg_2724, not_curried_2720);
                            int32_t res_2726;
                            
                            if (cond_2725) {
                                res_2726 = not_curried_2720;
                            } else {
                                res_2726 = arg_2724;
                            }
                            
                            int32_t arg_2727 = not_curried_2721 +
                                    not_curried_2715;
                            char cond_2728 = slt32(arg_2727, not_curried_2717);
                            int32_t res_2729;
                            
                            if (cond_2728) {
                                res_2729 = not_curried_2717;
                            } else {
                                res_2729 = arg_2727;
                            }
                            
                            int32_t arg_2730 = not_curried_2716 +
                                    not_curried_2721;
                            char cond_2731 = slt32(arg_2730, not_curried_2718);
                            int32_t res_2732;
                            
                            if (cond_2731) {
                                res_2732 = not_curried_2718;
                            } else {
                                res_2732 = arg_2730;
                            }
                            
                            char cond_2733 = slt32(res_2732, not_curried_2722);
                            int32_t res_2734;
                            
                            if (cond_2733) {
                                res_2734 = not_curried_2722;
                            } else {
                                res_2734 = res_2732;
                            }
                            not_curried_2719 = res_2723;
                            not_curried_2720 = res_2726;
                            not_curried_2721 = res_2729;
                            not_curried_2722 = res_2734;
                        }
                        // write result
                        {
                            *(__local volatile
                              int32_t *) &not_curried_mem_local_2735[local_id_2708 *
                                                                     sizeof(int32_t)] =
                                not_curried_2719;
                            *(__local volatile
                              int32_t *) &not_curried_mem_local_2737[local_id_2708 *
                                                                     sizeof(int32_t)] =
                                not_curried_2720;
                            *(__local volatile
                              int32_t *) &not_curried_mem_local_2739[local_id_2708 *
                                                                     sizeof(int32_t)] =
                                not_curried_2721;
                            *(__local volatile
                              int32_t *) &not_curried_mem_local_2741[local_id_2708 *
                                                                     sizeof(int32_t)] =
                                not_curried_2722;
                        }
                    }
                    skip_threads_2747 *= 2;
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every wave except the first
    {
        if (!(squot32(local_id_2708, wave_size_2710) == 0)) {
            // read operands
            {
                not_curried_2058 = *(__local volatile
                                     int32_t *) &not_curried_mem_local_2735[(squot32(local_id_2708,
                                                                                     wave_size_2710) -
                                                                             1) *
                                                                            sizeof(int32_t)];
                not_curried_2059 = *(__local volatile
                                     int32_t *) &not_curried_mem_local_2737[(squot32(local_id_2708,
                                                                                     wave_size_2710) -
                                                                             1) *
                                                                            sizeof(int32_t)];
                not_curried_2060 = *(__local volatile
                                     int32_t *) &not_curried_mem_local_2739[(squot32(local_id_2708,
                                                                                     wave_size_2710) -
                                                                             1) *
                                                                            sizeof(int32_t)];
                not_curried_2061 = *(__local volatile
                                     int32_t *) &not_curried_mem_local_2741[(squot32(local_id_2708,
                                                                                     wave_size_2710) -
                                                                             1) *
                                                                            sizeof(int32_t)];
            }
            // perform operation
            {
                int32_t res_2074 = not_curried_2058 + not_curried_2062;
                int32_t arg_2075 = not_curried_2062 + not_curried_2059;
                char cond_2078 = slt32(arg_2075, not_curried_2063);
                int32_t res_2079;
                
                if (cond_2078) {
                    res_2079 = not_curried_2063;
                } else {
                    res_2079 = arg_2075;
                }
                
                int32_t arg_2081 = not_curried_2064 + not_curried_2058;
                char cond_2084 = slt32(arg_2081, not_curried_2060);
                int32_t res_2085;
                
                if (cond_2084) {
                    res_2085 = not_curried_2060;
                } else {
                    res_2085 = arg_2081;
                }
                
                int32_t arg_2087 = not_curried_2059 + not_curried_2064;
                char cond_2090 = slt32(arg_2087, not_curried_2061);
                int32_t res_2091;
                
                if (cond_2090) {
                    res_2091 = not_curried_2061;
                } else {
                    res_2091 = arg_2087;
                }
                
                char cond_2095 = slt32(res_2091, not_curried_2065);
                int32_t res_2096;
                
                if (cond_2095) {
                    res_2096 = not_curried_2065;
                } else {
                    res_2096 = res_2091;
                }
                not_curried_2062 = res_2074;
                not_curried_2063 = res_2079;
                not_curried_2064 = res_2085;
                not_curried_2065 = res_2096;
            }
        }
    }
    *(__global int32_t *) &mem_2623[(group_id_2709 * group_size_2330 +
                                     local_id_2708) * 4] = not_curried_2062;
    *(__global int32_t *) &mem_2626[(group_id_2709 * group_size_2330 +
                                     local_id_2708) * 4] = not_curried_2063;
    *(__global int32_t *) &mem_2629[(group_id_2709 * group_size_2330 +
                                     local_id_2708) * 4] = not_curried_2064;
    *(__global int32_t *) &mem_2632[(group_id_2709 * group_size_2330 +
                                     local_id_2708) * 4] = not_curried_2065;
}
__kernel void map_kernel_2367(__global unsigned char *mem_2632,
                              int32_t num_groups_2329, __global
                              unsigned char *mem_2629,
                              int32_t last_in_group_index_2369, __global
                              unsigned char *mem_2626, int32_t group_size_2330,
                              __global unsigned char *mem_2623, __global
                              unsigned char *mem_2646, __global
                              unsigned char *mem_2648, __global
                              unsigned char *mem_2650, __global
                              unsigned char *mem_2652)
{
    const uint lasts_map_index_2367 = get_global_id(0);
    
    if (lasts_map_index_2367 >= num_groups_2329)
        return;
    
    int32_t group_id_2368;
    
    // compute thread index
    {
        group_id_2368 = lasts_map_index_2367;
    }
    // read kernel parameters
    { }
    
    char cond_2375 = slt32(0, group_id_2368);
    int32_t preceding_group_2370 = group_id_2368 - 1;
    int32_t group_lasts_2380;
    int32_t group_lasts_2381;
    int32_t group_lasts_2382;
    int32_t group_lasts_2383;
    
    if (cond_2375) {
        int32_t x_2371 = *(__global int32_t *) &mem_2623[(preceding_group_2370 *
                                                          group_size_2330 +
                                                          last_in_group_index_2369) *
                                                         4];
        int32_t x_2372 = *(__global int32_t *) &mem_2626[(preceding_group_2370 *
                                                          group_size_2330 +
                                                          last_in_group_index_2369) *
                                                         4];
        int32_t x_2373 = *(__global int32_t *) &mem_2629[(preceding_group_2370 *
                                                          group_size_2330 +
                                                          last_in_group_index_2369) *
                                                         4];
        int32_t x_2374 = *(__global int32_t *) &mem_2632[(preceding_group_2370 *
                                                          group_size_2330 +
                                                          last_in_group_index_2369) *
                                                         4];
        
        group_lasts_2380 = x_2371;
        group_lasts_2381 = x_2372;
        group_lasts_2382 = x_2373;
        group_lasts_2383 = x_2374;
    } else {
        group_lasts_2380 = 0;
        group_lasts_2381 = 0;
        group_lasts_2382 = 0;
        group_lasts_2383 = 0;
    }
    // write kernel result
    {
        *(__global int32_t *) &mem_2646[group_id_2368 * 4] = group_lasts_2380;
        *(__global int32_t *) &mem_2648[group_id_2368 * 4] = group_lasts_2381;
        *(__global int32_t *) &mem_2650[group_id_2368 * 4] = group_lasts_2382;
        *(__global int32_t *) &mem_2652[group_id_2368 * 4] = group_lasts_2383;
    }
}
__kernel void scan_kernel_2388(__local volatile
                               int32_t *restrict not_curried_mem_local_aligned_0,
                               __local volatile
                               int32_t *restrict not_curried_mem_local_aligned_1,
                               __local volatile
                               int32_t *restrict not_curried_mem_local_aligned_2,
                               __local volatile
                               int32_t *restrict not_curried_mem_local_aligned_3,
                               __global unsigned char *mem_2648, __global
                               unsigned char *mem_2652, int32_t num_groups_2329,
                               __global unsigned char *mem_2646, __global
                               unsigned char *mem_2650, __global
                               unsigned char *mem_2654, __global
                               unsigned char *mem_2656, __global
                               unsigned char *mem_2658, __global
                               unsigned char *mem_2660, __global
                               unsigned char *mem_2663, __global
                               unsigned char *mem_2666, __global
                               unsigned char *mem_2669, __global
                               unsigned char *mem_2672)
{
    __local volatile char *restrict not_curried_mem_local_2777 =
                          not_curried_mem_local_aligned_0;
    __local volatile char *restrict not_curried_mem_local_2779 =
                          not_curried_mem_local_aligned_1;
    __local volatile char *restrict not_curried_mem_local_2781 =
                          not_curried_mem_local_aligned_2;
    __local volatile char *restrict not_curried_mem_local_2783 =
                          not_curried_mem_local_aligned_3;
    int32_t local_id_2750;
    int32_t group_id_2751;
    int32_t wave_size_2752;
    int32_t thread_chunk_size_2754;
    int32_t skip_waves_2753;
    int32_t my_index_2388;
    int32_t other_index_2389;
    int32_t not_curried_2390;
    int32_t not_curried_2391;
    int32_t not_curried_2392;
    int32_t not_curried_2393;
    int32_t not_curried_2394;
    int32_t not_curried_2395;
    int32_t not_curried_2396;
    int32_t not_curried_2397;
    int32_t my_index_2755;
    int32_t other_index_2756;
    int32_t not_curried_2757;
    int32_t not_curried_2758;
    int32_t not_curried_2759;
    int32_t not_curried_2760;
    int32_t not_curried_2761;
    int32_t not_curried_2762;
    int32_t not_curried_2763;
    int32_t not_curried_2764;
    int32_t my_index_2410;
    int32_t other_index_2411;
    int32_t not_curried_2412;
    int32_t not_curried_2413;
    int32_t not_curried_2414;
    int32_t not_curried_2415;
    int32_t not_curried_2416;
    int32_t not_curried_2417;
    int32_t not_curried_2418;
    int32_t not_curried_2419;
    
    local_id_2750 = get_local_id(0);
    group_id_2751 = get_group_id(0);
    skip_waves_2753 = get_global_id(0);
    wave_size_2752 = LOCKSTEP_WIDTH;
    my_index_2410 = skip_waves_2753;
    
    int32_t starting_point_2786 = skip_waves_2753;
    int32_t remaining_elements_2787 = num_groups_2329 - starting_point_2786;
    
    if (sle32(remaining_elements_2787, 0) || sle32(num_groups_2329,
                                                   starting_point_2786)) {
        thread_chunk_size_2754 = 0;
    } else {
        if (slt32(num_groups_2329, skip_waves_2753 + 1)) {
            thread_chunk_size_2754 = num_groups_2329 - skip_waves_2753;
        } else {
            thread_chunk_size_2754 = 1;
        }
    }
    not_curried_2412 = 0;
    not_curried_2413 = 0;
    not_curried_2414 = 0;
    not_curried_2415 = 0;
    // sequentially scan a chunk
    {
        for (int elements_scanned_2785 = 0; elements_scanned_2785 <
             thread_chunk_size_2754; elements_scanned_2785++) {
            not_curried_2416 = *(__global
                                 int32_t *) &mem_2646[(skip_waves_2753 +
                                                       elements_scanned_2785) *
                                                      4];
            not_curried_2417 = *(__global
                                 int32_t *) &mem_2648[(skip_waves_2753 +
                                                       elements_scanned_2785) *
                                                      4];
            not_curried_2418 = *(__global
                                 int32_t *) &mem_2650[(skip_waves_2753 +
                                                       elements_scanned_2785) *
                                                      4];
            not_curried_2419 = *(__global
                                 int32_t *) &mem_2652[(skip_waves_2753 +
                                                       elements_scanned_2785) *
                                                      4];
            
            int32_t res_2420 = not_curried_2412 + not_curried_2416;
            int32_t arg_2421 = not_curried_2416 + not_curried_2413;
            char cond_2422 = slt32(arg_2421, not_curried_2417);
            int32_t res_2423;
            
            if (cond_2422) {
                res_2423 = not_curried_2417;
            } else {
                res_2423 = arg_2421;
            }
            
            int32_t arg_2424 = not_curried_2418 + not_curried_2412;
            char cond_2425 = slt32(arg_2424, not_curried_2414);
            int32_t res_2426;
            
            if (cond_2425) {
                res_2426 = not_curried_2414;
            } else {
                res_2426 = arg_2424;
            }
            
            int32_t arg_2427 = not_curried_2413 + not_curried_2418;
            char cond_2428 = slt32(arg_2427, not_curried_2415);
            int32_t res_2429;
            
            if (cond_2428) {
                res_2429 = not_curried_2415;
            } else {
                res_2429 = arg_2427;
            }
            
            char cond_2430 = slt32(res_2429, not_curried_2419);
            int32_t res_2431;
            
            if (cond_2430) {
                res_2431 = not_curried_2419;
            } else {
                res_2431 = res_2429;
            }
            not_curried_2412 = res_2420;
            not_curried_2413 = res_2423;
            not_curried_2414 = res_2426;
            not_curried_2415 = res_2431;
            *(__global int32_t *) &mem_2654[(skip_waves_2753 +
                                             elements_scanned_2785) * 4] =
                not_curried_2412;
            *(__global int32_t *) &mem_2656[(skip_waves_2753 +
                                             elements_scanned_2785) * 4] =
                not_curried_2413;
            *(__global int32_t *) &mem_2658[(skip_waves_2753 +
                                             elements_scanned_2785) * 4] =
                not_curried_2414;
            *(__global int32_t *) &mem_2660[(skip_waves_2753 +
                                             elements_scanned_2785) * 4] =
                not_curried_2415;
            my_index_2410 += 1;
        }
    }
    *(__local volatile int32_t *) &not_curried_mem_local_2777[local_id_2750 *
                                                              sizeof(int32_t)] =
        not_curried_2412;
    *(__local volatile int32_t *) &not_curried_mem_local_2779[local_id_2750 *
                                                              sizeof(int32_t)] =
        not_curried_2413;
    *(__local volatile int32_t *) &not_curried_mem_local_2781[local_id_2750 *
                                                              sizeof(int32_t)] =
        not_curried_2414;
    *(__local volatile int32_t *) &not_curried_mem_local_2783[local_id_2750 *
                                                              sizeof(int32_t)] =
        not_curried_2415;
    not_curried_2394 = *(__local volatile
                         int32_t *) &not_curried_mem_local_2777[local_id_2750 *
                                                                sizeof(int32_t)];
    not_curried_2395 = *(__local volatile
                         int32_t *) &not_curried_mem_local_2779[local_id_2750 *
                                                                sizeof(int32_t)];
    not_curried_2396 = *(__local volatile
                         int32_t *) &not_curried_mem_local_2781[local_id_2750 *
                                                                sizeof(int32_t)];
    not_curried_2397 = *(__local volatile
                         int32_t *) &not_curried_mem_local_2783[local_id_2750 *
                                                                sizeof(int32_t)];
    // in-wave scan (no barriers needed)
    {
        int32_t skip_threads_2788 = 1;
        
        while (slt32(skip_threads_2788, wave_size_2752)) {
            if (sle32(skip_threads_2788, local_id_2750 - squot32(local_id_2750,
                                                                 wave_size_2752) *
                      wave_size_2752)) {
                // read operands
                {
                    not_curried_2390 = *(__local volatile
                                         int32_t *) &not_curried_mem_local_2777[(local_id_2750 -
                                                                                 skip_threads_2788) *
                                                                                sizeof(int32_t)];
                    not_curried_2391 = *(__local volatile
                                         int32_t *) &not_curried_mem_local_2779[(local_id_2750 -
                                                                                 skip_threads_2788) *
                                                                                sizeof(int32_t)];
                    not_curried_2392 = *(__local volatile
                                         int32_t *) &not_curried_mem_local_2781[(local_id_2750 -
                                                                                 skip_threads_2788) *
                                                                                sizeof(int32_t)];
                    not_curried_2393 = *(__local volatile
                                         int32_t *) &not_curried_mem_local_2783[(local_id_2750 -
                                                                                 skip_threads_2788) *
                                                                                sizeof(int32_t)];
                }
                // perform operation
                {
                    int32_t res_2398 = not_curried_2390 + not_curried_2394;
                    int32_t arg_2399 = not_curried_2394 + not_curried_2391;
                    char cond_2400 = slt32(arg_2399, not_curried_2395);
                    int32_t res_2401;
                    
                    if (cond_2400) {
                        res_2401 = not_curried_2395;
                    } else {
                        res_2401 = arg_2399;
                    }
                    
                    int32_t arg_2402 = not_curried_2396 + not_curried_2390;
                    char cond_2403 = slt32(arg_2402, not_curried_2392);
                    int32_t res_2404;
                    
                    if (cond_2403) {
                        res_2404 = not_curried_2392;
                    } else {
                        res_2404 = arg_2402;
                    }
                    
                    int32_t arg_2405 = not_curried_2391 + not_curried_2396;
                    char cond_2406 = slt32(arg_2405, not_curried_2393);
                    int32_t res_2407;
                    
                    if (cond_2406) {
                        res_2407 = not_curried_2393;
                    } else {
                        res_2407 = arg_2405;
                    }
                    
                    char cond_2408 = slt32(res_2407, not_curried_2397);
                    int32_t res_2409;
                    
                    if (cond_2408) {
                        res_2409 = not_curried_2397;
                    } else {
                        res_2409 = res_2407;
                    }
                    not_curried_2394 = res_2398;
                    not_curried_2395 = res_2401;
                    not_curried_2396 = res_2404;
                    not_curried_2397 = res_2409;
                }
                // write result
                {
                    *(__local volatile
                      int32_t *) &not_curried_mem_local_2777[local_id_2750 *
                                                             sizeof(int32_t)] =
                        not_curried_2394;
                    *(__local volatile
                      int32_t *) &not_curried_mem_local_2779[local_id_2750 *
                                                             sizeof(int32_t)] =
                        not_curried_2395;
                    *(__local volatile
                      int32_t *) &not_curried_mem_local_2781[local_id_2750 *
                                                             sizeof(int32_t)] =
                        not_curried_2396;
                    *(__local volatile
                      int32_t *) &not_curried_mem_local_2783[local_id_2750 *
                                                             sizeof(int32_t)] =
                        not_curried_2397;
                }
            }
            skip_threads_2788 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of wave 'i' writes its result to offset 'i'
    {
        if ((local_id_2750 - squot32(local_id_2750, wave_size_2752) *
             wave_size_2752) == wave_size_2752 - 1) {
            *(__local volatile
              int32_t *) &not_curried_mem_local_2777[squot32(local_id_2750,
                                                             wave_size_2752) *
                                                     sizeof(int32_t)] =
                not_curried_2394;
            *(__local volatile
              int32_t *) &not_curried_mem_local_2779[squot32(local_id_2750,
                                                             wave_size_2752) *
                                                     sizeof(int32_t)] =
                not_curried_2395;
            *(__local volatile
              int32_t *) &not_curried_mem_local_2781[squot32(local_id_2750,
                                                             wave_size_2752) *
                                                     sizeof(int32_t)] =
                not_curried_2396;
            *(__local volatile
              int32_t *) &not_curried_mem_local_2783[squot32(local_id_2750,
                                                             wave_size_2752) *
                                                     sizeof(int32_t)] =
                not_curried_2397;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first wave, after which offset 'i' contains carry-in for warp 'i+1'
    {
        if (squot32(local_id_2750, wave_size_2752) == 0) {
            not_curried_2761 = *(__local volatile
                                 int32_t *) &not_curried_mem_local_2777[local_id_2750 *
                                                                        sizeof(int32_t)];
            not_curried_2762 = *(__local volatile
                                 int32_t *) &not_curried_mem_local_2779[local_id_2750 *
                                                                        sizeof(int32_t)];
            not_curried_2763 = *(__local volatile
                                 int32_t *) &not_curried_mem_local_2781[local_id_2750 *
                                                                        sizeof(int32_t)];
            not_curried_2764 = *(__local volatile
                                 int32_t *) &not_curried_mem_local_2783[local_id_2750 *
                                                                        sizeof(int32_t)];
            // in-wave scan (no barriers needed)
            {
                int32_t skip_threads_2789 = 1;
                
                while (slt32(skip_threads_2789, wave_size_2752)) {
                    if (sle32(skip_threads_2789, local_id_2750 -
                              squot32(local_id_2750, wave_size_2752) *
                              wave_size_2752)) {
                        // read operands
                        {
                            not_curried_2757 = *(__local volatile
                                                 int32_t *) &not_curried_mem_local_2777[(local_id_2750 -
                                                                                         skip_threads_2789) *
                                                                                        sizeof(int32_t)];
                            not_curried_2758 = *(__local volatile
                                                 int32_t *) &not_curried_mem_local_2779[(local_id_2750 -
                                                                                         skip_threads_2789) *
                                                                                        sizeof(int32_t)];
                            not_curried_2759 = *(__local volatile
                                                 int32_t *) &not_curried_mem_local_2781[(local_id_2750 -
                                                                                         skip_threads_2789) *
                                                                                        sizeof(int32_t)];
                            not_curried_2760 = *(__local volatile
                                                 int32_t *) &not_curried_mem_local_2783[(local_id_2750 -
                                                                                         skip_threads_2789) *
                                                                                        sizeof(int32_t)];
                        }
                        // perform operation
                        {
                            int32_t res_2765 = not_curried_2757 +
                                    not_curried_2761;
                            int32_t arg_2766 = not_curried_2761 +
                                    not_curried_2758;
                            char cond_2767 = slt32(arg_2766, not_curried_2762);
                            int32_t res_2768;
                            
                            if (cond_2767) {
                                res_2768 = not_curried_2762;
                            } else {
                                res_2768 = arg_2766;
                            }
                            
                            int32_t arg_2769 = not_curried_2763 +
                                    not_curried_2757;
                            char cond_2770 = slt32(arg_2769, not_curried_2759);
                            int32_t res_2771;
                            
                            if (cond_2770) {
                                res_2771 = not_curried_2759;
                            } else {
                                res_2771 = arg_2769;
                            }
                            
                            int32_t arg_2772 = not_curried_2758 +
                                    not_curried_2763;
                            char cond_2773 = slt32(arg_2772, not_curried_2760);
                            int32_t res_2774;
                            
                            if (cond_2773) {
                                res_2774 = not_curried_2760;
                            } else {
                                res_2774 = arg_2772;
                            }
                            
                            char cond_2775 = slt32(res_2774, not_curried_2764);
                            int32_t res_2776;
                            
                            if (cond_2775) {
                                res_2776 = not_curried_2764;
                            } else {
                                res_2776 = res_2774;
                            }
                            not_curried_2761 = res_2765;
                            not_curried_2762 = res_2768;
                            not_curried_2763 = res_2771;
                            not_curried_2764 = res_2776;
                        }
                        // write result
                        {
                            *(__local volatile
                              int32_t *) &not_curried_mem_local_2777[local_id_2750 *
                                                                     sizeof(int32_t)] =
                                not_curried_2761;
                            *(__local volatile
                              int32_t *) &not_curried_mem_local_2779[local_id_2750 *
                                                                     sizeof(int32_t)] =
                                not_curried_2762;
                            *(__local volatile
                              int32_t *) &not_curried_mem_local_2781[local_id_2750 *
                                                                     sizeof(int32_t)] =
                                not_curried_2763;
                            *(__local volatile
                              int32_t *) &not_curried_mem_local_2783[local_id_2750 *
                                                                     sizeof(int32_t)] =
                                not_curried_2764;
                        }
                    }
                    skip_threads_2789 *= 2;
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every wave except the first
    {
        if (!(squot32(local_id_2750, wave_size_2752) == 0)) {
            // read operands
            {
                not_curried_2390 = *(__local volatile
                                     int32_t *) &not_curried_mem_local_2777[(squot32(local_id_2750,
                                                                                     wave_size_2752) -
                                                                             1) *
                                                                            sizeof(int32_t)];
                not_curried_2391 = *(__local volatile
                                     int32_t *) &not_curried_mem_local_2779[(squot32(local_id_2750,
                                                                                     wave_size_2752) -
                                                                             1) *
                                                                            sizeof(int32_t)];
                not_curried_2392 = *(__local volatile
                                     int32_t *) &not_curried_mem_local_2781[(squot32(local_id_2750,
                                                                                     wave_size_2752) -
                                                                             1) *
                                                                            sizeof(int32_t)];
                not_curried_2393 = *(__local volatile
                                     int32_t *) &not_curried_mem_local_2783[(squot32(local_id_2750,
                                                                                     wave_size_2752) -
                                                                             1) *
                                                                            sizeof(int32_t)];
            }
            // perform operation
            {
                int32_t res_2398 = not_curried_2390 + not_curried_2394;
                int32_t arg_2399 = not_curried_2394 + not_curried_2391;
                char cond_2400 = slt32(arg_2399, not_curried_2395);
                int32_t res_2401;
                
                if (cond_2400) {
                    res_2401 = not_curried_2395;
                } else {
                    res_2401 = arg_2399;
                }
                
                int32_t arg_2402 = not_curried_2396 + not_curried_2390;
                char cond_2403 = slt32(arg_2402, not_curried_2392);
                int32_t res_2404;
                
                if (cond_2403) {
                    res_2404 = not_curried_2392;
                } else {
                    res_2404 = arg_2402;
                }
                
                int32_t arg_2405 = not_curried_2391 + not_curried_2396;
                char cond_2406 = slt32(arg_2405, not_curried_2393);
                int32_t res_2407;
                
                if (cond_2406) {
                    res_2407 = not_curried_2393;
                } else {
                    res_2407 = arg_2405;
                }
                
                char cond_2408 = slt32(res_2407, not_curried_2397);
                int32_t res_2409;
                
                if (cond_2408) {
                    res_2409 = not_curried_2397;
                } else {
                    res_2409 = res_2407;
                }
                not_curried_2394 = res_2398;
                not_curried_2395 = res_2401;
                not_curried_2396 = res_2404;
                not_curried_2397 = res_2409;
            }
        }
    }
    *(__global int32_t *) &mem_2663[(group_id_2751 * num_groups_2329 +
                                     local_id_2750) * 4] = not_curried_2394;
    *(__global int32_t *) &mem_2666[(group_id_2751 * num_groups_2329 +
                                     local_id_2750) * 4] = not_curried_2395;
    *(__global int32_t *) &mem_2669[(group_id_2751 * num_groups_2329 +
                                     local_id_2750) * 4] = not_curried_2396;
    *(__global int32_t *) &mem_2672[(group_id_2751 * num_groups_2329 +
                                     local_id_2750) * 4] = not_curried_2397;
}
__kernel void map_kernel_2464(__global unsigned char *mem_2632, __global
                              unsigned char *mem_2672, int32_t num_groups_2329,
                              __global unsigned char *mem_2629, __global
                              unsigned char *mem_2669, __global
                              unsigned char *mem_2626, int32_t group_size_2330,
                              __global unsigned char *mem_2666, __global
                              unsigned char *mem_2623, __global
                              unsigned char *mem_2663, __global
                              unsigned char *mem_2675, __global
                              unsigned char *mem_2678, __global
                              unsigned char *mem_2681, __global
                              unsigned char *mem_2684)
{
    const uint chunk_carry_out_index_2464 = get_global_id(0);
    
    if (chunk_carry_out_index_2464 >= num_groups_2329 * group_size_2330)
        return;
    
    int32_t group_id_2465;
    int32_t elem_id_2466;
    int32_t not_curried_2444;
    int32_t not_curried_2445;
    int32_t not_curried_2446;
    int32_t not_curried_2447;
    int32_t not_curried_2448;
    int32_t not_curried_2449;
    int32_t not_curried_2450;
    int32_t not_curried_2451;
    
    // compute thread index
    {
        group_id_2465 = squot32(chunk_carry_out_index_2464, group_size_2330);
        elem_id_2466 = chunk_carry_out_index_2464 -
            squot32(chunk_carry_out_index_2464, group_size_2330) *
            group_size_2330;
    }
    // read kernel parameters
    {
        not_curried_2444 = *(__global int32_t *) &mem_2663[group_id_2465 * 4];
        not_curried_2445 = *(__global int32_t *) &mem_2666[group_id_2465 * 4];
        not_curried_2446 = *(__global int32_t *) &mem_2669[group_id_2465 * 4];
        not_curried_2447 = *(__global int32_t *) &mem_2672[group_id_2465 * 4];
        not_curried_2448 = *(__global int32_t *) &mem_2623[(group_id_2465 *
                                                            group_size_2330 +
                                                            elem_id_2466) * 4];
        not_curried_2449 = *(__global int32_t *) &mem_2626[(group_id_2465 *
                                                            group_size_2330 +
                                                            elem_id_2466) * 4];
        not_curried_2450 = *(__global int32_t *) &mem_2629[(group_id_2465 *
                                                            group_size_2330 +
                                                            elem_id_2466) * 4];
        not_curried_2451 = *(__global int32_t *) &mem_2632[(group_id_2465 *
                                                            group_size_2330 +
                                                            elem_id_2466) * 4];
    }
    
    int32_t res_2452 = not_curried_2444 + not_curried_2448;
    int32_t arg_2453 = not_curried_2448 + not_curried_2445;
    char cond_2454 = slt32(arg_2453, not_curried_2449);
    int32_t res_2455;
    
    if (cond_2454) {
        res_2455 = not_curried_2449;
    } else {
        res_2455 = arg_2453;
    }
    
    int32_t arg_2456 = not_curried_2450 + not_curried_2444;
    char cond_2457 = slt32(arg_2456, not_curried_2446);
    int32_t res_2458;
    
    if (cond_2457) {
        res_2458 = not_curried_2446;
    } else {
        res_2458 = arg_2456;
    }
    
    int32_t arg_2459 = not_curried_2445 + not_curried_2450;
    char cond_2460 = slt32(arg_2459, not_curried_2447);
    int32_t res_2461;
    
    if (cond_2460) {
        res_2461 = not_curried_2447;
    } else {
        res_2461 = arg_2459;
    }
    
    char cond_2462 = slt32(res_2461, not_curried_2451);
    int32_t res_2463;
    
    if (cond_2462) {
        res_2463 = not_curried_2451;
    } else {
        res_2463 = res_2461;
    }
    // write kernel result
    {
        *(__global int32_t *) &mem_2675[(group_id_2465 * group_size_2330 +
                                         elem_id_2466) * 4] = res_2452;
        *(__global int32_t *) &mem_2678[(group_id_2465 * group_size_2330 +
                                         elem_id_2466) * 4] = res_2455;
        *(__global int32_t *) &mem_2681[(group_id_2465 * group_size_2330 +
                                         elem_id_2466) * 4] = res_2458;
        *(__global int32_t *) &mem_2684[(group_id_2465 * group_size_2330 +
                                         elem_id_2466) * 4] = res_2463;
    }
}
__kernel void map_kernel_2495(__global unsigned char *mem_2684, __global
                              unsigned char *mem_2644, __global
                              unsigned char *mem_2681, __global
                              unsigned char *mem_2641, int32_t size_634,
                              int32_t per_thread_elements_2334, __global
                              unsigned char *mem_2678, int32_t group_size_2330,
                              __global unsigned char *mem_2638, __global
                              unsigned char *mem_2635, __global
                              unsigned char *mem_2675, __global
                              unsigned char *mem_2686, __global
                              unsigned char *mem_2688, __global
                              unsigned char *mem_2690, __global
                              unsigned char *mem_2692)
{
    const uint result_map_index_2495 = get_global_id(0);
    
    if (result_map_index_2495 >= size_634)
        return;
    
    int32_t j_2496;
    int32_t not_curried_2479;
    int32_t not_curried_2480;
    int32_t not_curried_2481;
    int32_t not_curried_2482;
    
    // compute thread index
    {
        j_2496 = result_map_index_2495;
    }
    // read kernel parameters
    {
        not_curried_2479 = *(__global int32_t *) &mem_2635[(squot32(j_2496,
                                                                    per_thread_elements_2334) *
                                                            per_thread_elements_2334 +
                                                            (j_2496 -
                                                             squot32(j_2496,
                                                                     per_thread_elements_2334) *
                                                             per_thread_elements_2334)) *
                                                           4];
        not_curried_2480 = *(__global int32_t *) &mem_2638[(squot32(j_2496,
                                                                    per_thread_elements_2334) *
                                                            per_thread_elements_2334 +
                                                            (j_2496 -
                                                             squot32(j_2496,
                                                                     per_thread_elements_2334) *
                                                             per_thread_elements_2334)) *
                                                           4];
        not_curried_2481 = *(__global int32_t *) &mem_2641[(squot32(j_2496,
                                                                    per_thread_elements_2334) *
                                                            per_thread_elements_2334 +
                                                            (j_2496 -
                                                             squot32(j_2496,
                                                                     per_thread_elements_2334) *
                                                             per_thread_elements_2334)) *
                                                           4];
        not_curried_2482 = *(__global int32_t *) &mem_2644[(squot32(j_2496,
                                                                    per_thread_elements_2334) *
                                                            per_thread_elements_2334 +
                                                            (j_2496 -
                                                             squot32(j_2496,
                                                                     per_thread_elements_2334) *
                                                             per_thread_elements_2334)) *
                                                           4];
    }
    
    int32_t thread_id_2497 = squot32(j_2496, per_thread_elements_2334);
    char cond_2498 = 0 == thread_id_2497;
    int32_t carry_in_index_2499 = thread_id_2497 - 1;
    int32_t new_index_2507 = squot32(carry_in_index_2499, group_size_2330);
    int32_t y_2509 = new_index_2507 * group_size_2330;
    int32_t x_2510 = carry_in_index_2499 - y_2509;
    int32_t final_result_2503;
    int32_t final_result_2504;
    int32_t final_result_2505;
    int32_t final_result_2506;
    
    if (cond_2498) {
        final_result_2503 = not_curried_2479;
        final_result_2504 = not_curried_2480;
        final_result_2505 = not_curried_2481;
        final_result_2506 = not_curried_2482;
    } else {
        int32_t not_curried_2475 = *(__global
                                     int32_t *) &mem_2675[(new_index_2507 *
                                                           group_size_2330 +
                                                           x_2510) * 4];
        int32_t not_curried_2476 = *(__global
                                     int32_t *) &mem_2678[(new_index_2507 *
                                                           group_size_2330 +
                                                           x_2510) * 4];
        int32_t not_curried_2477 = *(__global
                                     int32_t *) &mem_2681[(new_index_2507 *
                                                           group_size_2330 +
                                                           x_2510) * 4];
        int32_t not_curried_2478 = *(__global
                                     int32_t *) &mem_2684[(new_index_2507 *
                                                           group_size_2330 +
                                                           x_2510) * 4];
        int32_t res_2483 = not_curried_2475 + not_curried_2479;
        int32_t arg_2484 = not_curried_2479 + not_curried_2476;
        char cond_2485 = slt32(arg_2484, not_curried_2480);
        int32_t res_2486;
        
        if (cond_2485) {
            res_2486 = not_curried_2480;
        } else {
            res_2486 = arg_2484;
        }
        
        int32_t arg_2487 = not_curried_2481 + not_curried_2475;
        char cond_2488 = slt32(arg_2487, not_curried_2477);
        int32_t res_2489;
        
        if (cond_2488) {
            res_2489 = not_curried_2477;
        } else {
            res_2489 = arg_2487;
        }
        
        int32_t arg_2490 = not_curried_2476 + not_curried_2481;
        char cond_2491 = slt32(arg_2490, not_curried_2478);
        int32_t res_2492;
        
        if (cond_2491) {
            res_2492 = not_curried_2478;
        } else {
            res_2492 = arg_2490;
        }
        
        char cond_2493 = slt32(res_2492, not_curried_2482);
        int32_t res_2494;
        
        if (cond_2493) {
            res_2494 = not_curried_2482;
        } else {
            res_2494 = res_2492;
        }
        final_result_2503 = res_2483;
        final_result_2504 = res_2486;
        final_result_2505 = res_2489;
        final_result_2506 = res_2494;
    }
    // write kernel result
    {
        *(__global int32_t *) &mem_2686[j_2496 * 4] = final_result_2503;
        *(__global int32_t *) &mem_2688[j_2496 * 4] = final_result_2504;
        *(__global int32_t *) &mem_2690[j_2496 * 4] = final_result_2505;
        *(__global int32_t *) &mem_2692[j_2496 * 4] = final_result_2506;
    }
}
);
static cl_kernel map_kernel_2324;
static int map_kernel_2324total_runtime = 0;
static int map_kernel_2324runs = 0;
static cl_kernel fut_kernel_map_transpose_i32;
static int fut_kernel_map_transpose_i32total_runtime = 0;
static int fut_kernel_map_transpose_i32runs = 0;
static cl_kernel scan_kernel_2335;
static int scan_kernel_2335total_runtime = 0;
static int scan_kernel_2335runs = 0;
static cl_kernel map_kernel_2367;
static int map_kernel_2367total_runtime = 0;
static int map_kernel_2367runs = 0;
static cl_kernel scan_kernel_2388;
static int scan_kernel_2388total_runtime = 0;
static int scan_kernel_2388runs = 0;
static cl_kernel map_kernel_2464;
static int map_kernel_2464total_runtime = 0;
static int map_kernel_2464runs = 0;
static cl_kernel map_kernel_2495;
static int map_kernel_2495total_runtime = 0;
static int map_kernel_2495runs = 0;
void setup_opencl_and_load_kernels()

{
    cl_int error;
    cl_program prog = setup_opencl(fut_opencl_prelude, fut_opencl_program);
    
    {
        map_kernel_2324 = clCreateKernel(prog, "map_kernel_2324", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_2324");
    }
    {
        fut_kernel_map_transpose_i32 = clCreateKernel(prog,
                                                      "fut_kernel_map_transpose_i32",
                                                      &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n",
                    "fut_kernel_map_transpose_i32");
    }
    {
        scan_kernel_2335 = clCreateKernel(prog, "scan_kernel_2335", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "scan_kernel_2335");
    }
    {
        map_kernel_2367 = clCreateKernel(prog, "map_kernel_2367", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_2367");
    }
    {
        scan_kernel_2388 = clCreateKernel(prog, "scan_kernel_2388", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "scan_kernel_2388");
    }
    {
        map_kernel_2464 = clCreateKernel(prog, "map_kernel_2464", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_2464");
    }
    {
        map_kernel_2495 = clCreateKernel(prog, "map_kernel_2495", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_2495");
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
    
    cl_int clCreateBuffer_succeeded_2899;
    
    block->mem = clCreateBuffer(fut_cl_context, CL_MEM_READ_WRITE, size >
                                0 ? size : 1, NULL,
                                &clCreateBuffer_succeeded_2899);
    OPENCL_SUCCEED(clCreateBuffer_succeeded_2899);
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
struct tuple_int32_t_device_mem_int32_t_int32_t_device_mem_int32_t_device_mem_int32_t_device_mem {
    int32_t elem_0;
    struct memblock_device elem_1;
    int32_t elem_2;
    int32_t elem_3;
    struct memblock_device elem_4;
    int32_t elem_5;
    struct memblock_device elem_6;
    int32_t elem_7;
    struct memblock_device elem_8;
} ;
static
struct tuple_int32_t_device_mem_int32_t_int32_t_device_mem_int32_t_device_mem_int32_t_device_mem
futhark_main(int32_t xs_mem_size_2583, struct memblock_device xs_mem_2584, int32_t size_634);
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
struct tuple_int32_t_device_mem_int32_t_int32_t_device_mem_int32_t_device_mem_int32_t_device_mem futhark_main(int32_t xs_mem_size_2583,
                                                                                                              struct memblock_device xs_mem_2584,
                                                                                                              int32_t size_634)
{
    int32_t out_memsize_2694;
    struct memblock_device out_mem_2693;
    
    out_mem_2693.references = NULL;
    
    int32_t out_arrsize_2695;
    int32_t out_memsize_2697;
    struct memblock_device out_mem_2696;
    
    out_mem_2696.references = NULL;
    
    int32_t out_memsize_2699;
    struct memblock_device out_mem_2698;
    
    out_mem_2698.references = NULL;
    
    int32_t out_memsize_2701;
    struct memblock_device out_mem_2700;
    
    out_mem_2700.references = NULL;
    
    int32_t bytes_2585 = 4 * size_634;
    struct memblock_device mem_2586;
    
    mem_2586.references = NULL;
    memblock_alloc_device(&mem_2586, bytes_2585);
    
    struct memblock_device mem_2588;
    
    mem_2588.references = NULL;
    memblock_alloc_device(&mem_2588, bytes_2585);
    
    struct memblock_device mem_2590;
    
    mem_2590.references = NULL;
    memblock_alloc_device(&mem_2590, bytes_2585);
    
    int32_t group_size_2702;
    int32_t num_groups_2703;
    
    group_size_2702 = cl_group_size;
    num_groups_2703 = squot32(size_634 + group_size_2702 - 1, group_size_2702);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2324, 0, sizeof(xs_mem_2584.mem),
                                  &xs_mem_2584.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2324, 1, sizeof(size_634),
                                  &size_634));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2324, 2, sizeof(mem_2586.mem),
                                  &mem_2586.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2324, 3, sizeof(mem_2588.mem),
                                  &mem_2588.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2324, 4, sizeof(mem_2590.mem),
                                  &mem_2590.mem));
    if (1 * (num_groups_2703 * group_size_2702) != 0) {
        const size_t global_work_size_2795[1] = {num_groups_2703 *
                     group_size_2702};
        const size_t local_work_size_2799[1] = {group_size_2702};
        int64_t time_start_2796, time_end_2797;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_2324");
            fprintf(stderr, "%zu", global_work_size_2795[0]);
            fprintf(stderr, "].\n");
            time_start_2796 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_2324, 1,
                                              NULL, global_work_size_2795,
                                              local_work_size_2799, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_2797 = get_wall_time();
            
            long time_diff_2798 = time_end_2797 - time_start_2796;
            
            if (detail_timing) {
                map_kernel_2324total_runtime += time_diff_2798;
                map_kernel_2324runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_2324",
                        (int) time_diff_2798);
            }
        }
    }
    
    int32_t num_groups_2329;
    
    num_groups_2329 = cl_num_groups;
    
    int32_t group_size_2330;
    
    group_size_2330 = cl_group_size;
    
    int32_t num_threads_2331 = num_groups_2329 * group_size_2330;
    int32_t y_2332 = num_threads_2331 - 1;
    int32_t x_2333 = size_634 + y_2332;
    int32_t per_thread_elements_2334 = squot32(x_2333, num_threads_2331);
    int32_t y_2527 = smod32(size_634, num_threads_2331);
    int32_t x_2528 = num_threads_2331 - y_2527;
    int32_t y_2529 = smod32(x_2528, num_threads_2331);
    int32_t padded_size_2530 = size_634 + y_2529;
    int32_t padding_2531 = padded_size_2530 - size_634;
    int32_t x_2533 = padded_size_2530 + y_2332;
    int32_t offset_multiple_2534 = squot32(x_2533, num_threads_2331);
    int32_t bytes_2591 = 4 * padding_2531;
    struct memblock_device mem_2592;
    
    mem_2592.references = NULL;
    memblock_alloc_device(&mem_2592, bytes_2591);
    
    int32_t bytes_2593 = 4 * padded_size_2530;
    struct memblock_device mem_2594;
    
    mem_2594.references = NULL;
    memblock_alloc_device(&mem_2594, bytes_2593);
    
    int32_t tmp_offs_2704 = 0;
    
    if (size_634 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(fut_cl_queue, xs_mem_2584.mem,
                                           mem_2594.mem, 0, tmp_offs_2704 * 4,
                                           size_634 * sizeof(int32_t), 0, NULL,
                                           NULL));
        if (cl_debug)
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
    }
    tmp_offs_2704 += size_634;
    if (padding_2531 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(fut_cl_queue, mem_2592.mem,
                                           mem_2594.mem, 0, tmp_offs_2704 * 4,
                                           padding_2531 * sizeof(int32_t), 0,
                                           NULL, NULL));
        if (cl_debug)
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
    }
    tmp_offs_2704 += padding_2531;
    
    int32_t x_2596 = 4 * per_thread_elements_2334;
    int32_t bytes_2595 = x_2596 * num_threads_2331;
    struct memblock_device mem_2597;
    
    mem_2597.references = NULL;
    memblock_alloc_device(&mem_2597, bytes_2595);
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 0,
                                  sizeof(mem_2597.mem), &mem_2597.mem));
    
    int32_t kernel_arg_2800 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 1,
                                  sizeof(kernel_arg_2800), &kernel_arg_2800));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 2,
                                  sizeof(mem_2594.mem), &mem_2594.mem));
    
    int32_t kernel_arg_2801 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 3,
                                  sizeof(kernel_arg_2801), &kernel_arg_2801));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 4,
                                  sizeof(per_thread_elements_2334),
                                  &per_thread_elements_2334));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 5,
                                  sizeof(num_threads_2331), &num_threads_2331));
    
    int32_t kernel_arg_2802 = per_thread_elements_2334 * num_threads_2331;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 6,
                                  sizeof(kernel_arg_2802), &kernel_arg_2802));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 7, (16 + 1) *
                                  16 * sizeof(int32_t), NULL));
    if (1 * (per_thread_elements_2334 + srem32(16 -
                                               srem32(per_thread_elements_2334,
                                                      16), 16)) *
        (num_threads_2331 + srem32(16 - srem32(num_threads_2331, 16), 16)) *
        1 != 0) {
        const size_t global_work_size_2803[3] = {per_thread_elements_2334 +
                                                 srem32(16 -
                                                        srem32(per_thread_elements_2334,
                                                               16), 16),
                                                 num_threads_2331 + srem32(16 -
                                                                           srem32(num_threads_2331,
                                                                                  16),
                                                                           16),
                                                 1};
        const size_t local_work_size_2807[3] = {16, 16, 1};
        int64_t time_start_2804, time_end_2805;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "fut_kernel_map_transpose_i32");
            fprintf(stderr, "%zu", global_work_size_2803[0]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_2803[1]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_2803[2]);
            fprintf(stderr, "].\n");
            time_start_2804 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue,
                                              fut_kernel_map_transpose_i32, 3,
                                              NULL, global_work_size_2803,
                                              local_work_size_2807, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_2805 = get_wall_time();
            
            long time_diff_2806 = time_end_2805 - time_start_2804;
            
            if (detail_timing) {
                fut_kernel_map_transpose_i32total_runtime += time_diff_2806;
                fut_kernel_map_transpose_i32runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "fut_kernel_map_transpose_i32", (int) time_diff_2806);
            }
        }
    }
    
    struct memblock_device mem_2599;
    
    mem_2599.references = NULL;
    memblock_alloc_device(&mem_2599, bytes_2593);
    
    int32_t tmp_offs_2705 = 0;
    
    if (size_634 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(fut_cl_queue, mem_2586.mem,
                                           mem_2599.mem, 0, tmp_offs_2705 * 4,
                                           size_634 * sizeof(int32_t), 0, NULL,
                                           NULL));
        if (cl_debug)
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
    }
    tmp_offs_2705 += size_634;
    if (padding_2531 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(fut_cl_queue, mem_2592.mem,
                                           mem_2599.mem, 0, tmp_offs_2705 * 4,
                                           padding_2531 * sizeof(int32_t), 0,
                                           NULL, NULL));
        if (cl_debug)
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
    }
    tmp_offs_2705 += padding_2531;
    
    struct memblock_device mem_2602;
    
    mem_2602.references = NULL;
    memblock_alloc_device(&mem_2602, bytes_2595);
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 0,
                                  sizeof(mem_2602.mem), &mem_2602.mem));
    
    int32_t kernel_arg_2808 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 1,
                                  sizeof(kernel_arg_2808), &kernel_arg_2808));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 2,
                                  sizeof(mem_2599.mem), &mem_2599.mem));
    
    int32_t kernel_arg_2809 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 3,
                                  sizeof(kernel_arg_2809), &kernel_arg_2809));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 4,
                                  sizeof(per_thread_elements_2334),
                                  &per_thread_elements_2334));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 5,
                                  sizeof(num_threads_2331), &num_threads_2331));
    
    int32_t kernel_arg_2810 = per_thread_elements_2334 * num_threads_2331;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 6,
                                  sizeof(kernel_arg_2810), &kernel_arg_2810));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 7, (16 + 1) *
                                  16 * sizeof(int32_t), NULL));
    if (1 * (per_thread_elements_2334 + srem32(16 -
                                               srem32(per_thread_elements_2334,
                                                      16), 16)) *
        (num_threads_2331 + srem32(16 - srem32(num_threads_2331, 16), 16)) *
        1 != 0) {
        const size_t global_work_size_2811[3] = {per_thread_elements_2334 +
                                                 srem32(16 -
                                                        srem32(per_thread_elements_2334,
                                                               16), 16),
                                                 num_threads_2331 + srem32(16 -
                                                                           srem32(num_threads_2331,
                                                                                  16),
                                                                           16),
                                                 1};
        const size_t local_work_size_2815[3] = {16, 16, 1};
        int64_t time_start_2812, time_end_2813;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "fut_kernel_map_transpose_i32");
            fprintf(stderr, "%zu", global_work_size_2811[0]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_2811[1]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_2811[2]);
            fprintf(stderr, "].\n");
            time_start_2812 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue,
                                              fut_kernel_map_transpose_i32, 3,
                                              NULL, global_work_size_2811,
                                              local_work_size_2815, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_2813 = get_wall_time();
            
            long time_diff_2814 = time_end_2813 - time_start_2812;
            
            if (detail_timing) {
                fut_kernel_map_transpose_i32total_runtime += time_diff_2814;
                fut_kernel_map_transpose_i32runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "fut_kernel_map_transpose_i32", (int) time_diff_2814);
            }
        }
    }
    
    struct memblock_device mem_2604;
    
    mem_2604.references = NULL;
    memblock_alloc_device(&mem_2604, bytes_2593);
    
    int32_t tmp_offs_2706 = 0;
    
    if (size_634 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(fut_cl_queue, mem_2588.mem,
                                           mem_2604.mem, 0, tmp_offs_2706 * 4,
                                           size_634 * sizeof(int32_t), 0, NULL,
                                           NULL));
        if (cl_debug)
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
    }
    tmp_offs_2706 += size_634;
    if (padding_2531 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(fut_cl_queue, mem_2592.mem,
                                           mem_2604.mem, 0, tmp_offs_2706 * 4,
                                           padding_2531 * sizeof(int32_t), 0,
                                           NULL, NULL));
        if (cl_debug)
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
    }
    tmp_offs_2706 += padding_2531;
    
    struct memblock_device mem_2607;
    
    mem_2607.references = NULL;
    memblock_alloc_device(&mem_2607, bytes_2595);
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 0,
                                  sizeof(mem_2607.mem), &mem_2607.mem));
    
    int32_t kernel_arg_2816 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 1,
                                  sizeof(kernel_arg_2816), &kernel_arg_2816));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 2,
                                  sizeof(mem_2604.mem), &mem_2604.mem));
    
    int32_t kernel_arg_2817 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 3,
                                  sizeof(kernel_arg_2817), &kernel_arg_2817));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 4,
                                  sizeof(per_thread_elements_2334),
                                  &per_thread_elements_2334));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 5,
                                  sizeof(num_threads_2331), &num_threads_2331));
    
    int32_t kernel_arg_2818 = per_thread_elements_2334 * num_threads_2331;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 6,
                                  sizeof(kernel_arg_2818), &kernel_arg_2818));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 7, (16 + 1) *
                                  16 * sizeof(int32_t), NULL));
    if (1 * (per_thread_elements_2334 + srem32(16 -
                                               srem32(per_thread_elements_2334,
                                                      16), 16)) *
        (num_threads_2331 + srem32(16 - srem32(num_threads_2331, 16), 16)) *
        1 != 0) {
        const size_t global_work_size_2819[3] = {per_thread_elements_2334 +
                                                 srem32(16 -
                                                        srem32(per_thread_elements_2334,
                                                               16), 16),
                                                 num_threads_2331 + srem32(16 -
                                                                           srem32(num_threads_2331,
                                                                                  16),
                                                                           16),
                                                 1};
        const size_t local_work_size_2823[3] = {16, 16, 1};
        int64_t time_start_2820, time_end_2821;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "fut_kernel_map_transpose_i32");
            fprintf(stderr, "%zu", global_work_size_2819[0]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_2819[1]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_2819[2]);
            fprintf(stderr, "].\n");
            time_start_2820 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue,
                                              fut_kernel_map_transpose_i32, 3,
                                              NULL, global_work_size_2819,
                                              local_work_size_2823, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_2821 = get_wall_time();
            
            long time_diff_2822 = time_end_2821 - time_start_2820;
            
            if (detail_timing) {
                fut_kernel_map_transpose_i32total_runtime += time_diff_2822;
                fut_kernel_map_transpose_i32runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "fut_kernel_map_transpose_i32", (int) time_diff_2822);
            }
        }
    }
    
    struct memblock_device mem_2609;
    
    mem_2609.references = NULL;
    memblock_alloc_device(&mem_2609, bytes_2593);
    
    int32_t tmp_offs_2707 = 0;
    
    if (size_634 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(fut_cl_queue, mem_2590.mem,
                                           mem_2609.mem, 0, tmp_offs_2707 * 4,
                                           size_634 * sizeof(int32_t), 0, NULL,
                                           NULL));
        if (cl_debug)
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
    }
    tmp_offs_2707 += size_634;
    if (padding_2531 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(fut_cl_queue, mem_2592.mem,
                                           mem_2609.mem, 0, tmp_offs_2707 * 4,
                                           padding_2531 * sizeof(int32_t), 0,
                                           NULL, NULL));
        if (cl_debug)
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
    }
    tmp_offs_2707 += padding_2531;
    
    struct memblock_device mem_2612;
    
    mem_2612.references = NULL;
    memblock_alloc_device(&mem_2612, bytes_2595);
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 0,
                                  sizeof(mem_2612.mem), &mem_2612.mem));
    
    int32_t kernel_arg_2824 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 1,
                                  sizeof(kernel_arg_2824), &kernel_arg_2824));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 2,
                                  sizeof(mem_2609.mem), &mem_2609.mem));
    
    int32_t kernel_arg_2825 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 3,
                                  sizeof(kernel_arg_2825), &kernel_arg_2825));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 4,
                                  sizeof(per_thread_elements_2334),
                                  &per_thread_elements_2334));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 5,
                                  sizeof(num_threads_2331), &num_threads_2331));
    
    int32_t kernel_arg_2826 = per_thread_elements_2334 * num_threads_2331;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 6,
                                  sizeof(kernel_arg_2826), &kernel_arg_2826));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 7, (16 + 1) *
                                  16 * sizeof(int32_t), NULL));
    if (1 * (per_thread_elements_2334 + srem32(16 -
                                               srem32(per_thread_elements_2334,
                                                      16), 16)) *
        (num_threads_2331 + srem32(16 - srem32(num_threads_2331, 16), 16)) *
        1 != 0) {
        const size_t global_work_size_2827[3] = {per_thread_elements_2334 +
                                                 srem32(16 -
                                                        srem32(per_thread_elements_2334,
                                                               16), 16),
                                                 num_threads_2331 + srem32(16 -
                                                                           srem32(num_threads_2331,
                                                                                  16),
                                                                           16),
                                                 1};
        const size_t local_work_size_2831[3] = {16, 16, 1};
        int64_t time_start_2828, time_end_2829;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "fut_kernel_map_transpose_i32");
            fprintf(stderr, "%zu", global_work_size_2827[0]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_2827[1]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_2827[2]);
            fprintf(stderr, "].\n");
            time_start_2828 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue,
                                              fut_kernel_map_transpose_i32, 3,
                                              NULL, global_work_size_2827,
                                              local_work_size_2831, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_2829 = get_wall_time();
            
            long time_diff_2830 = time_end_2829 - time_start_2828;
            
            if (detail_timing) {
                fut_kernel_map_transpose_i32total_runtime += time_diff_2830;
                fut_kernel_map_transpose_i32runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "fut_kernel_map_transpose_i32", (int) time_diff_2830);
            }
        }
    }
    
    struct memblock_device mem_2614;
    
    mem_2614.references = NULL;
    memblock_alloc_device(&mem_2614, bytes_2593);
    
    struct memblock_device mem_2616;
    
    mem_2616.references = NULL;
    memblock_alloc_device(&mem_2616, bytes_2593);
    
    struct memblock_device mem_2618;
    
    mem_2618.references = NULL;
    memblock_alloc_device(&mem_2618, bytes_2593);
    
    struct memblock_device mem_2620;
    
    mem_2620.references = NULL;
    memblock_alloc_device(&mem_2620, bytes_2593);
    
    int32_t x_2622 = 4 * num_groups_2329;
    int32_t bytes_2621 = x_2622 * group_size_2330;
    struct memblock_device mem_2623;
    
    mem_2623.references = NULL;
    memblock_alloc_device(&mem_2623, bytes_2621);
    
    struct memblock_device mem_2626;
    
    mem_2626.references = NULL;
    memblock_alloc_device(&mem_2626, bytes_2621);
    
    struct memblock_device mem_2629;
    
    mem_2629.references = NULL;
    memblock_alloc_device(&mem_2629, bytes_2621);
    
    struct memblock_device mem_2632;
    
    mem_2632.references = NULL;
    memblock_alloc_device(&mem_2632, bytes_2621);
    
    int32_t total_size_2736 = sizeof(int32_t) * group_size_2330;
    int32_t total_size_2738 = sizeof(int32_t) * group_size_2330;
    int32_t total_size_2740 = sizeof(int32_t) * group_size_2330;
    int32_t total_size_2742;
    
    total_size_2742 = sizeof(int32_t) * group_size_2330;
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2335, 0, total_size_2736, NULL));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2335, 1, total_size_2738, NULL));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2335, 2, total_size_2740, NULL));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2335, 3, total_size_2742, NULL));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2335, 4, sizeof(mem_2612.mem),
                                  &mem_2612.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2335, 5, sizeof(mem_2597.mem),
                                  &mem_2597.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2335, 6, sizeof(size_634),
                                  &size_634));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2335, 7,
                                  sizeof(per_thread_elements_2334),
                                  &per_thread_elements_2334));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2335, 8, sizeof(group_size_2330),
                                  &group_size_2330));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2335, 9, sizeof(mem_2602.mem),
                                  &mem_2602.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2335, 10, sizeof(mem_2607.mem),
                                  &mem_2607.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2335, 11,
                                  sizeof(num_threads_2331), &num_threads_2331));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2335, 12, sizeof(mem_2614.mem),
                                  &mem_2614.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2335, 13, sizeof(mem_2616.mem),
                                  &mem_2616.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2335, 14, sizeof(mem_2618.mem),
                                  &mem_2618.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2335, 15, sizeof(mem_2620.mem),
                                  &mem_2620.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2335, 16, sizeof(mem_2623.mem),
                                  &mem_2623.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2335, 17, sizeof(mem_2626.mem),
                                  &mem_2626.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2335, 18, sizeof(mem_2629.mem),
                                  &mem_2629.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2335, 19, sizeof(mem_2632.mem),
                                  &mem_2632.mem));
    if (1 * (num_groups_2329 * group_size_2330) != 0) {
        const size_t global_work_size_2832[1] = {num_groups_2329 *
                     group_size_2330};
        const size_t local_work_size_2836[1] = {group_size_2330};
        int64_t time_start_2833, time_end_2834;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "scan_kernel_2335");
            fprintf(stderr, "%zu", global_work_size_2832[0]);
            fprintf(stderr, "].\n");
            time_start_2833 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, scan_kernel_2335, 1,
                                              NULL, global_work_size_2832,
                                              local_work_size_2836, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_2834 = get_wall_time();
            
            long time_diff_2835 = time_end_2834 - time_start_2833;
            
            if (detail_timing) {
                scan_kernel_2335total_runtime += time_diff_2835;
                scan_kernel_2335runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_kernel_2335", (int) time_diff_2835);
            }
        }
    }
    
    int32_t x_2634 = 4 * num_threads_2331;
    int32_t bytes_2633 = x_2634 * per_thread_elements_2334;
    struct memblock_device mem_2635;
    
    mem_2635.references = NULL;
    memblock_alloc_device(&mem_2635, bytes_2633);
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 0,
                                  sizeof(mem_2635.mem), &mem_2635.mem));
    
    int32_t kernel_arg_2837 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 1,
                                  sizeof(kernel_arg_2837), &kernel_arg_2837));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 2,
                                  sizeof(mem_2614.mem), &mem_2614.mem));
    
    int32_t kernel_arg_2838 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 3,
                                  sizeof(kernel_arg_2838), &kernel_arg_2838));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 4,
                                  sizeof(num_threads_2331), &num_threads_2331));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 5,
                                  sizeof(per_thread_elements_2334),
                                  &per_thread_elements_2334));
    
    int32_t kernel_arg_2839 = num_threads_2331 * per_thread_elements_2334;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 6,
                                  sizeof(kernel_arg_2839), &kernel_arg_2839));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 7, (16 + 1) *
                                  16 * sizeof(int32_t), NULL));
    if (1 * (num_threads_2331 + srem32(16 - srem32(num_threads_2331, 16), 16)) *
        (per_thread_elements_2334 + srem32(16 - srem32(per_thread_elements_2334,
                                                       16), 16)) * 1 != 0) {
        const size_t global_work_size_2840[3] = {num_threads_2331 + srem32(16 -
                                                                           srem32(num_threads_2331,
                                                                                  16),
                                                                           16),
                                                 per_thread_elements_2334 +
                                                 srem32(16 -
                                                        srem32(per_thread_elements_2334,
                                                               16), 16), 1};
        const size_t local_work_size_2844[3] = {16, 16, 1};
        int64_t time_start_2841, time_end_2842;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "fut_kernel_map_transpose_i32");
            fprintf(stderr, "%zu", global_work_size_2840[0]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_2840[1]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_2840[2]);
            fprintf(stderr, "].\n");
            time_start_2841 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue,
                                              fut_kernel_map_transpose_i32, 3,
                                              NULL, global_work_size_2840,
                                              local_work_size_2844, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_2842 = get_wall_time();
            
            long time_diff_2843 = time_end_2842 - time_start_2841;
            
            if (detail_timing) {
                fut_kernel_map_transpose_i32total_runtime += time_diff_2843;
                fut_kernel_map_transpose_i32runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "fut_kernel_map_transpose_i32", (int) time_diff_2843);
            }
        }
    }
    
    struct memblock_device mem_2638;
    
    mem_2638.references = NULL;
    memblock_alloc_device(&mem_2638, bytes_2633);
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 0,
                                  sizeof(mem_2638.mem), &mem_2638.mem));
    
    int32_t kernel_arg_2845 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 1,
                                  sizeof(kernel_arg_2845), &kernel_arg_2845));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 2,
                                  sizeof(mem_2616.mem), &mem_2616.mem));
    
    int32_t kernel_arg_2846 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 3,
                                  sizeof(kernel_arg_2846), &kernel_arg_2846));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 4,
                                  sizeof(num_threads_2331), &num_threads_2331));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 5,
                                  sizeof(per_thread_elements_2334),
                                  &per_thread_elements_2334));
    
    int32_t kernel_arg_2847 = num_threads_2331 * per_thread_elements_2334;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 6,
                                  sizeof(kernel_arg_2847), &kernel_arg_2847));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 7, (16 + 1) *
                                  16 * sizeof(int32_t), NULL));
    if (1 * (num_threads_2331 + srem32(16 - srem32(num_threads_2331, 16), 16)) *
        (per_thread_elements_2334 + srem32(16 - srem32(per_thread_elements_2334,
                                                       16), 16)) * 1 != 0) {
        const size_t global_work_size_2848[3] = {num_threads_2331 + srem32(16 -
                                                                           srem32(num_threads_2331,
                                                                                  16),
                                                                           16),
                                                 per_thread_elements_2334 +
                                                 srem32(16 -
                                                        srem32(per_thread_elements_2334,
                                                               16), 16), 1};
        const size_t local_work_size_2852[3] = {16, 16, 1};
        int64_t time_start_2849, time_end_2850;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "fut_kernel_map_transpose_i32");
            fprintf(stderr, "%zu", global_work_size_2848[0]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_2848[1]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_2848[2]);
            fprintf(stderr, "].\n");
            time_start_2849 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue,
                                              fut_kernel_map_transpose_i32, 3,
                                              NULL, global_work_size_2848,
                                              local_work_size_2852, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_2850 = get_wall_time();
            
            long time_diff_2851 = time_end_2850 - time_start_2849;
            
            if (detail_timing) {
                fut_kernel_map_transpose_i32total_runtime += time_diff_2851;
                fut_kernel_map_transpose_i32runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "fut_kernel_map_transpose_i32", (int) time_diff_2851);
            }
        }
    }
    
    struct memblock_device mem_2641;
    
    mem_2641.references = NULL;
    memblock_alloc_device(&mem_2641, bytes_2633);
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 0,
                                  sizeof(mem_2641.mem), &mem_2641.mem));
    
    int32_t kernel_arg_2853 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 1,
                                  sizeof(kernel_arg_2853), &kernel_arg_2853));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 2,
                                  sizeof(mem_2618.mem), &mem_2618.mem));
    
    int32_t kernel_arg_2854 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 3,
                                  sizeof(kernel_arg_2854), &kernel_arg_2854));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 4,
                                  sizeof(num_threads_2331), &num_threads_2331));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 5,
                                  sizeof(per_thread_elements_2334),
                                  &per_thread_elements_2334));
    
    int32_t kernel_arg_2855 = num_threads_2331 * per_thread_elements_2334;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 6,
                                  sizeof(kernel_arg_2855), &kernel_arg_2855));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 7, (16 + 1) *
                                  16 * sizeof(int32_t), NULL));
    if (1 * (num_threads_2331 + srem32(16 - srem32(num_threads_2331, 16), 16)) *
        (per_thread_elements_2334 + srem32(16 - srem32(per_thread_elements_2334,
                                                       16), 16)) * 1 != 0) {
        const size_t global_work_size_2856[3] = {num_threads_2331 + srem32(16 -
                                                                           srem32(num_threads_2331,
                                                                                  16),
                                                                           16),
                                                 per_thread_elements_2334 +
                                                 srem32(16 -
                                                        srem32(per_thread_elements_2334,
                                                               16), 16), 1};
        const size_t local_work_size_2860[3] = {16, 16, 1};
        int64_t time_start_2857, time_end_2858;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "fut_kernel_map_transpose_i32");
            fprintf(stderr, "%zu", global_work_size_2856[0]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_2856[1]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_2856[2]);
            fprintf(stderr, "].\n");
            time_start_2857 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue,
                                              fut_kernel_map_transpose_i32, 3,
                                              NULL, global_work_size_2856,
                                              local_work_size_2860, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_2858 = get_wall_time();
            
            long time_diff_2859 = time_end_2858 - time_start_2857;
            
            if (detail_timing) {
                fut_kernel_map_transpose_i32total_runtime += time_diff_2859;
                fut_kernel_map_transpose_i32runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "fut_kernel_map_transpose_i32", (int) time_diff_2859);
            }
        }
    }
    
    struct memblock_device mem_2644;
    
    mem_2644.references = NULL;
    memblock_alloc_device(&mem_2644, bytes_2633);
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 0,
                                  sizeof(mem_2644.mem), &mem_2644.mem));
    
    int32_t kernel_arg_2861 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 1,
                                  sizeof(kernel_arg_2861), &kernel_arg_2861));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 2,
                                  sizeof(mem_2620.mem), &mem_2620.mem));
    
    int32_t kernel_arg_2862 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 3,
                                  sizeof(kernel_arg_2862), &kernel_arg_2862));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 4,
                                  sizeof(num_threads_2331), &num_threads_2331));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 5,
                                  sizeof(per_thread_elements_2334),
                                  &per_thread_elements_2334));
    
    int32_t kernel_arg_2863 = num_threads_2331 * per_thread_elements_2334;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 6,
                                  sizeof(kernel_arg_2863), &kernel_arg_2863));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 7, (16 + 1) *
                                  16 * sizeof(int32_t), NULL));
    if (1 * (num_threads_2331 + srem32(16 - srem32(num_threads_2331, 16), 16)) *
        (per_thread_elements_2334 + srem32(16 - srem32(per_thread_elements_2334,
                                                       16), 16)) * 1 != 0) {
        const size_t global_work_size_2864[3] = {num_threads_2331 + srem32(16 -
                                                                           srem32(num_threads_2331,
                                                                                  16),
                                                                           16),
                                                 per_thread_elements_2334 +
                                                 srem32(16 -
                                                        srem32(per_thread_elements_2334,
                                                               16), 16), 1};
        const size_t local_work_size_2868[3] = {16, 16, 1};
        int64_t time_start_2865, time_end_2866;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "fut_kernel_map_transpose_i32");
            fprintf(stderr, "%zu", global_work_size_2864[0]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_2864[1]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_2864[2]);
            fprintf(stderr, "].\n");
            time_start_2865 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue,
                                              fut_kernel_map_transpose_i32, 3,
                                              NULL, global_work_size_2864,
                                              local_work_size_2868, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_2866 = get_wall_time();
            
            long time_diff_2867 = time_end_2866 - time_start_2865;
            
            if (detail_timing) {
                fut_kernel_map_transpose_i32total_runtime += time_diff_2867;
                fut_kernel_map_transpose_i32runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "fut_kernel_map_transpose_i32", (int) time_diff_2867);
            }
        }
    }
    
    int32_t last_in_group_index_2369 = group_size_2330 - 1;
    struct memblock_device mem_2646;
    
    mem_2646.references = NULL;
    memblock_alloc_device(&mem_2646, x_2622);
    
    struct memblock_device mem_2648;
    
    mem_2648.references = NULL;
    memblock_alloc_device(&mem_2648, x_2622);
    
    struct memblock_device mem_2650;
    
    mem_2650.references = NULL;
    memblock_alloc_device(&mem_2650, x_2622);
    
    struct memblock_device mem_2652;
    
    mem_2652.references = NULL;
    memblock_alloc_device(&mem_2652, x_2622);
    
    int32_t group_size_2748;
    int32_t num_groups_2749;
    
    group_size_2748 = cl_group_size;
    num_groups_2749 = squot32(num_groups_2329 + group_size_2748 - 1,
                              group_size_2748);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2367, 0, sizeof(mem_2632.mem),
                                  &mem_2632.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2367, 1, sizeof(num_groups_2329),
                                  &num_groups_2329));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2367, 2, sizeof(mem_2629.mem),
                                  &mem_2629.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2367, 3,
                                  sizeof(last_in_group_index_2369),
                                  &last_in_group_index_2369));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2367, 4, sizeof(mem_2626.mem),
                                  &mem_2626.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2367, 5, sizeof(group_size_2330),
                                  &group_size_2330));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2367, 6, sizeof(mem_2623.mem),
                                  &mem_2623.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2367, 7, sizeof(mem_2646.mem),
                                  &mem_2646.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2367, 8, sizeof(mem_2648.mem),
                                  &mem_2648.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2367, 9, sizeof(mem_2650.mem),
                                  &mem_2650.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2367, 10, sizeof(mem_2652.mem),
                                  &mem_2652.mem));
    if (1 * (num_groups_2749 * group_size_2748) != 0) {
        const size_t global_work_size_2869[1] = {num_groups_2749 *
                     group_size_2748};
        const size_t local_work_size_2873[1] = {group_size_2748};
        int64_t time_start_2870, time_end_2871;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_2367");
            fprintf(stderr, "%zu", global_work_size_2869[0]);
            fprintf(stderr, "].\n");
            time_start_2870 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_2367, 1,
                                              NULL, global_work_size_2869,
                                              local_work_size_2873, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_2871 = get_wall_time();
            
            long time_diff_2872 = time_end_2871 - time_start_2870;
            
            if (detail_timing) {
                map_kernel_2367total_runtime += time_diff_2872;
                map_kernel_2367runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_2367",
                        (int) time_diff_2872);
            }
        }
    }
    
    struct memblock_device mem_2654;
    
    mem_2654.references = NULL;
    memblock_alloc_device(&mem_2654, x_2622);
    
    struct memblock_device mem_2656;
    
    mem_2656.references = NULL;
    memblock_alloc_device(&mem_2656, x_2622);
    
    struct memblock_device mem_2658;
    
    mem_2658.references = NULL;
    memblock_alloc_device(&mem_2658, x_2622);
    
    struct memblock_device mem_2660;
    
    mem_2660.references = NULL;
    memblock_alloc_device(&mem_2660, x_2622);
    
    struct memblock_device mem_2663;
    
    mem_2663.references = NULL;
    memblock_alloc_device(&mem_2663, x_2622);
    
    struct memblock_device mem_2666;
    
    mem_2666.references = NULL;
    memblock_alloc_device(&mem_2666, x_2622);
    
    struct memblock_device mem_2669;
    
    mem_2669.references = NULL;
    memblock_alloc_device(&mem_2669, x_2622);
    
    struct memblock_device mem_2672;
    
    mem_2672.references = NULL;
    memblock_alloc_device(&mem_2672, x_2622);
    
    int32_t total_size_2778 = sizeof(int32_t) * num_groups_2329;
    int32_t total_size_2780 = sizeof(int32_t) * num_groups_2329;
    int32_t total_size_2782 = sizeof(int32_t) * num_groups_2329;
    int32_t total_size_2784;
    
    total_size_2784 = sizeof(int32_t) * num_groups_2329;
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2388, 0, total_size_2778, NULL));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2388, 1, total_size_2780, NULL));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2388, 2, total_size_2782, NULL));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2388, 3, total_size_2784, NULL));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2388, 4, sizeof(mem_2648.mem),
                                  &mem_2648.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2388, 5, sizeof(mem_2652.mem),
                                  &mem_2652.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2388, 6, sizeof(num_groups_2329),
                                  &num_groups_2329));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2388, 7, sizeof(mem_2646.mem),
                                  &mem_2646.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2388, 8, sizeof(mem_2650.mem),
                                  &mem_2650.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2388, 9, sizeof(mem_2654.mem),
                                  &mem_2654.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2388, 10, sizeof(mem_2656.mem),
                                  &mem_2656.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2388, 11, sizeof(mem_2658.mem),
                                  &mem_2658.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2388, 12, sizeof(mem_2660.mem),
                                  &mem_2660.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2388, 13, sizeof(mem_2663.mem),
                                  &mem_2663.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2388, 14, sizeof(mem_2666.mem),
                                  &mem_2666.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2388, 15, sizeof(mem_2669.mem),
                                  &mem_2669.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_2388, 16, sizeof(mem_2672.mem),
                                  &mem_2672.mem));
    if (1 * num_groups_2329 != 0) {
        const size_t global_work_size_2874[1] = {num_groups_2329};
        const size_t local_work_size_2878[1] = {num_groups_2329};
        int64_t time_start_2875, time_end_2876;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "scan_kernel_2388");
            fprintf(stderr, "%zu", global_work_size_2874[0]);
            fprintf(stderr, "].\n");
            time_start_2875 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, scan_kernel_2388, 1,
                                              NULL, global_work_size_2874,
                                              local_work_size_2878, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_2876 = get_wall_time();
            
            long time_diff_2877 = time_end_2876 - time_start_2875;
            
            if (detail_timing) {
                scan_kernel_2388total_runtime += time_diff_2877;
                scan_kernel_2388runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_kernel_2388", (int) time_diff_2877);
            }
        }
    }
    
    struct memblock_device mem_2675;
    
    mem_2675.references = NULL;
    memblock_alloc_device(&mem_2675, bytes_2621);
    
    struct memblock_device mem_2678;
    
    mem_2678.references = NULL;
    memblock_alloc_device(&mem_2678, bytes_2621);
    
    struct memblock_device mem_2681;
    
    mem_2681.references = NULL;
    memblock_alloc_device(&mem_2681, bytes_2621);
    
    struct memblock_device mem_2684;
    
    mem_2684.references = NULL;
    memblock_alloc_device(&mem_2684, bytes_2621);
    
    int32_t group_size_2790;
    int32_t num_groups_2791;
    
    group_size_2790 = cl_group_size;
    num_groups_2791 = squot32(num_groups_2329 * group_size_2330 +
                              group_size_2790 - 1, group_size_2790);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2464, 0, sizeof(mem_2632.mem),
                                  &mem_2632.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2464, 1, sizeof(mem_2672.mem),
                                  &mem_2672.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2464, 2, sizeof(num_groups_2329),
                                  &num_groups_2329));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2464, 3, sizeof(mem_2629.mem),
                                  &mem_2629.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2464, 4, sizeof(mem_2669.mem),
                                  &mem_2669.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2464, 5, sizeof(mem_2626.mem),
                                  &mem_2626.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2464, 6, sizeof(group_size_2330),
                                  &group_size_2330));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2464, 7, sizeof(mem_2666.mem),
                                  &mem_2666.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2464, 8, sizeof(mem_2623.mem),
                                  &mem_2623.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2464, 9, sizeof(mem_2663.mem),
                                  &mem_2663.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2464, 10, sizeof(mem_2675.mem),
                                  &mem_2675.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2464, 11, sizeof(mem_2678.mem),
                                  &mem_2678.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2464, 12, sizeof(mem_2681.mem),
                                  &mem_2681.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2464, 13, sizeof(mem_2684.mem),
                                  &mem_2684.mem));
    if (1 * (num_groups_2791 * group_size_2790) != 0) {
        const size_t global_work_size_2879[1] = {num_groups_2791 *
                     group_size_2790};
        const size_t local_work_size_2883[1] = {group_size_2790};
        int64_t time_start_2880, time_end_2881;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_2464");
            fprintf(stderr, "%zu", global_work_size_2879[0]);
            fprintf(stderr, "].\n");
            time_start_2880 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_2464, 1,
                                              NULL, global_work_size_2879,
                                              local_work_size_2883, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_2881 = get_wall_time();
            
            long time_diff_2882 = time_end_2881 - time_start_2880;
            
            if (detail_timing) {
                map_kernel_2464total_runtime += time_diff_2882;
                map_kernel_2464runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_2464",
                        (int) time_diff_2882);
            }
        }
    }
    
    struct memblock_device mem_2686;
    
    mem_2686.references = NULL;
    memblock_alloc_device(&mem_2686, bytes_2585);
    
    struct memblock_device mem_2688;
    
    mem_2688.references = NULL;
    memblock_alloc_device(&mem_2688, bytes_2585);
    
    struct memblock_device mem_2690;
    
    mem_2690.references = NULL;
    memblock_alloc_device(&mem_2690, bytes_2585);
    
    struct memblock_device mem_2692;
    
    mem_2692.references = NULL;
    memblock_alloc_device(&mem_2692, bytes_2585);
    
    int32_t group_size_2792;
    int32_t num_groups_2793;
    
    group_size_2792 = cl_group_size;
    num_groups_2793 = squot32(size_634 + group_size_2792 - 1, group_size_2792);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2495, 0, sizeof(mem_2684.mem),
                                  &mem_2684.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2495, 1, sizeof(mem_2644.mem),
                                  &mem_2644.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2495, 2, sizeof(mem_2681.mem),
                                  &mem_2681.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2495, 3, sizeof(mem_2641.mem),
                                  &mem_2641.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2495, 4, sizeof(size_634),
                                  &size_634));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2495, 5,
                                  sizeof(per_thread_elements_2334),
                                  &per_thread_elements_2334));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2495, 6, sizeof(mem_2678.mem),
                                  &mem_2678.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2495, 7, sizeof(group_size_2330),
                                  &group_size_2330));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2495, 8, sizeof(mem_2638.mem),
                                  &mem_2638.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2495, 9, sizeof(mem_2635.mem),
                                  &mem_2635.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2495, 10, sizeof(mem_2675.mem),
                                  &mem_2675.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2495, 11, sizeof(mem_2686.mem),
                                  &mem_2686.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2495, 12, sizeof(mem_2688.mem),
                                  &mem_2688.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2495, 13, sizeof(mem_2690.mem),
                                  &mem_2690.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_2495, 14, sizeof(mem_2692.mem),
                                  &mem_2692.mem));
    if (1 * (num_groups_2793 * group_size_2792) != 0) {
        const size_t global_work_size_2884[1] = {num_groups_2793 *
                     group_size_2792};
        const size_t local_work_size_2888[1] = {group_size_2792};
        int64_t time_start_2885, time_end_2886;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_2495");
            fprintf(stderr, "%zu", global_work_size_2884[0]);
            fprintf(stderr, "].\n");
            time_start_2885 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_2495, 1,
                                              NULL, global_work_size_2884,
                                              local_work_size_2888, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_2886 = get_wall_time();
            
            long time_diff_2887 = time_end_2886 - time_start_2885;
            
            if (detail_timing) {
                map_kernel_2495total_runtime += time_diff_2887;
                map_kernel_2495runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_2495",
                        (int) time_diff_2887);
            }
        }
    }
    memblock_set_device(&out_mem_2693, &mem_2686);
    out_arrsize_2695 = size_634;
    out_memsize_2694 = bytes_2585;
    memblock_set_device(&out_mem_2696, &mem_2688);
    out_memsize_2697 = bytes_2585;
    memblock_set_device(&out_mem_2698, &mem_2690);
    out_memsize_2699 = bytes_2585;
    memblock_set_device(&out_mem_2700, &mem_2692);
    out_memsize_2701 = bytes_2585;
    
    struct tuple_int32_t_device_mem_int32_t_int32_t_device_mem_int32_t_device_mem_int32_t_device_mem
    retval_2794;
    
    retval_2794.elem_0 = out_memsize_2694;
    retval_2794.elem_1.references = NULL;
    memblock_set_device(&retval_2794.elem_1, &out_mem_2693);
    retval_2794.elem_2 = out_arrsize_2695;
    retval_2794.elem_3 = out_memsize_2697;
    retval_2794.elem_4.references = NULL;
    memblock_set_device(&retval_2794.elem_4, &out_mem_2696);
    retval_2794.elem_5 = out_memsize_2699;
    retval_2794.elem_6.references = NULL;
    memblock_set_device(&retval_2794.elem_6, &out_mem_2698);
    retval_2794.elem_7 = out_memsize_2701;
    retval_2794.elem_8.references = NULL;
    memblock_set_device(&retval_2794.elem_8, &out_mem_2700);
    memblock_unref_device(&out_mem_2693);
    memblock_unref_device(&out_mem_2696);
    memblock_unref_device(&out_mem_2698);
    memblock_unref_device(&out_mem_2700);
    memblock_unref_device(&mem_2586);
    memblock_unref_device(&mem_2588);
    memblock_unref_device(&mem_2590);
    memblock_unref_device(&mem_2592);
    memblock_unref_device(&mem_2594);
    memblock_unref_device(&mem_2597);
    memblock_unref_device(&mem_2599);
    memblock_unref_device(&mem_2602);
    memblock_unref_device(&mem_2604);
    memblock_unref_device(&mem_2607);
    memblock_unref_device(&mem_2609);
    memblock_unref_device(&mem_2612);
    memblock_unref_device(&mem_2614);
    memblock_unref_device(&mem_2616);
    memblock_unref_device(&mem_2618);
    memblock_unref_device(&mem_2620);
    memblock_unref_device(&mem_2623);
    memblock_unref_device(&mem_2626);
    memblock_unref_device(&mem_2629);
    memblock_unref_device(&mem_2632);
    memblock_unref_device(&mem_2635);
    memblock_unref_device(&mem_2638);
    memblock_unref_device(&mem_2641);
    memblock_unref_device(&mem_2644);
    memblock_unref_device(&mem_2646);
    memblock_unref_device(&mem_2648);
    memblock_unref_device(&mem_2650);
    memblock_unref_device(&mem_2652);
    memblock_unref_device(&mem_2654);
    memblock_unref_device(&mem_2656);
    memblock_unref_device(&mem_2658);
    memblock_unref_device(&mem_2660);
    memblock_unref_device(&mem_2663);
    memblock_unref_device(&mem_2666);
    memblock_unref_device(&mem_2669);
    memblock_unref_device(&mem_2672);
    memblock_unref_device(&mem_2675);
    memblock_unref_device(&mem_2678);
    memblock_unref_device(&mem_2681);
    memblock_unref_device(&mem_2684);
    memblock_unref_device(&mem_2686);
    memblock_unref_device(&mem_2688);
    memblock_unref_device(&mem_2690);
    memblock_unref_device(&mem_2692);
    return retval_2794;
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
    
    int32_t xs_mem_size_2583;
    struct memblock xs_mem_2584;
    
    xs_mem_2584.references = NULL;
    memblock_alloc(&xs_mem_2584, 0);
    
    int32_t size_634;
    struct tuple_int32_t_device_mem_int32_t_int32_t_device_mem_int32_t_device_mem_int32_t_device_mem
    main_ret_2889;
    
    {
        int64_t shape[1];
        
        if (read_array(sizeof(int32_t), read_int32, (void **) &xs_mem_2584.mem,
                       shape, 1) != 0)
            panic(1, "Syntax error when reading %s.\n", "[i32]");
        size_634 = shape[0];
        xs_mem_size_2583 = sizeof(int32_t) * shape[0];
    }
    
    struct memblock_device xs_mem_device_2890;
    
    xs_mem_device_2890.references = NULL;
    memblock_alloc_device(&xs_mem_device_2890, xs_mem_size_2583);
    if (xs_mem_size_2583 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue,
                                            xs_mem_device_2890.mem, CL_TRUE, 0,
                                            xs_mem_size_2583, xs_mem_2584.mem +
                                            0, 0, NULL, NULL));
    
    int32_t out_memsize_2694;
    struct memblock out_mem_2693;
    
    out_mem_2693.references = NULL;
    
    int32_t out_arrsize_2695;
    int32_t out_memsize_2697;
    struct memblock out_mem_2696;
    
    out_mem_2696.references = NULL;
    
    int32_t out_memsize_2699;
    struct memblock out_mem_2698;
    
    out_mem_2698.references = NULL;
    
    int32_t out_memsize_2701;
    struct memblock out_mem_2700;
    
    out_mem_2700.references = NULL;
    if (perform_warmup) {
        time_runs = 0;
        t_start = get_wall_time();
        main_ret_2889 = futhark_main(xs_mem_size_2583, xs_mem_device_2890,
                                     size_634);
        OPENCL_SUCCEED(clFinish(fut_cl_queue));
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%ld\n", elapsed_usec);
        memblock_unref_device(&main_ret_2889.elem_1);
        memblock_unref_device(&main_ret_2889.elem_4);
        memblock_unref_device(&main_ret_2889.elem_6);
        memblock_unref_device(&main_ret_2889.elem_8);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        if (run == num_runs - 1)
            detail_timing = 1;
        t_start = get_wall_time();
        main_ret_2889 = futhark_main(xs_mem_size_2583, xs_mem_device_2890,
                                     size_634);
        OPENCL_SUCCEED(clFinish(fut_cl_queue));
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%ld\n", elapsed_usec);
        if (run < num_runs - 1) {
            memblock_unref_device(&main_ret_2889.elem_1);
            memblock_unref_device(&main_ret_2889.elem_4);
            memblock_unref_device(&main_ret_2889.elem_6);
            memblock_unref_device(&main_ret_2889.elem_8);
        }
    }
    memblock_unref(&xs_mem_2584);
    out_memsize_2694 = main_ret_2889.elem_0;
    memblock_alloc(&out_mem_2693, out_memsize_2694);
    if (out_memsize_2694 > 0)
        OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                           main_ret_2889.elem_1.mem, CL_TRUE, 0,
                                           out_memsize_2694, out_mem_2693.mem +
                                           0, 0, NULL, NULL));
    out_arrsize_2695 = main_ret_2889.elem_2;
    out_memsize_2697 = main_ret_2889.elem_3;
    memblock_alloc(&out_mem_2696, out_memsize_2697);
    if (out_memsize_2697 > 0)
        OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                           main_ret_2889.elem_4.mem, CL_TRUE, 0,
                                           out_memsize_2697, out_mem_2696.mem +
                                           0, 0, NULL, NULL));
    out_memsize_2699 = main_ret_2889.elem_5;
    memblock_alloc(&out_mem_2698, out_memsize_2699);
    if (out_memsize_2699 > 0)
        OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                           main_ret_2889.elem_6.mem, CL_TRUE, 0,
                                           out_memsize_2699, out_mem_2698.mem +
                                           0, 0, NULL, NULL));
    out_memsize_2701 = main_ret_2889.elem_7;
    memblock_alloc(&out_mem_2700, out_memsize_2701);
    if (out_memsize_2701 > 0)
        OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                           main_ret_2889.elem_8.mem, CL_TRUE, 0,
                                           out_memsize_2701, out_mem_2700.mem +
                                           0, 0, NULL, NULL));
    if (out_arrsize_2695 == 0)
        printf("empty(%s)", "i32");
    else {
        int print_i_2891;
        
        putchar('[');
        for (print_i_2891 = 0; print_i_2891 < out_arrsize_2695;
             print_i_2891++) {
            int32_t *print_elem_2892 = (int32_t *) out_mem_2693.mem +
                    print_i_2891 * 1;
            
            printf("%di32", *print_elem_2892);
            if (print_i_2891 != out_arrsize_2695 - 1)
                printf(", ");
        }
        putchar(']');
    }
    printf("\n");
    if (out_arrsize_2695 == 0)
        printf("empty(%s)", "i32");
    else {
        int print_i_2893;
        
        putchar('[');
        for (print_i_2893 = 0; print_i_2893 < out_arrsize_2695;
             print_i_2893++) {
            int32_t *print_elem_2894 = (int32_t *) out_mem_2696.mem +
                    print_i_2893 * 1;
            
            printf("%di32", *print_elem_2894);
            if (print_i_2893 != out_arrsize_2695 - 1)
                printf(", ");
        }
        putchar(']');
    }
    printf("\n");
    if (out_arrsize_2695 == 0)
        printf("empty(%s)", "i32");
    else {
        int print_i_2895;
        
        putchar('[');
        for (print_i_2895 = 0; print_i_2895 < out_arrsize_2695;
             print_i_2895++) {
            int32_t *print_elem_2896 = (int32_t *) out_mem_2698.mem +
                    print_i_2895 * 1;
            
            printf("%di32", *print_elem_2896);
            if (print_i_2895 != out_arrsize_2695 - 1)
                printf(", ");
        }
        putchar(']');
    }
    printf("\n");
    if (out_arrsize_2695 == 0)
        printf("empty(%s)", "i32");
    else {
        int print_i_2897;
        
        putchar('[');
        for (print_i_2897 = 0; print_i_2897 < out_arrsize_2695;
             print_i_2897++) {
            int32_t *print_elem_2898 = (int32_t *) out_mem_2700.mem +
                    print_i_2897 * 1;
            
            printf("%di32", *print_elem_2898);
            if (print_i_2897 != out_arrsize_2695 - 1)
                printf(", ");
        }
        putchar(']');
    }
    printf("\n");
    
    int total_runtime = 0;
    int total_runs = 0;
    
    if (cl_debug) {
        fprintf(stderr,
                "Kernel map_kernel_2324              executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_2324runs, (long) map_kernel_2324total_runtime /
                (map_kernel_2324runs != 0 ? map_kernel_2324runs : 1),
                (long) map_kernel_2324total_runtime);
        total_runtime += map_kernel_2324total_runtime;
        total_runs += map_kernel_2324runs;
        fprintf(stderr,
                "Kernel fut_kernel_map_transpose_i32 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                fut_kernel_map_transpose_i32runs,
                (long) fut_kernel_map_transpose_i32total_runtime /
                (fut_kernel_map_transpose_i32runs !=
                 0 ? fut_kernel_map_transpose_i32runs : 1),
                (long) fut_kernel_map_transpose_i32total_runtime);
        total_runtime += fut_kernel_map_transpose_i32total_runtime;
        total_runs += fut_kernel_map_transpose_i32runs;
        fprintf(stderr,
                "Kernel scan_kernel_2335             executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                scan_kernel_2335runs, (long) scan_kernel_2335total_runtime /
                (scan_kernel_2335runs != 0 ? scan_kernel_2335runs : 1),
                (long) scan_kernel_2335total_runtime);
        total_runtime += scan_kernel_2335total_runtime;
        total_runs += scan_kernel_2335runs;
        fprintf(stderr,
                "Kernel map_kernel_2367              executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_2367runs, (long) map_kernel_2367total_runtime /
                (map_kernel_2367runs != 0 ? map_kernel_2367runs : 1),
                (long) map_kernel_2367total_runtime);
        total_runtime += map_kernel_2367total_runtime;
        total_runs += map_kernel_2367runs;
        fprintf(stderr,
                "Kernel scan_kernel_2388             executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                scan_kernel_2388runs, (long) scan_kernel_2388total_runtime /
                (scan_kernel_2388runs != 0 ? scan_kernel_2388runs : 1),
                (long) scan_kernel_2388total_runtime);
        total_runtime += scan_kernel_2388total_runtime;
        total_runs += scan_kernel_2388runs;
        fprintf(stderr,
                "Kernel map_kernel_2464              executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_2464runs, (long) map_kernel_2464total_runtime /
                (map_kernel_2464runs != 0 ? map_kernel_2464runs : 1),
                (long) map_kernel_2464total_runtime);
        total_runtime += map_kernel_2464total_runtime;
        total_runs += map_kernel_2464runs;
        fprintf(stderr,
                "Kernel map_kernel_2495              executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_2495runs, (long) map_kernel_2495total_runtime /
                (map_kernel_2495runs != 0 ? map_kernel_2495runs : 1),
                (long) map_kernel_2495total_runtime);
        total_runtime += map_kernel_2495total_runtime;
        total_runs += map_kernel_2495runs;
    }
    if (cl_debug)
        fprintf(stderr, "Ran %d kernels with cumulative runtime: %6ldus\n",
                total_runs, total_runtime);
    memblock_unref_device(&main_ret_2889.elem_1);
    memblock_unref_device(&main_ret_2889.elem_4);
    memblock_unref_device(&main_ret_2889.elem_6);
    memblock_unref_device(&main_ret_2889.elem_8);
    if (runtime_file != NULL)
        fclose(runtime_file);
    return 0;
}
