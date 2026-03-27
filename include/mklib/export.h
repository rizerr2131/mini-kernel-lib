#ifndef MKLIB_EXPORT_H_
#define MKLIB_EXPORT_H_

#if defined(_WIN32)
  #if defined(MKLIB_BUILD_SHARED)
    #define MKLIB_API __declspec(dllexport)
  #elif defined(MKLIB_USE_SHARED)
    #define MKLIB_API __declspec(dllimport)
  #else
    #define MKLIB_API
  #endif
#else
  #define MKLIB_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
  #define MKLIB_EXTERN_C_BEGIN extern "C" {
  #define MKLIB_EXTERN_C_END }
#else
  #define MKLIB_EXTERN_C_BEGIN
  #define MKLIB_EXTERN_C_END
#endif

#endif  // MKLIB_EXPORT_H_
