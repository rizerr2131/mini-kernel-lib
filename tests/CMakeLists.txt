add_executable(mklib_smoke smoke_test.cpp)
target_link_libraries(mklib_smoke PRIVATE mklib::mklib)

add_test(NAME mklib_smoke COMMAND mklib_smoke)

add_executable(mklib_conv_test conv_test.cpp)
target_link_libraries(mklib_conv_test PRIVATE mklib::mklib)

add_test(NAME mklib_conv_test COMMAND mklib_conv_test)

add_executable(mklib_gemm_test gemm_test.cpp)
target_link_libraries(mklib_gemm_test PRIVATE mklib::mklib)

add_test(NAME mklib_gemm_test COMMAND mklib_gemm_test)

add_executable(mklib_reduce_test reduction_test.cpp)
target_link_libraries(mklib_reduce_test PRIVATE mklib::mklib)

add_test(NAME mklib_reduce_test COMMAND mklib_reduce_test)
