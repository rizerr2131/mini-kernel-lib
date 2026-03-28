add_executable(mklib_conv_bench conv_bench.cpp)
target_link_libraries(mklib_conv_bench PRIVATE mklib::mklib)

add_executable(mklib_gemm_bench gemm_bench.cpp)
target_link_libraries(mklib_gemm_bench PRIVATE mklib::mklib)

add_executable(mklib_reduce_bench reduce_bench.cpp)
target_link_libraries(mklib_reduce_bench PRIVATE mklib::mklib)
