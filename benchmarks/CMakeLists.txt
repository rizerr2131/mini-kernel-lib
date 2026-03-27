add_executable(mklib_gemm_bench gemm_bench.cpp)
target_link_libraries(mklib_gemm_bench PRIVATE mklib::mklib)
