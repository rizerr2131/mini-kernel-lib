add_executable(mklib_smoke smoke_test.cpp)
target_link_libraries(mklib_smoke PRIVATE mklib::mklib)

add_test(NAME mklib_smoke COMMAND mklib_smoke)
