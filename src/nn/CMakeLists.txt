
project(test version 1.0)

add_library(testLib STATIC test.cpp)

target_include_directories(testLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(testExe test.cpp)

target_link_libraries(testExe PRIVATE testLib)