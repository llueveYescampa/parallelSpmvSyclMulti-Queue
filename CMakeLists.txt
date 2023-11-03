cmake_minimum_required(VERSION 3.10)

## set appropriately the following line to create a Double Precision version
set(DOUBLE_PRECISION ON)

project(parallelSpmvSycl CXX)

set(CXX_STANDARD C++17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

## adding the following lines while cleaning the code
MESSAGE( STATUS "CMAKE_CXX_COMPILER_ID: "         ${CMAKE_CXX_COMPILER_ID} )

if ( CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" )
    set(CMAKE_CXX_FLAGS  "${CMAKE_C_FLAGS} -fsycl -Wall -fvectorize  -qopt-report-file=stderr -qopt-report=max  -xCORE-AVX2")
    #set(CMAKE_CXX_FLAGS  "${CMAKE_C_FLAGS} -Wall -qopt-report=5 -qopt-report-phase=vec -xCORE-AVX2")
    #set(CMAKE_CXX_FLAGS  "${CMAKE_C_FLAGS} -Wall -qopt-report=5 -qopt-report-phase=vec ")
endif ( )
## end of adding the following lines while cleaning the code


if (DOUBLE_PRECISION)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DDOUBLE")
endif()


include_directories("src/include")
add_subdirectory(src)

add_executable(parallelSpmvSycl ${parallelSpmvSycl_Src})
target_link_libraries(parallelSpmvSycl )
#target_link_libraries(parallelSpmvSycl  ${CMAKE_EXE_LINKER_FLAGS})


############### Debug messages ####################
#MESSAGE( STATUS "CMAKE_BINARY_DIR: "            ${CMAKE_BINARY_DIR} )
#MESSAGE( STATUS "CMAKE_C_COMPILER_ID: "         ${CMAKE_C_COMPILER_ID} )
#MESSAGE( STATUS "CMAKE_C_FLAGS: "               ${CMAKE_C_FLAGS} )
#MESSAGE( STATUS "CMAKE_CURRENT_SOURCE_DIR: "    ${CMAKE_CURRENT_SOURCE_DIR} )
########### end of Debug messages #################
