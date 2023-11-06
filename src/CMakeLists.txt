# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
  
#Create a variable called linalg_SOURCES containing all .cpp files:

add_subdirectory(main)
add_subdirectory(spmv)
add_subdirectory(matrixPartition)
add_subdirectory(reader)

set(parallelSpmvSycl_Src 
    ${main_Src}
    ${spmv_Src}
    ${matrixPartition_Src}
    ${reader_Src}
    PARENT_SCOPE
    )

