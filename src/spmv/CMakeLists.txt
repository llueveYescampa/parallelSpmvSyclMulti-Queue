# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
  
#Create a variable called linalg_SOURCES containing all .cpp files:

set(spmv_Src 
    src/spmv/spmv.cpp
    PARENT_SCOPE
    )

