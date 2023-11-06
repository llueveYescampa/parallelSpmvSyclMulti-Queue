# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
  
#Create a variable called linalg_SOURCES containing all .cpp files:

set(matrixPartition_Src
    src/matrixPartition/getRowsNnzPerQueue.cpp
    PARENT_SCOPE
    )

