# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
  
#Create a variable called linalg_SOURCES containing all .cpp files:

set(reader_Src 
    src/reader/reader.cpp
    src/reader/getRowsNnzPerQueue.cpp
    src/reader/vectorReader.cpp
    PARENT_SCOPE
    )

