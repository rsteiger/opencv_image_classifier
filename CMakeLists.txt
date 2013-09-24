cmake_minimum_required(VERSION 2.8)
project( IMAGE_CLASSIFIER )

find_package( OpenCV REQUIRED )
find_package( Boost COMPONENTS system filesystem serialization REQUIRED )

include_directories( ${Boost_INCLUDE_DIR} )
include_directories( ${OpenCV_INCLUDE_DIR} )

set( CMAKE_CXX_FLAGS "-std=c++11" )

file ( GLOB CV_SOURCES src/cv/*.cpp )
file ( GLOB ML_SOURCES src/ml/*.cpp )

add_executable( visual_vocabulary src/visual_vocabulary.cpp ${CV_SOURCES} )
target_link_libraries( visual_vocabulary ${OpenCV_LIBS} ${Boost_LIBRARIES} )

add_executable( classifier src/classifier.cpp ${CV_SOURCES} ${ML_SOURCES} )
target_link_libraries( classifier ${OpenCV_LIBS} ${Boost_LIBRARIES} )


# Tests
enable_testing()
find_library(UnitTest_LIBS UnitTest++ PATHS /usr/local/lib)
 
include_directories( src )
add_executable( functional_test test/functional_test.cpp ${CV_SOURCES} ${ML_SOURCES} )
list(APPEND CMAKE_LIBRARY_PATH /usr/local/lib) 
target_link_libraries( functional_test ${OpenCV_LIBS} ${Boost_LIBRARIES} ${UnitTest_LIBS})

add_test(NAME functional_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test COMMAND functional_test images)
