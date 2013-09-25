cmake_minimum_required(VERSION 2.8)
project( IMAGE_CLASSIFIER )

find_package( OpenCV REQUIRED )
find_package( Boost COMPONENTS system filesystem serialization REQUIRED )

include_directories( ${Boost_INCLUDE_DIR} )
include_directories( ${OpenCV_INCLUDE_DIR} )

set( CMAKE_CXX_FLAGS "-std=c++11" )

# Build the CV pieces
file ( GLOB CV_SOURCES src/cv/*.cpp )
file ( GLOB CV_HEADERS src/cv/*.h )
add_library( CVLib ${CV_SOURCES} ${CV_HEADERS} )

# Build the ML pieces
file ( GLOB ML_SOURCES src/ml/*.cpp )
file ( GLOB ML_HEADERS src/ml/*.h )
add_library( MLLib ${ML_SOURCES} ${ML_HEADERS} )


# Examples
add_executable( visual_vocabulary example/visual_vocabulary.cpp )
target_link_libraries( visual_vocabulary ${OpenCV_LIBS} 
                                         ${Boost_LIBRARIES} 
                                         CVLib )

add_executable( classifier example/classifier.cpp )
target_link_libraries( classifier ${OpenCV_LIBS} 
                                  ${Boost_LIBRARIES}
                                  CVLib
                                  MLLib )

add_executable( classify example/classify.cpp )
target_link_libraries( classify ${OpenCV_LIBS} 
                                ${Boost_LIBRARIES}
                                CVLib
                                MLLib )

# Tests
enable_testing()

include_directories( src )
add_executable( functional_test test/functional_test.cpp )

find_library(UnitTest_LIBS UnitTest++ PATHS /usr/local/lib)
target_link_libraries( functional_test CVLib
                                       MLLib
                                       ${OpenCV_LIBS} 
                                       ${Boost_LIBRARIES} 
                                       ${UnitTest_LIBS})

add_test(NAME functional_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test COMMAND functional_test images)
