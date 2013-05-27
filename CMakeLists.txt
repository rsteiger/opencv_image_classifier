cmake_minimum_required(VERSION 2.8)
project( IMAGE_CLASSIFIER )

find_package( OpenCV REQUIRED )
find_package( Boost COMPONENTS system filesystem REQUIRED )

include_directories( ${Boost_INCLUDE_DIR} )
include_directories( ${OpenCV_INCLUDE_DIR} )

set( CMAKE_CXX_FLAGS "-std=c++11" )

file ( GLOB CV_SOURCES src/cv/*.cpp )

add_executable( visual_vocabulary src/visual_vocabulary.cpp ${CV_SOURCES} )
target_link_libraries( visual_vocabulary ${OpenCV_LIBS} ${Boost_LIBRARIES} )
