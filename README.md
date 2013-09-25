ImClass
========

This is a tool for performing image classification. This project primarily
serves as a starting point for developing and experimenting with more
sophisticated image classification algorithms using c++. This project is
intentionally small for the purpose of being flexible and easy to understand.


Getting Started
--------

ImClass is only known to build and run on Mac OS  10.8, however other systems
will probably work as well.

ImClass has the following dependencies:
 * [OpenCV (Open Source Computer Vision)](http://opencv.org/)
 * [Boost C++](http://www.boost.org/)
 * [UnitTest++](http://unittest-cpp.sourceforge.net/) (For testing)

If you are on a mac and have homebrew, you can install each of them with the
following:

    brew tap homebrew/science
    brew install opencv
    brew install boost
    brew install unittest-cpp

Once you have all of the dependencies installed, you can compile the program and
run the tests using CMake and Make as follows:

    mkdir build
    cd build
    cmake ..
    make
    make test


Example Programs
--------

The example programs are relatively simple. 
The program `visual_vocabulary` generates the visual vocabulary which is
basically a set of image feature descriptors that will be counted in each
image.

    $> visual_vocabulary directory/with/images [output.vv]

The program `classifier` uses a set of classified images in a directory
expected to have the layout: 

    images/apples/foo.png ... images/apples/bar.png
    images/bananas/alpha.png ... images/bananas/omega.png
    ...

Along with the visual vocabulary to train a bag-of-words classifier.

   $> classifier directory/with/images vocab.vv [classifier.cls]

The last program `classify` uses a visual vocabulary and a classifier to
determine the class of an unknown image.

   $> classify mysteryimage.png vocab.vv classifier.cls
   5
   $>

It prints out a number corresponding to the internal representation of the
determined class.


Testing
--------

In the `test` directory, there are a number of test image sets.
 * `images` - a set of images that the classifier performs well for
 * `photos` - a set of images that the classifier performs poorly on
 * `bad_images` - a set of images that are designed to break the classifier

