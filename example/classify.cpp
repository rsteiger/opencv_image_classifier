/**
 * OpenCV image classifier
 *
 * Richie Steigerwald
 *
 * Copyright 2013 Richie Steigerwald <richie.steigerwald@gmail.com>
 * This work is free. You can redistribute it and/or modify it under the
 * terms of the Do What The Fuck You Want To Public License, Version 2,
 * as published by Sam Hocevar. See http://www.wtfpl.net/ for more details.
 */

#include <fstream>
#include <map>
#include <iostream>

#include <opencv2/core/core.hpp> // Mat
#include <opencv2/highgui/highgui.hpp> // imread
#include <opencv2/nonfree/features2d.hpp> // SURF

#include "cv/bag_of_features.h"
#include "ml/classifier.h"

#include "files.hpp"

using namespace std;

void usage(const string &program);

float classify_image(const string &image, const visual_vocabulary &vocab, const classifier &cls) {
   cv::SurfFeatureDetector detector(200);
   cv::SurfDescriptorExtractor extractor;

   vector<cv::KeyPoint> keypoints;
   cv::Mat descriptors;

   bag_of_features bof;
   bof.set_vocabulary(vocab);

   classifier_factory fact;

   cv::Mat grayscale_image = cv::imread(image, CV_LOAD_IMAGE_GRAYSCALE);

   detector.detect(grayscale_image, keypoints);
   extractor.compute(grayscale_image, keypoints, descriptors);

   return cls.classify(bof.mat_feature_vector(keypoints, descriptors))[0];
}


int main(int argc, char **argv) {
   if (argc != 4) { usage(argv[0]); return 0; }

   // Load the visual vocabulary
   visual_vocabulary vocab;
   std::fstream fs;
   fs.open(argv[2], std::fstream::in);
   boost::archive::text_iarchive ia(fs);
   ia >> vocab;

   // Load the classifier
   classifier cls;
   std::fstream fs_cls;
   fs_cls.open(argv[3], std::fstream::in);
   boost::archive::text_iarchive ia_cls(fs_cls);
   ia_cls >> cls;

   std::cout << classify_image(argv[1], vocab, cls) << std::endl;
}

// Display usage information
void usage(const string &program) {
   cout << "Usage: " << program << " path/to/image vocab.vv classifier.cls" << endl;
}

