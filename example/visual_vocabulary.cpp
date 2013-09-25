/**
 * ImClass visual vocabulary generator
 *
 * Richie Steigerwald
 *
 * Copyright 2013 Richie Steigerwald <richie.steigerwald@gmail.com>
 * This work is free. You can redistribute it and/or modify it under the
 * terms of the Do What The Fuck You Want To Public License, Version 2,
 * as published by Sam Hocevar. See http://www.wtfpl.net/ for more details.
 */

#include <fstream>

#include <opencv2/core/core.hpp> // Mat
#include <opencv2/highgui/highgui.hpp> // imread
#include <opencv2/nonfree/features2d.hpp> // SURF

#include "cv/visual_vocabulary.h"
#include "files.hpp"

using namespace std;

void usage(const string &program);

/**
 * Gets all images in a directory then computes all of the features and the
 * descriptors for each image. All of the descriptors are added to a list then
 * compiled into a visual vocabulary.
 */
int main(int argc, char **argv) {
   if (argc < 2 || argc > 3) { usage(argv[0]); return 0; }

   // Get all files in directory recursively
   list<string> images = get_files_recursive(argv[1], ".png");
   
   visual_vocabulary_factory vv_fact; 
   cv::SurfFeatureDetector detector(200);
   cv::SurfDescriptorExtractor extractor;

   vector<cv::KeyPoint> keypoints;
   cv::Mat descriptors;

   // Compute features for each image and add to the descriptor list
   for (list<string>::iterator image = images.begin(); image != images.end(); image++) {
      cv::Mat grayscale_image = cv::imread(*image, CV_LOAD_IMAGE_GRAYSCALE);

      detector.detect(grayscale_image, keypoints);
      extractor.compute(grayscale_image, keypoints, descriptors);
      vv_fact.add_descriptors(descriptors);
   }

   // Generate a visual vocabulary
   visual_vocabulary vocab = vv_fact.compute_visual_vocabulary(visual_vocabulary::settings());

   // Save the visual vocabulary
   if (argc > 2) {
      std::fstream fs;
      fs.open(argv[2], std::fstream::out);

      boost::archive::text_oarchive oa(fs);
      oa << vocab;
   }
}


// Display usage information
void usage(const string &program) {
   cout << "Usage: " << program << " path/to/images [output.vv]" << endl;
}

