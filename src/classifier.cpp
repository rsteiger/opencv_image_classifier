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

class image {
   static map<string, float> labels;
   static float max_label;

   string file;
   float label;

   public:
   float getLabel() const { return label; }
   string getFile() const { return file; }
   image(const string &fname, const string &l) : file(fname) { 
      if (labels.find(l) == labels.end()) {
         labels[l] = max_label++;
      }
      label = labels[l];
   }
};

map<string, float> image::labels;
float image::max_label = 0;

/**
 * This function generates the visual vocabulary using every feature
 * detected in a list of images.
 */
visual_vocabulary generate_visual_vocab(const list<string> &images) {
   visual_vocabulary_factory vv_fact; 
   cv::SurfFeatureDetector detector(200);
   cv::SurfDescriptorExtractor extractor;

   vector<cv::KeyPoint> keypoints;
   cv::Mat descriptors;

   // Compute features for each image and add to the descriptor list
   for (list<string>::const_iterator image = images.begin(); image != images.end(); image++) {
      cv::Mat grayscale_image = cv::imread(*image, CV_LOAD_IMAGE_GRAYSCALE);

      detector.detect(grayscale_image, keypoints);
      extractor.compute(grayscale_image, keypoints, descriptors);
      vv_fact.add_descriptors(descriptors);
   }

   // Generate a visual vocabulary
   return vv_fact.compute_visual_vocabulary(visual_vocabulary::settings());
}

/**
 * This function trains a classifier using a list of images and a list of image labels
 */
classifier generate_classifier(const visual_vocabulary &vocab, const list<image> &images) {
   cv::SurfFeatureDetector detector(200);
   cv::SurfDescriptorExtractor extractor;

   vector<cv::KeyPoint> keypoints;
   cv::Mat descriptors;

   bag_of_features bof;
   bof.set_vocabulary(vocab);

   classifier_factory fact;
   // Compute features for each image and add to the descriptor list
   for (list<image>::const_iterator image = images.begin(); image != images.end(); image++) {
      cv::Mat grayscale_image = cv::imread(image->getFile(), CV_LOAD_IMAGE_GRAYSCALE);

      detector.detect(grayscale_image, keypoints);
      extractor.compute(grayscale_image, keypoints, descriptors);
      std::vector<double> feature_vector = 
       bof.feature_vector(keypoints, descriptors);
      fact.add_feature_vector(feature_vector, image->getLabel());
   }

   classifier cls = fact.create_classifier();

   for (list<image>::const_iterator image = images.begin(); image != images.end(); image++) {
      cv::Mat grayscale_image = cv::imread(image->getFile(), CV_LOAD_IMAGE_GRAYSCALE);

      detector.detect(grayscale_image, keypoints);
      extractor.compute(grayscale_image, keypoints, descriptors);
      std::vector<double> feature_vector = 
       bof.feature_vector(keypoints, descriptors);
      float label = cls.classify(feature_vector); 
      std::cout << label << " " << image->getLabel() << std::endl;
   }

   return cls;
}

/**
 * Gets all images in a directory then computes all of the features and the
 * descriptors for each image. Each descriptor is then compared to each visual
 * word in the provided visual vocabulary to create a feature vector. 
 */ 
int main(int argc, char **argv) {
   if (argc < 2 || argc > 2) { usage(argv[0]); return 0; }
   list<string> images = get_files_recursive(argv[1], ".png");

   // get the label for each image
   list<image> image_categories;
   for (list<string>::iterator im = images.begin(); im != images.end(); im++) {
      boost::filesystem::path p(*im);
      string label = p.parent_path().leaf().string();
      image_categories.push_back(image(*im, label));
   }

   visual_vocabulary vocab = generate_visual_vocab(images);
   classifier cls = generate_classifier(vocab, image_categories);
}

// Display usage information
void usage(const string &program) {
   cout << "Usage: " << program << " path/to/images" << endl;
}

