/**
 * Unit Tests for visual vocabulary generator
 *
 * Richie Steigerwald
 *
 * Copyright 2013 Richie Steigerwald <richie.steigerwald@gmail.com>
 * This work is free. You can redistribute it and/or modify it under the
 * terms of the Do What The Fuck You Want To Public License, Version 2,
 * as published by Sam Hocevar. See http://www.wtfpl.net/ for more details.
 */

#include <opencv2/core/core.hpp> // Mat
#include <opencv2/highgui/highgui.hpp> // imread
#include <opencv2/nonfree/features2d.hpp> // SURF

#include "cv/visual_vocabulary.h"
#include "cv/bag_of_features.h"
#include "ml/classifier.h"
#include "files.hpp"

#include <UnitTest++.h>

using namespace std;

list<string> images;

float row_standard_deviation(cv::Mat matrix);

/**
 * This test ensures that the images we are testing actually have keypoints and
 * they are not all clustered together.
 */
TEST(CheckKeypoints) {
   cv::SurfFeatureDetector detector(200);
   cv::SurfDescriptorExtractor extractor;

   vector<cv::KeyPoint> keypoints;

   for (list<string>::iterator image = images.begin(); image != images.end(); image++) {
      cv::Mat grayscale_image = cv::imread(*image, CV_LOAD_IMAGE_GRAYSCALE);
      detector.detect(grayscale_image, keypoints);

      // Check to make sure we actually have some features
      CHECK(keypoints.size() > 1);
   
      // Make sure the features are actually spread out across the image
      double sum_x = 0, sum_y = 0, sqsum_x = 0, sqsum_y = 0;
      for (int i = 0; i < keypoints.size(); i++) {
         sum_x += keypoints[i].pt.x;
         sum_y += keypoints[i].pt.y;
         sqsum_x += keypoints[i].pt.x * keypoints[i].pt.x;
         sqsum_y += keypoints[i].pt.y * keypoints[i].pt.y;
      }
      sum_x /= keypoints.size();
      sum_y /= keypoints.size();
      sqsum_x /= keypoints.size();
      sqsum_y /= keypoints.size();

      sum_x *= sum_x;
      sum_y *= sum_y;

      double stddev_x = sqrt(sqsum_x - sum_x);
      double stddev_y = sqrt(sqsum_y - sum_y);
   
      CHECK(stddev_x > 10);
      CHECK(stddev_y > 10);
   }
}

/**
 * This test ensures that the descriptors being computed for the keypoints
 * are not all the same
 */
TEST(CheckDescriptors) {
   cv::SurfFeatureDetector detector(200);
   cv::SurfDescriptorExtractor extractor;

   vector<cv::KeyPoint> keypoints;
   cv::Mat descriptors;

   // Compute features and descriptors for each image
   for (list<string>::iterator image = images.begin(); image != images.end(); image++) {
      cv::Mat grayscale_image = cv::imread(*image, CV_LOAD_IMAGE_GRAYSCALE);

      detector.detect(grayscale_image, keypoints);
      extractor.compute(grayscale_image, keypoints, descriptors);

      CHECK(descriptors.rows > 0 && descriptors.cols > 0);
      CHECK(row_standard_deviation(descriptors) > 0.1);
   }
}

/**
 * This test ensures that our visual vocabulary actually has variety
 */
TEST(CheckVisualVocabulary) {
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

   CHECK(row_standard_deviation(vocab.centroids) > 0.1);
}


/**
 * This test checks to see if the generated bags of words are actually
 * interesting
 */
TEST(ComputeFeatures) {
   visual_vocabulary_factory vv_fact; 
   cv::SurfFeatureDetector detector(200);
   cv::SurfDescriptorExtractor extractor;

   vector<vector<cv::KeyPoint> > keypoints_list;
   vector<cv::Mat > descriptors_list;

   // Compute features for each image and add to the descriptor list
   for (list<string>::iterator image = images.begin(); image != images.end(); image++) {
      vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors;

      cv::Mat grayscale_image = cv::imread(*image, CV_LOAD_IMAGE_GRAYSCALE);

      detector.detect(grayscale_image, keypoints);
      extractor.compute(grayscale_image, keypoints, descriptors);
      keypoints_list.push_back(keypoints);
      descriptors_list.push_back(descriptors);

      vv_fact.add_descriptors(descriptors);
   }

   // Generate a visual vocabulary
   visual_vocabulary vocab = vv_fact.compute_visual_vocabulary(visual_vocabulary::settings());

   // Compute bag of features representation for each image
   bag_of_features bof;
   bof.set_vocabulary(vocab); 

   classifier_factory fact;
   
   for (int i = 0; i < keypoints_list.size(); i++) {
      fact.add_feature_vector(bof.mat_feature_vector(keypoints_list[i], descriptors_list[i]),0);
   }
   
   CHECK(row_standard_deviation(fact.samples) > 1);
}

/**
 * This test checks to see that the classifier classifies all of the training
 * data correctly
 */
TEST(ClassifierResponses) {
   visual_vocabulary_factory vv_fact; 
   cv::SurfFeatureDetector detector(200);
   cv::SurfDescriptorExtractor extractor;

   vector<vector<cv::KeyPoint> > keypoints_list;
   vector<cv::Mat > descriptors_list;

   // Compute keypoints and descriptors for every image
   for (list<string>::iterator image = images.begin(); image != images.end(); image++) {
      vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors;

      cv::Mat grayscale_image = cv::imread(*image, CV_LOAD_IMAGE_GRAYSCALE);

      detector.detect(grayscale_image, keypoints);
      extractor.compute(grayscale_image, keypoints, descriptors);
      keypoints_list.push_back(keypoints);
      descriptors_list.push_back(descriptors);

      vv_fact.add_descriptors(descriptors);
   }

   // Generate a visual vocabulary
   visual_vocabulary vocab = vv_fact.compute_visual_vocabulary(visual_vocabulary::settings());

   // Compute bag of features representation for each image
   bag_of_features bof;
   bof.set_vocabulary(vocab); 

   classifier_factory fact;
   
   for (int i = 0; i < keypoints_list.size(); i++) {
      fact.add_feature_vector(bof.mat_feature_vector(keypoints_list[i], descriptors_list[i]),i);
   }

   // Create a classifier that searches for only one
   // nearest neighbor
   classifier::settings settings;
   settings.neighbors = 1;
   classifier cls = fact.create_classifier(settings); 
   
   vector<float> responses = cls.classify(fact.samples);
   for (int i = 0; i < responses.size(); i++) {
      CHECK(i == responses[i]);
   }
}

/**
 * This test checks to see whether or not the classification process is
 * accurate using cross-validation
 */
TEST(ClassifierAccuracy) {
   visual_vocabulary_factory vv_fact; 
   cv::SurfFeatureDetector detector(200);
   cv::SurfDescriptorExtractor extractor;

   map<string, float> labels;
   float max_label = 0;
   vector<float> label_list;
   // Construct the list of labels corresponding to each image
   for (list<string>::iterator image = images.begin(); image != images.end(); image++) {
      boost::filesystem::path p(*image);
      string label = p.parent_path().leaf().string();
      if (labels.find(label) == labels.end()) {
         labels[label] = max_label++;
      }
      label_list.push_back(labels[label]);
   }

   vector<vector<cv::KeyPoint> > keypoints_list;
   vector<cv::Mat > descriptors_list;

   // Compute keypoints and descriptors for every image
   for (list<string>::iterator image = images.begin(); image != images.end(); image++) {
      vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors;

      cv::Mat grayscale_image = cv::imread(*image, CV_LOAD_IMAGE_GRAYSCALE);

      detector.detect(grayscale_image, keypoints);
      extractor.compute(grayscale_image, keypoints, descriptors);
      keypoints_list.push_back(keypoints);
      descriptors_list.push_back(descriptors);

      vv_fact.add_descriptors(descriptors);
   }

   // Generate a visual vocabulary
   visual_vocabulary vocab = vv_fact.compute_visual_vocabulary(visual_vocabulary::settings());

   // Compute bag of features representation for one image
   bag_of_features bof;
   bof.set_vocabulary(vocab); 

   // Create a few folds for the data
   int num_folds = 5;
   std::vector<classifier_factory> factories(num_folds);
   std::vector<classifier_factory> folds(num_folds);
   for (int i = 0; i < keypoints_list.size(); i++) {
      vector<double> fv = bof.feature_vector(keypoints_list[i], descriptors_list[i]); 
      for (int j = 0; j < factories.size(); j++) {
         if (j == i % num_folds) {
            folds[j].add_feature_vector(fv, label_list[i]);
         } else {
            factories[j].add_feature_vector(fv, label_list[i]);
         }
      }
   }

   // Create and test classifiers for each fold
   int total_correct = 0;
   for (int j = 0; j < factories.size(); j++) {
      classifier cls = factories[j].create_classifier();
      std::vector<float> responses = cls.classify(folds[j].samples);
      for (int i = 0; i < responses.size(); i++) {
         total_correct += (responses[i] == folds[j].responses.at<float>(i,0));
      }
   }

   // For two class, hopefully better than random
   std::cout << ((float)total_correct / label_list.size()) << std::endl;
   CHECK((float)total_correct / label_list.size() > 0.5);
}



void usage(const string &program);

int main(int argc, char **argv) {
   if (argc != 2) { usage(argv[0]); return EXIT_FAILURE; }

   // Get all files in directory recursively
   images = get_files_recursive(argv[1], ".png");
   
   return UnitTest::RunAllTests();   
}

// Display usage information
void usage(const string &program) {
   cout << "Usage: " << program << " path/to/images" << endl;
}


float row_standard_deviation(cv::Mat matrix) {
   if (matrix.rows == 0)
      return 0;

   cv::Mat sq_mat;
   cv::multiply(matrix, matrix, sq_mat);

   // Get the mean and the mean of squares for descriptors
   cv::Mat sum, sq_sum;
   cv::reduce(matrix, sum, 0, CV_REDUCE_SUM);
   cv::reduce(sq_mat, sq_sum, 0, CV_REDUCE_SUM);
   sum /= matrix.rows;
   sq_sum /= matrix.rows;

   // Calculate standard deviation for descriptors
   cv::Mat std_dev;
   cv::multiply(sum, sum, sum);
   cv::subtract(sq_sum, sum, std_dev);
   cv::pow(std_dev, 0.5, std_dev);
   return cv::norm(std_dev);
}
