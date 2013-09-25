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

#pragma once

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include <vector>

#include "../cv/serialize_cvmat.h"

/**
 * This is meant to be a generic classifier that could potentially be
 * implemented using Support Vector Machines, Neural Networks, Decision Trees,
 * or whatever classifier you fancy. Currently the classifier is implemented
 * using k-nearest-neighbors.
 */
class classifier {

   public:
   struct settings {
      int neighbors;
      settings() : neighbors(5) { }
      settings(const settings &s) { }

      protected:
      // Class serialization
      friend class boost::serialization::access;
      template<class archive>
      void serialize(archive &ar, const unsigned int version);
   };

   protected:
   settings my_settings;
   cv::KNearest nearest_neighbors;
   cv::Mat samples;
   cv::Mat responses;

   void train();

   public:
   // Update the settings for classification
   void set_settings(const settings &s);
   settings get_settings() { return my_settings; }

   // Trains a classifier with a set of samples and responses
   void train(const cv::Mat &s, const cv::Mat &r);

   // Classifies samples and returns their corresponding labels
   std::vector<float> classify(const cv::Mat &samples) const;

   protected:
   // Class serialization
   friend class boost::serialization::access;
   template<class archive>
   void serialize(archive &ar, const unsigned int version);    


};

/**
 * The intent of this class is to simplify the construction of a
 * classifier by building a list of samples.
 */
struct classifier_factory {
   cv::Mat samples;
   cv::Mat responses;
   
   void add_feature_vector(const std::vector<double> &vector, float response);
   
   classifier create_classifier(const classifier::settings &s = classifier::settings());
};

template<class archive>
void classifier::serialize(archive &ar, const unsigned int version) {
   ar &my_settings;
   ar &samples;
   ar &responses;
   if (archive::is_loading::value) {
      train();
   }
}

template<class archive>
void classifier::settings::serialize(archive &ar, const unsigned int version) {
   ar &neighbors;
}


