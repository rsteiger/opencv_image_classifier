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

#include "classifier.h"

void classifier::set_settings(const settings &s) { 
   my_settings = s;
   train();
}

void classifier::train(const cv::Mat &s, const cv::Mat &r) {
   samples = s;
   responses = r;
   train();
}

void classifier::train() {
   if (samples.data != NULL) {
      nearest_neighbors.train(
       samples,                // train data, row sample
       responses,              // responses
       cv::Mat(),              // sample index
       false,                  // regression
       my_settings.neighbors); // number of neighbors
   }
}

std::vector<float> classifier::classify(const cv::Mat &samples) const {
   cv::Mat results(samples.rows, 1, CV_32F);
   nearest_neighbors.find_nearest(samples, my_settings.neighbors, &results);

   std::vector<float> responses;
   for (int i = 0; i < results.rows; i++) {
      responses.push_back(results.at<float>(i,0));
   }
   return responses;
}


void classifier_factory::add_feature_vector(const std::vector<double> &feature_vector, float response) {
   cv::Mat temp(1, feature_vector.size(), CV_32F);
   cv::Mat resp(1, 1, CV_32F);
   
   for (int i = 0; i < feature_vector.size(); i++) {
      temp.at<float>(0, i) = feature_vector[i];
   }
   resp.at<float>(0,0) = response;

   if (this->samples.data == NULL) {
      temp.copyTo(this->samples);
      resp.copyTo(this->responses);
   } else {
      assert(temp.cols == this->samples.cols);
      cv::vconcat(this->samples, temp, this->samples);
      cv::vconcat(this->responses, resp, this->responses);
   }
}

 
classifier classifier_factory::create_classifier(const classifier::settings &s) {
   classifier cls;
   cls.set_settings(s);
   cls.train(samples, responses);
   return cls;
}

