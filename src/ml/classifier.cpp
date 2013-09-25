#include "classifier.h"

#include <iostream>

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


float classifier::classify(const std::vector<double> &vector) const {
   cv::Mat temp(1, vector.size(), CV_32F);
   for (int i = 0; i < vector.size(); i++) {
      temp.at<float>(i, 0) = vector[i];
   }
   
   return classify(temp)[0];
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
