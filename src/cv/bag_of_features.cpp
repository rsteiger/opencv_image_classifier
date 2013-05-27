#include "bag_of_features.h"

#include <opencv2/core/core.hpp>

using namespace std;

/**
 * Computes soft assignment of a descriptor to the visual vocabulary
 * @param[in]  point  the descriptor for which to calculate the bag of features
 * @param[in]  vocab  the visual vocabulary to use for the bag of features
 */
cv::Mat bag_of_features::soft_assign(const cv::Mat &point) const {
   assert(point.cols == vocabulary.centroids.cols);

   cv::Mat feature_weights(1, vocabulary.centroids.rows, CV_64F);  

   // Calculate the contribution of this feature to each of the visual word counts
   for (int cluster_num = 0; cluster_num < vocabulary.centroids.rows; cluster_num++) {

      // Weight neighbor value with gaussian kernel function for soft kernel
      double distance = cv::norm(point, vocabulary.centroids.row(cluster_num));

      double inv_sigma_squared = 1.f / settings.kernel_distance_squared;
      double gaussian = exp(-pow(distance, 2.0) * inv_sigma_squared);

      // Add neighbor to weights list
      feature_weights.at<double>(0, cluster_num) = gaussian;
   }

   // Make sure the weight contributed by each feature is equivalent for soft kernel
   cv::normalize(feature_weights, feature_weights, feature_weights.cols, cv::NORM_L1);

   return feature_weights;
}

/**
 * Computes hard assignment of a descriptor to the visual vocabulary
 * @param[in]  point  the descriptor for which to calculate the bag of features
 * @param[in]  vocab  the visual vocabulary to use for the bag of features
 */
cv::Mat bag_of_features::hard_assign(const cv::Mat &point) const {
   assert(point.cols == vocabulary.centroids.cols);

   cv::Mat feature_weights = cv::Mat::zeros(1, vocabulary.centroids.rows, CV_32F);  
   int smallest_index = 0;
   double smallest_distance = numeric_limits<double>::infinity();

   // Calculate the contribution of this feature to each of the visual word counts
   for (int cluster_num = 0; cluster_num < vocabulary.centroids.rows; cluster_num++) {

      // Weight neighbor value with gaussian kernel function for soft kernel
      double distance = cv::norm(point, vocabulary.centroids.row(cluster_num));
      if (distance < smallest_distance) {
         smallest_distance = distance;
         smallest_index = cluster_num;
      }
   }

   // Add neighbor to weights list
   feature_weights.at<double>(0, smallest_index) = 1;

   return feature_weights;
}

/**
 * Computes the feature vector for a set of features
 * @param[in]  features      the list of features used to generate the descriptors
 * @param[in]  descriptors   a list of row-features to create a histogram for
 * @param[in]  vocab_list    a list of row-features to be used as the visual vocabulary
 * @return  a feature vector
 */
vector<double> bag_of_features::feature_vector(const vector<cv::KeyPoint>
      &features, const cv::Mat &descriptors) const {

   assert(descriptors.rows == features.size());

   // Create an empty histogram
   std::vector<double> image_histogram;
   image_histogram.resize(vocabulary.centroids.rows * pyramid_size(settings.spatial_pyramid_depth));
   std::fill(image_histogram.begin(), image_histogram.end(), 0);

   // For each feature, add its contribution to the histogram
   for (int feature_num = 0; feature_num < descriptors.rows; feature_num++) {
      cv::Mat normalized_weights, feature_weight, pyramid_weight;

      feature_weight = soft_assign(descriptors.row(feature_num));

      std::transform(image_histogram.begin(), image_histogram.end(), // input 1
            normalized_weights.begin<double>(),          // input 2
            image_histogram.begin(),                     // output
            std::plus<double>());                        // operator
   }

   cv::normalize(image_histogram, image_histogram, image_histogram.size(), cv::NORM_L1);
   return image_histogram;
}

