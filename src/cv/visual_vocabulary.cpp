#include "visual_vocabulary.h"

/**
 * Compute the visual vocabulary from the list of descriptors
 */
visual_vocabulary::visual_vocabulary(const cv::Mat &descriptors, const
      visual_vocabulary::settings &s) : my_settings(s) { 

   cv::Mat labels;

   cv::kmeans(descriptors,              // Matrix of input samples, one row per sample
         my_settings.size,              // K -- Number of clusters to split the set by
         labels,                        // output integer array that stores the cluster indices
         cv::TermCriteria(              // When to stop the algorithm:
            CV_TERMCRIT_ITER |          //   after a desired number of iterations, or
            CV_TERMCRIT_EPS,            //   after a desired level of accuracy
            0.0001,                     // accuracy
            10000),                     // number of iterations
         5,                             // The number of times the algorithm is attempted
         cv::KMEANS_PP_CENTERS,         // Efficient initial labeling criteria
         centroids);                    // The output centers 
}


/**
 * Add descriptors for visual vocab compilation
 */
void visual_vocabulary_factory::add_descriptors(const cv::Mat &descriptors) {
   if (this->descriptors.data == NULL) {
      this->descriptors = descriptors;
   } else {
      assert(descriptors.cols == this->descriptors.cols);
      cv::vconcat(this->descriptors, descriptors, this->descriptors);
   }
}

