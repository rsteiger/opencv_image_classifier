#pragma once

#include <list>
#include <vector>

#include <opencv2/core/core.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>

#include "serialize_cvmat.h"
#include "visual_vocabulary.h"

class bag_of_features {

   public:
      /**
       * These are options for our bag of words. These settings change how a
       * feature vector is computed.
       */
      struct settings {
         // expected squared distance between visual word and associated visual
         // vocab used for soft feature assignment
         float kernel_distance_squared = 0.25f;  

         // whether or not soft feature assignment is used
         bool soft_kernel = false;

         // how many spatial pyramid levels are used
         int spatial_pyramid_depth = 1;

         friend class boost::serialization::access;
         template<class archive>
         void serialize(archive &ar, const unsigned int version) {
            ar &kernel_distance_squared;
            ar &soft_kernel;
            ar &spatial_pyramid_depth;
         }
      };
   protected:
      settings settings;
      visual_vocabulary vocabulary;

      friend class boost::serialization::access;
      template<class archive>
      void serialize(archive &ar, const unsigned int version) {
         ar &settings;
         ar &vocabulary;
      }

      // computes assignment of descriptor to visual vocabulary
      cv::Mat soft_assign(const cv::Mat &point) const;
      cv::Mat hard_assign(const cv::Mat &point) const;

      // gets the size of the spatial pyramid representation
      int pyramid_size(int depth) const { return ((1 << (2 * depth)) - 1) / 3; }
      int pyramid_level_size(int depth) const { return 1 << depth; }

      // gets the weight of a spatial pyramid level
      float pyramid_weight(int depth) const { 
         if (!depth) return pyramid_weight(1);
         return 1.0 / (1 << (settings.spatial_pyramid_depth - depth + 1));
      }

   public:
      void set_vocabulary(const visual_vocabulary &vv) { vocabulary = vv; }
      void set_settings(const struct settings &s) { settings = s; }

      // Computes the feature vector for a set of features
      cv::Mat mat_feature_vector(const std::vector<cv::KeyPoint>
            &features, const cv::Mat &descriptors) const;
      std::vector<double> feature_vector(const std::vector<cv::KeyPoint>
            &features, const cv::Mat &descriptors) const;
};
