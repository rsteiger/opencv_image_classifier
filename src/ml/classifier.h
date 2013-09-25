#pragma once

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include <vector>

#include "../cv/serialize_cvmat.h"

/**
 * This is a generic classifier, currently uses k-means
 */
struct classifier {

   struct settings {
      int neighbors;

      friend class boost::serialization::access;
      template<class archive>
      void serialize(archive &ar, const unsigned int version) {
         ar &neighbors;
      }

      settings() : neighbors(5) { }
   };

   protected:
   settings my_settings;

   cv::KNearest nearest_neighbors;

   cv::Mat samples;
   cv::Mat responses;

   void train();

   friend class boost::serialization::access;
   template<class archive>
   void serialize(archive &ar, const unsigned int version) {
      ar &my_settings;
      ar &samples;
      ar &responses;
      if (archive::is_loading::value) {
         train();
      }
   }
    
   public:
   void set_settings(const settings &s) { 
      my_settings = s;
      train();
   }

   void train(const cv::Mat &s, const cv::Mat &r) {
      samples = s;
      responses = r;
      train();
   }

   float classify(const std::vector<double> &vector) const;
   std::vector<float> classify(const cv::Mat &samples) const;
};

struct classifier_factory {
   cv::Mat samples;
   cv::Mat responses;
   
   void add_feature_vector(const std::vector<double> &vector, float response);
   
   classifier create_classifier(const classifier::settings &s = classifier::settings()) {
      classifier cls;
      cls.set_settings(s);
      cls.train(samples, responses);
      return cls;
   }
};
