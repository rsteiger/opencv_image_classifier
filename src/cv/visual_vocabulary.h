#pragma once

#include <opencv2/core/core.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "serialize_cvmat.h"

struct visual_vocabulary {
   /**
    * These are options for the visual vocabulary
    */
   struct settings {
      // number of words in the visual vocabulary
      int size = 500;

      friend class boost::serialization::access;
      template<class archive>
      void serialize(archive &ar, const unsigned int version) {
         ar &size;
      }
   };

   cv::Mat centroids;

   protected:
   settings my_settings;

   friend class boost::serialization::access;

   template<class archive>
   void serialize(archive &ar, const unsigned int version) {
      ar &my_settings;
      ar &centroids;
   }

   public:
   visual_vocabulary(const cv::Mat &descriptors, const settings &s);
   visual_vocabulary(const visual_vocabulary &v) { }
   visual_vocabulary() { }
};

struct visual_vocabulary_factory {

   // Add descriptors used to compute the cluster centers
   void add_descriptors(const cv::Mat &descriptors);

   // Compute the visual vocabulary
   visual_vocabulary compute_visual_vocabulary(const visual_vocabulary::settings &s = visual_vocabulary::settings()) 
         { return visual_vocabulary(descriptors, s); }

   protected:
      cv::Mat descriptors;
};
