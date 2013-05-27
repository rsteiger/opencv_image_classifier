/**
 * File Utils
 *
 * Richie Steigerwald
 *
 * Copyright 2013 Richie Steigerwald <richie.steigerwald@gmail.com>
 * This work is free. You can redistribute it and/or modify it under the
 * terms of the Do What The Fuck You Want To Public License, Version 2,
 * as published by Sam Hocevar. See http://www.wtfpl.net/ for more details.
 */


#pragma once

#include <string>
#include <list>

#include <boost/filesystem.hpp>

// Recursively get all of the files in the directory
std::list<std::string> get_files_recursive(const std::string &dir, 
 const std::string &extension) {
   std::list<std::string> file_names;
   if (!boost::filesystem::exists(dir)) {
      throw std::runtime_error(std::string("Path not found: ")+dir);
   }
   boost::filesystem::directory_iterator end_itr;
   for (boost::filesystem::directory_iterator itr(dir); itr != end_itr; itr++) {
      if (boost::filesystem::is_directory(itr->status())) {
         // Search subdirectories for images
         std::list<std::string> nested_files = 
          get_files_recursive(itr->path().string(), extension);
         file_names.insert(file_names.end(), nested_files.begin(), nested_files.end());
      } else {
         // Add files if they match the extensions in the list
         if (boost::filesystem::extension(itr->path()).compare(extension) == 0) {
            file_names.push_back(itr->path().string());
         }
      }
   }
   return file_names;
}

