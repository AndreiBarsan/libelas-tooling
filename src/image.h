/*
Copyright 2011. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of libelas.
Authors: Andreas Geiger

libelas is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or any later version.

libelas is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libelas; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/

// basic image I/O, based on Pedro Felzenszwalb's code

#ifndef IMAGE_H
#define IMAGE_H

#include <cstdlib>
#include <climits>
#include <cstring>
#include <iostream>
#include <fstream>

// use imRef to access image data.
#define imRef(im, x, y) (im->access[y][x])
  
// use imPtr to get pointer to image data.
#define imPtr(im, x, y) &(im->access[y][x])

#define BUF_SIZE 256

typedef unsigned char uchar;
typedef struct { uchar r, g, b; } rgb;

inline bool operator==(const rgb &a, const rgb &b) {
  return ((a.r == b.r) && (a.g == b.g) && (a.b == b.b));
}

// image class
template <class T> class image {
public:

  // create image
  image(const int width, const int height, const bool init = false);

  // create image from existing data (e.g., loaded via OpenCV).
  // Makes a copy of the given data. Supplied pointer must point to a grayscale
  // image.
  image(const int width, const int height, T *data);

  // delete image
  ~image();

  // init image
  void init(const T &val);

  // deep copy
  image<T> *copy() const;
  
  // get image width/height
  int width() const { return w; }
  int height() const { return h; }
  
  // image data
  T *data;
  
  // row pointers
  T **access;
  
private:
  int w, h;
};

template <class T> image<T>::image(const int width, const int height, const bool init_memory) {
  w = width;
  h = height;
  data = new T[w * h];  // allocate space for image data
  access = new T*[h];   // allocate space for row pointers
  
  // initialize row pointers
  for (int i = 0; i < h; i++) {
    access[i] = data + (i * w);
  }
  
  if (init_memory) {
    memset(data, 0, w * h * sizeof(T));
  }
}

template <class T> image<T>::image(const int width, const int height, T *data) {
  w = width;
  h = height;

  this->data = new T[w * h];
  memcpy(this->data, data, w * h);

  // initialize row pointers
  access = new T*[h];
  for (int i = 0; i < h; i++) {
    access[i] = data + (i * w);
  }
}


template <class T> image<T>::~image() {
  delete [] data; 
  delete [] access;
}

template <class T> void image<T>::init(const T &val) {
  T *ptr = imPtr(this, 0, 0);
  T *end = imPtr(this, w-1, h-1);
  while (ptr <= end)
    *ptr++ = val;
}


template <class T> image<T> *image<T>::copy() const {
  image<T> *im = new image<T>(w, h, false);
  memcpy(im->data, data, w * h * sizeof(T));
  return im;
}

class pnm_error {};

void pnm_read(std::ifstream &file, char *buf);

image<uchar> *loadPGM(const char *name);

void savePGM(image<uchar> *im, const char *name);

#endif
