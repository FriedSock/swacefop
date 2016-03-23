#!/usr/bin/env ruby
require 'bundler'
Bundler.setup

require 'opencv'
require 'RMagick'



include OpenCV

SWACEFOP_ROOT = File.join(File.dirname(__FILE__), '../')

imagesnap_path = File.join(SWACEFOP_ROOT, 'vendor', 'imagesnap')
image_name = "webcam.jpg"
call_str = "#{imagesnap_path} images/#{image_name}"
puts "Taking image with command #{call_str}"
system(call_str)

# Load an image
puts "Loading image"
img = IplImage.load('images/webcam.jpg')

# Load the cascade for detecting faces
detector = CvHaarClassifierCascade::load('vendor/haarcascade_frontalface_alt.xml.gz')

# Detect faces and draw rectangles around them
detector.detect_objects(img) { |rect|
  img.rectangle!(rect.top_left, rect.bottom_right, :color => CvColor::Red)
}

img.save("images/detected.jpg")

# Create a window and show the image
window = GUI::Window.new('Face Detection')
window.show(img)
GUI::wait_key

