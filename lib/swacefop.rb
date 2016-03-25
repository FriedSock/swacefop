#!/usr/bin/env ruby
require 'bundler'
Bundler.setup

require 'opencv'
require 'RMagick'
require 'csv'
require 'matrix'

include OpenCV

SWACEFOP_ROOT = File.join(File.dirname(__FILE__), '../')

imagesnap_path = File.join(SWACEFOP_ROOT, 'vendor', 'imagesnap')
image_name = "webcam.jpg"
call_str = "#{imagesnap_path} images/#{image_name}"
puts "Taking image with command #{call_str}"
#system(call_str)

puts "Running Predictor"
python_command = "python lib/predictor.py images/#{image_name}"
system(python_command)


# Load an image
puts "Loading image"
img = IplImage.load('images/webcam.jpg')

landmarks1 = []
landmarks2 = []
CSV.foreach("csvs/landmarks1.csv") do |row|
  landmarks1 << CvPoint.new(*row.map(&:to_f))
end

CSV.foreach("csvs/landmarks2.csv") do |row|
  landmarks2 << CvPoint.new(*row.map(&:to_f))
end

m1 = []
CSV.foreach("csvs/m1.csv") do |row|
  m1 << row.map(&:to_f)
end
m1 = Matrix[*m1]

m2 = []
CSV.foreach("csvs/m2.csv") do |row|
  m2 << row.map(&:to_f)
end
m2 = Matrix[*m2]

puts "Drawing"

# Load the cascade for detecting faces
#detector = CvHaarClassifierCascade::load('vendor/haarcascade_frontalface_alt.xml.gz')


(landmarks1[0,27] + landmarks2[0,27]).each do |point|
  img.circle!(point, 2, color: CvColor::Red)
end

img.save("images/detected2.jpg")

transpoints1 = landmarks1.map do |point|
  pointm = Matrix[[point.x, point.y, 0]]
  transformed = pointm * m1
  x = transformed[0,0]
  y = transformed[0,1]
  CvPoint.new x,y
end

transpoints2 = landmarks2.map do |point|
  pointm = Matrix[[point.x, point.y, 0]]
  transformed = pointm * m2
  x = transformed[0,0]
  y = transformed[0,1]
  CvPoint.new x,y
end

c1x = landmarks1.map { |l| l.x }.reduce(:+) / 68
c1y = landmarks1.map { |l| l.y }.reduce(:+) / 68

ct1x = transpoints1.map { |l| l.x }.reduce(:+) / 68
ct1y = transpoints1.map { |l| l.y }.reduce(:+) / 68

cdx = c1x - ct1x
cdy = c1y - ct1y

transpoints1.map! do |point|
  point.x = point.x + cdx
  point.y = point.y + cdy
  point
end

c2x = landmarks2.map { |l| l.x }.reduce(:+) / 68
c2y = landmarks2.map { |l| l.y }.reduce(:+) / 68

ct2x = transpoints2.map { |l| l.x }.reduce(:+) / 68
ct2y = transpoints2.map { |l| l.y }.reduce(:+) / 68

cdx = c2x - ct2x
cdy = c2y - ct2y

transpoints2.map! do |point|
  point.x = point.x + cdx
  point.y = point.y + cdy
  point
end

cvmat = CvMat.new(2, 3)

(0..1).each do |x|
  (0..2).each do |y|
    require 'pry'; binding.pry
    cvmat[x,y] =  CvScalar.new(m1[x,y])
  end
end

require 'pry'; binding.pry
cvmat.warp_affine(img)



#img = IplImage.load('images/webcam.jpg')

(transpoints1 + transpoints2).each do |point|
  img.circle!(point, 2, color: CvColor::Green)
end

# Create a window and show the image
window = GUI::Window.new('Face Detection')
window.show(img)
GUI::wait_key








# 1. make a mask for each of the faces
# 2.
