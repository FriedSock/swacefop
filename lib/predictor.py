import cv2
import numpy
import dlib
import sys

def get_landmarks(im, index):
    rects = detector(im, 1)

    if len(rects) > 2:
        raise TooManyFaces
    if len(rects) < 0:
        raise NotEnoughFaces

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[index]).parts()])

def read_im_and_landmarks(fname, index):
  im = cv2.imread(image_path, cv2.IMREAD_COLOR)
  im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                           im.shape[0] * SCALE_FACTOR))
  s = get_landmarks(im, index)

  return im, s

def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])


PREDICTOR_PATH = "vendor/shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1

if __name__ == "__main__":
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(PREDICTOR_PATH)
  image_path =  sys.argv[1]

  im1, landmarks1 = read_im_and_landmarks(image_path, 0)
  im2, landmarks2 = read_im_and_landmarks(image_path, 1)

  numpy.savetxt("csvs/landmarks1.csv", landmarks1, delimiter=",")
  numpy.savetxt("csvs/landmarks2.csv", landmarks2, delimiter=",")
  M1 = transformation_from_points(landmarks1, landmarks2)
  M2 = transformation_from_points(landmarks2, landmarks1)
  numpy.savetxt("csvs/m1.csv", M1, delimiter=",")
  numpy.savetxt("csvs/m2.csv", M2, delimiter=",")

