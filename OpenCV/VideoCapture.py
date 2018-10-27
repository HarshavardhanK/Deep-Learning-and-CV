import cv2
import numpy as np

vc = cv2.VideoCapture(0)

rval, frame = vc.read()

while True:

  if frame is not None:
     cv2.imshow("preview", frame)
  rval, frame = vc.read()

  if cv2.waitKey(1) & 0xFF == ord('q'):
     break

vc.release()
cv2.destroyAllWindows()
