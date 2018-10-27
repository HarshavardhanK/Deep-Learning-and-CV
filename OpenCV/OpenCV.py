import cv2
import numpy as np
import matplotlib.pyplot as plt

file = "/Users/harshavardhank/Downloads/iphones.JPG"

img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
print(img.size)

#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.show()
