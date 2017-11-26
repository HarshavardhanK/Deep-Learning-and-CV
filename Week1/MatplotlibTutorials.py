import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf

plt.plot([1,2,3,4], [1, 4, 9, 16], 'ro')
plt.axis([0, 6, 0, 20])
plt.ylabel('Awesome numbers')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

x = [1,2,3,4]
y = [3,5,7,10] # 10, not 9, so the fit isn't perfect

fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit)
# fit_fn is now a function which takes in x and returns an estimate for y

plt.plot(x,y, 'yo', x, fit_fn(x), '--k')
plt.xlim(0, 5)
plt.ylim(0, 12)
plt.show()
