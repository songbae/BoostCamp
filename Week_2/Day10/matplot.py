import matplotlib.pyplot as plt
import numpy as np

fig =plt.figure()
fig.set_size_inches(10,5)
ax_1=fig.add_subplot(1,2,1)
ax_2=fig.add_subplot(1,2,2)
ax_1.plot(x_1,y_1,c='b')
ax_2.plot(x_2,y_2,c='g')
plt.show()
