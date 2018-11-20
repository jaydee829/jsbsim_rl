import sys
sys.path.append("baselines")

from baselines.common import plot_util as pu
results = pu.load_results('/home/jsbsim/logs')

import matplotlib.pyplot as plt
import numpy as np
r = results[0]
plt.plot(np.cumsum(r.monitor.l), r.monitor.r, label='raw score')
plt.plot(np.cumsum(r.monitor.l), pu.smooth(r.monitor.r, radius=10), label='smoothed score')
plt.legend()
plt.show()
