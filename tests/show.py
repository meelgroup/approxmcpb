import matplotlib.pyplot as plt
import pickle
import sys

assert(len(sys.argv) > 1)
print(sys.argv[1])
fig = pickle.load(open(sys.argv[1], 'rb'))
plt.show()
