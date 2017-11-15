import matplotlib.pyplot as plt
import numpy as np

e, h_l_1, h_l_2, h_l_3, h_l_4, h_l_5, h_l_6, h_l_7, h_l_8, h_l_12, h_l_16  = np.loadtxt('hogwild_bitweaving_e_l.txt', delimiter=',', unpack=True)

plt.plot(e, h_l_1, 'g-',  label="1-bit",    linewidth=1)

plt.plot(e, h_l_2, 'b-',  label="2-bit",    linewidth=1)

plt.plot(e, h_l_3, 'r-',  label="3-bit",    linewidth=1)

plt.plot(e, h_l_4, 'c-',  label="4-bit",    linewidth=1)

plt.plot(e, h_l_5, 'm-',  label="5-bit",    linewidth=1)

plt.plot(e, h_l_6, 'y-',  label="6-bit",    linewidth=1)

plt.plot(e, h_l_7, 'k-',  label="7-bit",    linewidth=1)

plt.plot(e, h_l_8, 'r--',  label="8-bit",    linewidth=1)

plt.plot(e, h_l_12, 'g--', label="12-bit",   linewidth=1)

plt.plot(e, h_l_16, 'b--', label="16-bit",   linewidth=1)

'''
'''
plt.axis([0, 40, 0, 0.28])

plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.legend()
plt.savefig("hogwild_bitweaving_e_t.pdf")
plt.show()
