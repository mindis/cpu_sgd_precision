import matplotlib.pyplot as plt
import numpy as np
h_c_t_1,h_c_l_1, h_c_t_2,h_c_l_2, h_c_t_3,h_c_l_3, h_c_t_4,h_c_l_4, h_c_t_5,h_c_l_5, h_c_t_6,h_c_l_6, h_c_t_7,h_c_l_7, h_c_t_8,h_c_l_8, h_c_t_12,h_c_l_12, h_c_t_16,h_c_l_16 = np.loadtxt('hogwild_bitweaving_t_l.txt', delimiter=',', unpack=True)

plt.plot(h_c_t_1, h_c_l_1, 'g-', label="1-bit",   linewidth=1)
plt.plot(h_c_t_2, h_c_l_2, 'b-', label="2-bit",   linewidth=1)
plt.plot(h_c_t_3, h_c_l_3, 'r-', label="3-bit",   linewidth=1)
plt.plot(h_c_t_4, h_c_l_4, 'c-', label="4-bit",   linewidth=1)
plt.plot(h_c_t_5, h_c_l_5, 'm-', label="5-bit",   linewidth=1)
plt.plot(h_c_t_6, h_c_l_6, 'y-', label="6-bit",   linewidth=1)
plt.plot(h_c_t_7, h_c_l_7, 'k-', label="7-bit",   linewidth=1)
plt.plot(h_c_t_8, h_c_l_8, 'r--', label="8-bit",   linewidth=1)
plt.plot(h_c_t_12, h_c_l_12, 'g--', label="12-bit",   linewidth=1)
plt.plot(h_c_t_16, h_c_l_16, 'b--', label="16-bit",   linewidth=1)

'''
h_f_t,h_f_l = np.loadtxt('hogwild_fp_t_l.txt', delimiter=',', unpack=True)
plt.plot(h_f_t, h_f_l, 'r^--', label="32-bit FP",   linewidth=2)

'''


'''
plt.plot(z,w, 'b-', label="ModelSync", linewidth=2)

plt.axis([0, 2, 0, 0.2])
'''
plt.axis([0, 1, 0, 0.25])

plt.xlabel('Time (s)')
plt.ylabel('Training loss')
plt.legend()
plt.savefig("hogwild_bitweaving_l_t.pdf")
plt.show()

