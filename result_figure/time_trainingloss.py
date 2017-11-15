import matplotlib.pyplot as plt
import numpy as np
h_c_t,h_c_l = np.loadtxt('hogwild_char_t_l.txt', delimiter=',', unpack=True)
plt.plot(h_c_t, h_c_l, 'g-', label="Hogwild Char",   linewidth=1)

h_s_t,h_s_l = np.loadtxt('hogwild_short_t_l.txt', delimiter=',', unpack=True)
plt.plot(h_s_t, h_s_l, 'g--', label="Hogwild Short",   linewidth=1)

h_f_t,h_f_l = np.loadtxt('hogwild_fp_t_l.txt', delimiter=',', unpack=True)
plt.plot(h_f_t, h_f_l, 'g-.', label="Hogwild FP",   linewidth=1)


m_c_t,m_c_l = np.loadtxt('modelsync_char_t_l.txt', delimiter=',', unpack=True)
plt.plot(m_c_t, m_c_l, 'b-', label="ModelSync Char",   linewidth=1)

m_s_t,m_s_l = np.loadtxt('modelsync_short_t_l.txt', delimiter=',', unpack=True)
plt.plot(m_s_t, m_s_l, 'b--', label="ModelSync Short",   linewidth=1)

m_f_t,m_f_l = np.loadtxt('modelsync_fp_t_l.txt', delimiter=',', unpack=True)
plt.plot(m_f_t, m_f_l, 'b-.', label="ModelSync FP",   linewidth=1)


'''
plt.plot(z,w, 'b-', label="ModelSync", linewidth=2)
'''
plt.axis([0, 6, 0, 0.1])

plt.xlabel('Time (s)')
plt.ylabel('Training loss')
plt.legend()
plt.savefig("hogwild_modelsim_l_t.pdf")
plt.show()

