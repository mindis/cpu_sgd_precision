import matplotlib.pyplot as plt
import numpy as np
h_c_t,h_c_l = np.loadtxt('hogwild_char_e_l.txt', delimiter=',', unpack=True)
plt.plot(h_c_t, h_c_l, 'g-', label="Hogwild Char",   linewidth=1)

h_s_t,h_s_l = np.loadtxt('hogwild_short_e_l.txt', delimiter=',', unpack=True)
plt.plot(h_s_t, h_s_l, 'g--', label="Hogwild Short",   linewidth=1)

h_f_t,h_f_l = np.loadtxt('hogwild_fp_e_l.txt', delimiter=',', unpack=True)
plt.plot(h_f_t, h_f_l, 'g-.', label="Hogwild FP",   linewidth=1)


m_c_t,m_c_l = np.loadtxt('modelsync_char_e_l.txt', delimiter=',', unpack=True)
plt.plot(m_c_t, m_c_l, 'b-', label="ModelSync Char",   linewidth=1)

m_s_t,m_s_l = np.loadtxt('modelsync_short_e_l.txt', delimiter=',', unpack=True)
plt.plot(m_s_t, m_s_l, 'b--', label="ModelSync Short",   linewidth=1)

m_f_t,m_f_l = np.loadtxt('modelsync_fp_e_l.txt', delimiter=',', unpack=True)
plt.plot(m_f_t, m_f_l, 'b-.', label="ModelSync FP",   linewidth=1)


'''
plt.plot(z,w, 'b-', label="ModelSync", linewidth=2)
'''
plt.axis([0, 40, 0, 0.28])

plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.legend()
plt.savefig("hogwild_modelsim_e_t.pdf")
plt.show()
