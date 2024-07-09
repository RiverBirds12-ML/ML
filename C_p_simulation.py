import matplotlib.pyplot as plt
import numpy as np

tau = 10
ica = -0.1
c_s_0 = 0.01 #cs
c_s_inf = c_s_0 - ica*tau

# cs = 0.01
# -i * tau = 0.01
# i = -10 

N = 10000

t_max = 10
dt = 0.1
time_steps = np.arange(0, t_max+dt, dt)

state = np.zeros(N)
N_I = []
N_Imfm = []

for t in time_steps:
    random_array = np.random.rand(N)
    cpt = c_s_inf + (c_s_0 - c_s_inf) * np.exp(-t/tau)
    for n in range (1, N):
        if random_array[n] < cpt*dt and state[n] == 0:
            state[n] = 1
    N_I.append(sum(state))
    N_Imfm.append( N - N*( np.exp(ica*tau*tau*np.exp(-t/tau)  - (c_s_0 - ica*tau)*t - ica*tau*tau) ) )


plt.figure(figsize=(9,9))
plt.plot(time_steps,N_I)
plt.plot(time_steps,N_Imfm)
plt.xlabel('t')
plt.ylabel('N_I')
plt.legend(['Markov', 'Analytical'])
plt.show()


# Simulation 2 : cp is a constant (c_p_inf)
# state = np.zeros(N)
# N_I = []
# N_Imfm = []
# for t in time_steps:
#     random_array = np.random.rand(N)
#     cpt = c_s_inf
#     for n in range (1, N):
#         if random_array[n] < cpt*dt and state[n] == 0:
#             state[n] = 1
#     N_I.append(sum(state))
#     N_Imfm.append( N - N*( np.exp( -cpt*t ) ) )


# plt.plot(time_steps,N_I)
# plt.plot(time_steps,N_Imfm)
# plt.legend(['Markov', 'Analytical', 'Instant Markov', 'Instant Analytical'])
# plt.show()