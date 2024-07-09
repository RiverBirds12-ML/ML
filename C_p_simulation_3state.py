import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

alpha = 0.01
# gamma = Ae^{-t/tau} + B
tau = 50
A = 0.01
B = 0.02

N = 10000

tmax = 100
dt = 0.1
time_steps = np.arange(0, tmax+dt, dt)

state = np.zeros(N)
gamma = (A+B)*np.ones(N)
d_gamma = np.zeros(N)
N_C = []
N_C_pred = []
N_O_rec = []
N_C_rec = []
N_I_rec = []
t_rec = []
Sumation = []
Oo = 0.1*N
state[:int(Oo)] = 1
N_O_sim = Oo
N_C_sim = N - Oo
N_I_sim = 0

N_O_sim2 = Oo
N_C_sim2 = N - Oo
N_I_sim2 = 0
N_O_rec2 = []
N_C_rec2 = []
N_I_rec2 = []

O_term_rec = []



for t in time_steps:
    random_array = np.random.rand(N)
    # calculate numerical integral
    for n in range(1,N):
        if state[n] == 1:
            d_gamma[n] = dt * (B-gamma[n])/tau

        if state[n] == 0 and random_array[n] < alpha*dt:
            state[n] = 1
        elif state[n] == 1 and random_array[n] < gamma[n]*dt:
            state[n] = 2
    gamma += d_gamma
    N_C.append(np.sum(state == 1))

    dNC = dt*(-alpha*N_C_sim)
    #dNO = dt*(alpha*N_C_sim - dt*sum(alpha*N_C_rec)*(A*np.exp(-(t - t_rec)/tau)+B)*(-A*(np.exp(-t/tau)-np.exp(-t_rec/tau)) + B*(t-t_rec) ) - (A*np.exp(-t/tau)+B)*Oo*np.exp( -sum( A* (np.exp(-(t - t_rec)/tau)+B )   ) ) )
    array1 = alpha*np.array(N_C_rec)
    array2 = A * np.exp( -(t - np.array(t_rec)) / tau ) + B
    #array3 = np.exp(A*tau*(np.exp(-(t-np.array(t_rec))/tau)-1) + B*(t-np.array(t_rec)) )
    array3 = np.exp(A*tau*(np.exp(-(t-np.array(t_rec))/tau)-1) - B*(t-np.array(t_rec)) )
    #dNO = dt*(alpha*N_C_sim - dt*sum (  (alpha*np.array(N_C_rec))*(A*np.exp(-(t - np.array(t_rec))/tau)+B)*(A*(np.exp(-t/tau)-np.exp(-np.array(t_rec)/tau)) - B*(t-np.array(t_rec)) ) ) - (A*np.exp(-t/tau)+B)*Oo*np.exp( -A*tau*np.exp(-t/tau) + B*t + A*tau ) )
    #dNO = dt*(alpha*N_C_sim - dt*np.sum(array1*array2*array3) )
    dNO = dt*(alpha*N_C_sim - dt*np.sum(array1*array2*array3) - (A*np.exp(-t/tau)+B)*Oo*np.exp( A*tau*np.exp(-t/tau) - B*t - A*tau ) )
    O_term_rec.append(dt*np.sum(array1*array2*array3)/B)
    dNI = -dNC-dNO

    N_C_sim += dNC
    N_O_sim += dNO
    N_I_sim += dNI

    N_O_rec.append(N_O_sim)
    N_I_rec.append(N_I_sim)
    N_C_rec.append(N_C_sim)
    t_rec.append(t)

    # Mean-Field
    dNC2 = dt*(-alpha*N_C_sim2)
    dNO2 = dt*(alpha*N_C_sim2 - B*N_O_sim2)
    dNI2 = -dNC2-dNO2
    N_C_sim2 += dNC2
    N_O_sim2 += dNO2
    N_I_sim2 += dNI2
    N_O_rec2.append(N_O_sim2)
    N_I_rec2.append(N_I_sim2)
    N_C_rec2.append(N_C_sim2)

    Sumation.append(np.sum(array1*array2*array3))

    if t % 10 == 0:
        print(t)

plt.figure(figsize=(9,9))
plt.plot(t_rec,N_C)
plt.plot(t_rec,N_O_rec)
plt.plot(t_rec,N_O_rec2)
#plt.plot(t_rec,O_term_rec)
plt.xlabel('t')
plt.ylabel('N_O')
plt.legend(['Markov', 'Analytical', 'Mean Field', 'O_term_rec'])
#plt.legend(['Markov', 'Analytical', 'Mean Field', 'O_term_rec'])
plt.show()



# N_C_inst = []
# N_C_pred_inst = []

# A2 = 0
# state = np.zeros(N)
# for t in time_steps:
#     random_array = np.random.rand(N)
#     gamma_t = A2 * np.exp(-t/tau) + B
#     # calculate numerical integral
#     int_result = quad(integrand, 0, t, args=(A2, B, alpha, tau))
#     for n in range(1,N):
#         if state[n] == 0 and random_array[n] < alpha*dt:
#             state[n] = 1
#         elif state[n] == 1 and random_array[n] < gamma_t*dt:
#             state[n] = 2
#     N_C_inst.append(np.sum(state == 2))
#     N_C_pred_inst.append( N - N*np.exp(-alpha*t) - N* alpha*int_result[0]*np.exp(A2*tau*np.exp(-t/tau) - B*t ) )
#     if t % 10 == 0:
#         print(tmax + t)

# plt.plot(time_steps,N_C_inst)
# plt.plot(time_steps,N_C_pred_inst)
# plt.legend(['Markov', 'Analytical', 'instant Markov', 'instant Analytical'])
# plt.show()

    