
# coding: utf-8
import numpy as np
import scipy as sc
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


alpha = 0.7
delta = 0.15
n_k = 81
n_z = 11
m = 3
sigma = 0.2
rho = 0.7
#pick some value for r
r = 0.05

#1. Set up grids for capital stock and \log z

#set up shock grid
z_grid = np.zeros(n_z)
k_grid = np.zeros(n_k)

d = 2*(m*sigma/np.sqrt(1-rho**2))/(n_z - 1)
#discretize AR(1)
z_grid = np.linspace(-m*sigma/np.sqrt(1-rho**2), m*sigma/np.sqrt(1-rho**2), n_z)
    
#calculate transition matrix    
trans_matrix = np.zeros([n_z, n_z])
for i in range(n_z):
    for j in range(n_z):
        if j == 0:
            trans_matrix[i,j] = norm.cdf((z_grid[0] + d/2 - rho*z_grid[i])/sigma)
        elif j == n_z - 1:
            trans_matrix[i,n_z - 1] = 1 - norm.cdf((z_grid[n_z - 1] - d/2 - rho*z_grid[i])/sigma)
        else:
            trans_matrix[i,j] = norm.cdf((z_grid[j] + d/2 - rho*z_grid[i])/sigma) - norm.cdf((z_grid[j] - d/2 - rho*z_grid[i])/sigma)
        
    
#find steady state and set up geometric grid around it
def setup_k_grid(alpha):
    k_max = (np.exp(z_grid[-1])/delta)**(1/(1 - alpha))
    k_grid = np.zeros(n_k)
    for j in range(n_k):
        k_grid[n_k - j - 1] = k_max*(1 - delta)**j
    return k_grid



#do vfi using alpha and capital stock grid k_grid
def vfi(alpha, k_grid):
    stacked_k = np.tile(k_grid, (n_k, 1)).T

    payoff_matr = np.zeros([n_k*n_z, n_k])
    k_theta = stacked_k**alpha
    for i in range(n_z):
        payoff_matr[i*n_k:(i + 1)*n_k,:] = np.exp(z_grid[i])*k_theta
        payoff_matr[i*n_k:(i + 1)*n_k,:] = payoff_matr[i*n_k:(i + 1)*n_k,:] - (stacked_k.T - (1 - delta)*stacked_k)

    tol = 1e-8
    err = 1
    v_old = np.zeros([n_z,n_k])
    v_new = np.ones([n_z, n_k])
    beta = 1/(1 + r)
    c = beta/(1-beta)
    policy = np.zeros([n_z, n_k]).astype(int)

    while err > tol:
        q = trans_matrix.dot(v_old)
        for i in range(n_z):
            hat_v = payoff_matr[i*n_k:(i+1)*n_k,:] + beta*q[i,:].T.reshape([n_k, 1]).dot(np.ones([1, n_k])).T

            v_new[i,:] = np.max(hat_v, axis = 1)
        #McQueen
        low_b = c*np.min(v_new - v_old)
        high_b = c*np.max(v_new - v_old)
        v_new = v_new + 0.5*(low_b + high_b)
        err = np.abs(np.max(v_new - v_old))

        v_old = v_new.copy()
        
    #get policy
    for i in range(n_z):
        hat_v = payoff_matr[i*n_k:(i+1)*n_k,:] + beta*q[i,:].T.reshape([n_k, 1]).dot(np.ones([1, n_k])).T
        policy[i,:] = np.argmax(hat_v, axis = 1)
        
    return v_new, policy


#call these functions
k_grid = setup_k_grid(alpha)
v, policy = vfi(alpha, k_grid)


#print out grids
#capital stock grid
print(pd.DataFrame(k_grid[:7]).round(3).T.to_latex())
print(pd.DataFrame(k_grid[-4:]).round(2).T.to_latex())
#productivity grid
print(pd.DataFrame(z_grid).round(3).T.to_latex())



#get colors for the plots

path = r"C:\Users\elakkis\Box Sync\Teaching\FIN873\new_PS2\\"
plt.figure(figsize = (16,9))
for i in range(n_k):
    plt.plot(np.exp(z_grid), v[:,i])
plt.xlabel(r"$z$")
plt.ylabel("Value Function")
plt.savefig(path + "v_func_levels_z.png")



plt.figure(figsize = (16,9))
for i in range(n_z):
    plt.plot(k_grid, v[i,:])
plt.xlabel(r"$k$")
plt.ylabel("Value Function")
plt.savefig(path + "v_func_levels_k.png")



plt.figure(figsize = (16,9))
for i in range(n_k):
    plt.plot(np.exp(z_grid), k_grid[policy[:,i]])
plt.xlabel(r"$z$")
plt.ylabel("Policy Function")
plt.savefig(path + "p_func_levels_z.png")



plt.figure(figsize = (16,9))
for i in range(n_z):
    plt.plot(k_grid, k_grid[policy[i,:]])
plt.xlabel(r"$k$")
plt.ylabel("Policy Function")
plt.savefig(path + "p_func_levels_k.png")


#plot Policy and Value functions
X, Y = np.meshgrid(k_grid, np.exp(z_grid))
fig = plt.figure(figsize = (16,9))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, v, cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
plt.xlabel(r"$K$")
plt.ylabel(r"$Z$")
plt.title("Value Function")
ax.view_init(10, -100)
plt.savefig(path + "value3d.png")
plt.show()


X, Y = np.meshgrid(k_grid, np.exp(z_grid))
fig = plt.figure(figsize = (16,9))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, k_grid[policy], cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
plt.xlabel(r"$K$")
plt.ylabel(r"$Z$")
plt.title("Policy Function")
ax.view_init(10, -30)
plt.savefig(path + "policy3d.png")
plt.show()




#4. Do simulations
np.random.seed(1)
#this function simulates the model s times using policy and value function v
#returns path of capital (indexes from the grid)
#and the value function at each simulation
def simulate(s, n_drop, policy, v):
    values = np.zeros(n_drop + s)
    #store indexes in k
    k = np.zeros(n_drop + s).astype(int)
    #store indexes in z
    z = np.zeros(n_drop + s).astype(int)
    z_indexes = np.linspace(0, n_z -1, n_z).astype(int)

    k[0] = 0
    z[0] = 0
    values[0] = v[z[0], k[0]]

    for i in range(1,n_drop + s):
        z[i] = np.random.choice(a = z_indexes, p = trans_matrix[z[i-1],:])
        k[i] = policy[z[i], k[i-1]]
        values[i] = v[z[i], k[i-1]]

    return k[n_drop:], values[n_drop:]

#simulate
s = 100000
n_drop = 100
k, values = simulate(s, n_drop, policy, v)


#this function calculates investment and q from capital stock and v
def calc_data(k, values):
    inv_sim = (np.roll(k, -1)[:-1] - (1 - delta)*k[:-1])/k[:-1]
    q_sim = (values/k)[:-1]
    return inv_sim, q_sim

#compute simulated moments
inv_sim, q_sim = calc_data(k_grid[k], values)
var_i_sim = np.var(inv_sim)
av_q_sim = np.mean(q_sim)

print("Variance of simulated investment is ", var_i_sim)
print("Average of simulated Q is ", av_q_sim)



n_alpha = 50
alpha_grid = np.linspace(0.2, 0.8, n_alpha)

inv_values = np.zeros(n_alpha)
q_values = np.zeros(n_alpha)
i = 0
for alpha in alpha_grid:
    k, values = simulate(s, n_drop, policy, v)

    k_grid = setup_k_grid(alpha)
    v, policy = vfi(alpha, k_grid)
    
    #simulate the path
    k_sim, values_sim = simulate(s, n_drop, policy, v)
    
    #calculate moment on simulated data
    inv_sim, q_sim = calc_data(k_grid[k_sim], values_sim)
    inv_values[i] = np.var(inv_sim)
    q_values[i] = np.mean(q_sim)
    i = i + 1
    
fig = plt.figure(figsize = (16,9))

plt.plot(alpha_grid, inv_values)
plt.ylabel("Investment")
plt.xlabel(r"$\alpha$")
plt.savefig(r".inv_plot.png")
#5.
#read in the data
file = r"problemset1.txt"
data = np.loadtxt(file)
N = data.shape[0]

#fill in the variable vectors
gvkey = data[:,0]
year = data[:,1]
investment = data[:,2]
t_q = data[:,3]
n = data.shape[0]
#calculate moments on the data
var_i = np.var(investment)
mu_q = np.mean(t_q)
data_mom = np.array([var_i, mu_q])
print("Variance of investment in the data: ", var_i)
print("Mean Q in the data: ", mu_q)


#6. 

#this function sets up moment vector using real and simulated data
def g(data_mom, sim_data):
    result = np.zeros(2)
    result[0] = data_mom[0] - np.var(sim_data[0])
    result[1] = data_mom[1] - np.mean(sim_data[1])
    
    return result

#this function does SMM
def smm(data_mom, weight_matrix):
	#setup the grid for grid search
    n_alpha = 100
    alpha_grid = np.linspace(0.2, 0.8, n_alpha)
    obj_func_values = np.zeros(n_alpha)
    err = 10**10
    alpha_min = alpha_grid[0]
    #iterate over the grid
    for i in range(n_alpha):
        alpha = alpha_grid[i]
        #setup the grid and do VFI
        k_grid = setup_k_grid(alpha)
        v, policy = vfi(alpha, k_grid)
        
        #simulate the path
        k_sim, values_sim = simulate(s, n_drop, policy, v)
        
        #calculate moment on simulated data
        inv_sim, q_sim = calc_data(k_grid[k_sim], values_sim)
        sim_moment = g(data_mom, np.array([inv_sim, q_sim]))
        
        #get the objective function
        obj_func = sim_moment.T.dot(weight_matrix).dot(sim_moment)

        #see if it is the minimum
        if obj_func < err:
            alpha_min = alpha
            err = obj_func
        obj_func_values[i] = obj_func
    return alpha_min, err, obj_func_values

#this function computes gradient of g numerically using a difference scheme
def numerical_grad(alpha):
    np.random.seed(1)

    gs = np.zeros([2, 2])
    
    step = 0.01*alpha
    i = 0
    for alph in [alpha - step, alpha + step]:
        k_grid = setup_k_grid(alpha)
        #simulate the path
        v, policy = vfi(alph, k_grid)
        k_sim, values_sim = simulate(s, n_drop, policy, v)
        #calculate moment on simulated data
        inv_sim, q_sim = calc_data(k_grid[k_sim], values_sim)
        np.random.seed(1)
        sim_moment = g(data_mom, np.array([inv_sim, q_sim]))
        gs[:,i] = sim_moment.copy()
        i = i + 1
    res = (gs[:,1] - gs[:,0])/(2*step)
        
    return res



#this function calculates standard errors and statistics on moments as well as the gradient
def se_tstat(alpha_min, alpha_grid, W):
    grad = numerical_grad(alpha_min)
    var_alpha = (1 + N/s)/(grad.T.dot(W).dot(grad))
    se_alpha = np.sqrt(var_alpha/n)

    k_grid = setup_k_grid(alpha_min)
    v, policy = vfi(alpha_min, k_grid)

    #simulate the path
    k_sim, values_sim = simulate(s, n_drop, policy, v)

    #calculate moment on simulated data
    inv_sim, q_sim = calc_data(k_grid[k_sim], values_sim)
    sim_moment = g(data_mom, np.array([inv_sim, q_sim]))

    t_stat_g = sim_moment/np.sqrt(np.diag(np.linalg.inv(W)))*np.sqrt(1 + N/s)
    return se_alpha, t_stat_g, grad


#do SMM
alpha_min, err, obj_func_eye = smm(data_mom, np.eye(2))

n_alpha = 50
alpha_grid = np.linspace(0.2, 0.8, n_alpha)


#for identity matrix
W = np.eye(2)
se_alpha, t_stat, grad = se_tstat(alpha_min, alpha_grid, W)
print("Alpha is ", alpha_min)
print("Standard error : ", se_alpha)
print("t-stats on moments: ", t_stat)



#7. Estimate using unclustered weight matrix
def unclustered_weight_matrix(y, x):
    n = x.shape[0]
    
    #fill in influence function to the matrix
    phi_sigma2 = (y - np.mean(y))**2 - np.mean((y - np.mean(y))**2)
    phi_mu_x = x - np.mean(x)
    
    
    #stack up influence functions in the matrix
    capital_phi = np.zeros([n, 2])
    capital_phi = np.column_stack([phi_sigma2, phi_mu_x])
    
    #return matrix of influence functions as well as weighting matrix
    return capital_phi, np.linalg.inv(capital_phi.T.dot(capital_phi)/(n))


capital_phi, uncl_weight = unclustered_weight_matrix(investment, t_q)
alpha_min, err, obj_func_uncl = smm(data_mom, uncl_weight)



W = uncl_weight
se_alpha, t_stat, grad = se_tstat(alpha_min, alpha_grid, W)
print("Alpha is ", alpha_min)
print("Standard error : ", se_alpha)
print("t-stats on moments: ", t_stat)

#diagnostic
S = - (grad.T.dot(W).dot(grad))**(-1)*grad.T.dot(W)
Lambda = np.linalg.inv(W)
ident_1 = S[0]*np.sqrt(Lambda[0,0]/se_alpha)
ident_2 = S[1]*np.sqrt(Lambda[1,1]/se_alpha)
print("Sensitivities of estimate to moments: ", ident_1, ident_2)



#8. for Clustered weighting matrix
N = np.unique(gvkey).shape[0]
T = np.unique(year).shape[0]
k = capital_phi.shape[1]
    

def clustered_weight_matrix(x, y, capital_phi):
    vmx = np.zeros([k,k])
    for i in range(N):
        #select observations for firm i
        indexes = np.arange(i*T, i*T + T)
        #sum up the observations for this firm
        phi_new = np.sum(capital_phi[indexes,:], 0).reshape([1,2])
        #add the inner product to the variance
        vmx = vmx + phi_new.T.dot(phi_new)
    vmx = vmx/(N*T)
    #return the inverse of covariance matrix
    return np.linalg.inv(vmx)
clust_weight = clustered_weight_matrix(investment, t_q, capital_phi)



alpha_min, err, obj_func_clust = smm(data_mom, clust_weight)


W = clust_weight
se_alpha, t_stat, grad = se_tstat(alpha_min, alpha_grid, W)
print("Alpha is ", alpha_min)
print("Standard error : ", se_alpha)
print("t-stats on moments: ", t_stat)


#diagnostic
S = - (grad.T.dot(W).dot(grad))**(-1)*grad.T.dot(W)
Lambda = np.linalg.inv(W)

ident_1 = S[0]*np.sqrt(Lambda[0,0]/se_alpha)
ident_2 = S[1]*np.sqrt(Lambda[1,1]/se_alpha)
print("Sensitivities of estimate to moments: ", ident_1, ident_2)


#plot objectives
plt.figure(figsize = (16, 9))
plt.plot(alpha_grid, obj_func_eye, label = "Identity Matrix")
plt.plot(alpha_grid, obj_func_uncl, label = "Unclustered Optimal Weight Matrix")
plt.plot(alpha_grid, obj_func_clust, label = "Clustered Optimal Weight Matrix")
plt.title("GMM Objective Functions")
plt.xlabel(r"$\alpha$")
plt.legend(loc = "best")
plt.savefig(path + "obj.png")
plt.show()

