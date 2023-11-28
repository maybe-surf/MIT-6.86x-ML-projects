#%%
import sys
sys.path.append("C:/Users/serge/codez/mit/netflix")
import numpy as np
import kmeans
import common
import naive_em
import em
#%%
X = np.loadtxt("C:/Users/serge/codez/mit/netflix/toy_data.txt")

#%%kmeans
costs = []
mixes = []
posts = []
for K in [1, 2, 3, 4]:
    costs_temp = []
    mixes_temp = []
    posts_temp = []
    for seed in [0, 1, 2, 3, 4]:
        mix, post = common.init(X, K, seed)
        mix, post, cost = kmeans.run(X, mix, post)
        mixes_temp.append(mix)
        posts_temp.append(post)
        costs_temp.append(cost)
    min_cost_index = costs_temp.index(min(costs_temp))
    costs.append(costs_temp[min_cost_index])
    mixes.append(mixes_temp[min_cost_index])
    posts.append(posts_temp[min_cost_index])
for i in range(4):
    common.plot(X, mixes[i], posts[i], str(i))
        
#%%emgaus
costs = []
mixes = []
posts = []
for K in [1, 2, 3, 4]:
    costs_temp = []
    mixes_temp = []
    posts_temp = []
    for seed in [0, 1, 2, 3, 4]:
        mix, post = common.init(X, K, seed)
        mix, post, cost = naive_em.run(X, mix, post)
        mixes_temp.append(mix)
        posts_temp.append(post)
        costs_temp.append(cost)
    max_costs_index = costs_temp.index(max(costs_temp))
    costs.append(costs_temp[max_costs_index])
    mixes.append(mixes_temp[max_costs_index])
    posts.append(posts_temp[max_costs_index])
for i in range(4):
    common.plot(X, mixes[i], posts[i], "em" + str(i))

#%%BIc
bics = []
for i in range(4):
    bics.append(common.bic(X, mixes[i], costs[i]))
ind = bics.index(max(bics))
print(ind)
print(bics[ind])    

#%%mstep
X = [[0.69775695, 0.25128852],
 [0.81294951, 0.42590846],
 [0.28426226, 0.98823408],
 [0.9424375,  0.44178856],
 [0.18769823, 0.65756261],
 [0.00695086, 0.97902624],
 [0.3279987,  0.81230274],
 [0.49032846, 0.78744149],
 [0.73871246, 0.99751101],
 [0.29015913, 0.74610902],
 [0.03659566, 0.37209743],
 [0.99866155, 0.07639136],
 [0.71720314, 0.01857465],
 [0.68741343, 0.39315548],
 [0.75482165, 0.87478627]]

post =[[0.29730309, 0.35673471, 0.27907528, 0.06688691],
 [0.25632076, 0.24233462, 0.05039407, 0.45095055],
 [0.21076853, 0.23833666, 0.27937076, 0.27152406],
 [0.35848699, 0.00557544, 0.10275452, 0.53318305],
 [0.29981593, 0.24477302, 0.1542003,  0.30121076],
 [0.33423931, 0.25496467, 0.30025737, 0.11053865],
 [0.25485535, 0.19869058, 0.40298031, 0.14347376],
 [0.09947556, 0.21524719, 0.32733954, 0.35793771],
 [0.62044633, 0.12353661, 0.15896639, 0.09705067],
 [0.33904834, 0.02146381, 0.23018124, 0.40930661],
 [0.28546199, 0.16976964, 0.20020989, 0.34455848],
 [0.11553977, 0.28701705, 0.26614167, 0.3313015 ],
 [0.02098461, 0.34120839, 0.35690937, 0.28089763],
 [0.04349664, 0.31177397, 0.36439621, 0.28033317],
 [0.34327427, 0.12655833, 0.42648533, 0.10368206]]

X = np.array(X)
post = np.array(post)
#%%
gmix = common.GaussianMixture(0, 0, 0)
#%%
res = em.mstep(X, post, gmix)

#%%em k1 12
costs = []
mixes = []
posts = []
for K in [1, 12]:
    print(K)
    costs_temp = []
    mixes_temp = []
    posts_temp = []
    for seed in [0, 1, 2, 3, 4]:
        print(seed)
        mix, post = common.init(X, K, seed)
        mix, post, cost = em.run(X, mix, post)
        mixes_temp.append(mix)
        posts_temp.append(post)
        costs_temp.append(cost)
    max_costs_index = costs_temp.index(max(costs_temp))
    costs.append(costs_temp[max_costs_index])
    mixes.append(mixes_temp[max_costs_index])
    posts.append(posts_temp[max_costs_index])

#%%
for i in range(4):
    common.plot(X, mixes[i], posts[i], "em" + str(i))
#%%
X = [[0.85794562, 0.84725174],
 [0.6235637,  0.38438171],
 [0.29753461, 0.05671298],
 [0.,         0.47766512],
 [0.,         0.        ],
 [0.3927848,  0.        ],
 [0.,         0.64817187],
 [0.36824154, 0.        ],
 [0.,         0.87008726],
 [0.47360805, 0.        ],
 [0.,         0.        ],
 [0.,         0.        ],
 [0.53737323, 0.75861562],
 [0.10590761, 0.        ],
 [0.18633234, 0.        ]]

Mu = [[0.6235637,  0.38438171],
 [0.3927848,  0.        ],
 [0.,         0.        ],
 [0.,         0.87008726],
 [0.36824154, 0.        ],
 [0.10590761, 0.        ]]
Var = [0.16865269, 0.14023295, 0.1637321,  0.3077471,  0.13718238, 0.14220473]
P = [0.1680912,  0.15835331, 0.21384187, 0.14223565, 0.14295074, 0.17452722]

X = np.array(X)
Mu = np.array(Mu)
Var = np.array(Var)
P = np.array(P)

mix2 = common.GaussianMixture(Mu, Var, P)

#%%
X_pred = em.fill_matrix(X, mix2)














