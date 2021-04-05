import cvxpy as cp
import numpy as np

# Generate data.
m = 20
n = 15
dim_latent = 10
size_corpus = 100

np.random.seed(1)
h_test = np.random.randn(dim_latent)
corpus = np.random.randn(dim_latent, size_corpus)

# Define and solve the CVXPY problem.
w = cp.Variable(size_corpus)
cost = cp.sum_squares(corpus @ w - h_test)
obj = cp.Minimize(cost)
constr1 = np.ones(size_corpus) @ w == 1
constr2 = w >= 0
constr = [constr1, constr2]
prob = cp.Problem(cp.Minimize(cost), constr)
prob.solve(verbose=True)

# Print result.
print("\nThe optimal value is", prob.value)
print("The optimal w is")
print(w.value)
print("The norm of the residual is ", cp.norm(corpus @ w - h_test, p=2).value)

w_sol = np.array(w.value)
print(np.sum(w_sol))
print(w_sol.shape)
print(f'The residual fraction is {((corpus @ w_sol - h_test)**2).sum() / (h_test**2).sum()}')