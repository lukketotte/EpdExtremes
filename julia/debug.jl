include("./FFT.jl")
using .MepdCopula

# dC is way to low for high values of p compared to low

# dG1 behaves as expected
dG1([0.5 4.1], 0.3)
dG1([0.5 4.1], 0.9)

# dG behaves as expected
dG([2.5, -1.1], cor_mat, 0.4)
dG([2.5, -1.1], cor_mat, 0.95)

dC([0.01 0.01], cor_mat, 0.2)
dC([0.01 0.01], cor_mat, 0.5)
dC([0.01 0.01], cor_mat, 0.9)

qG1_val_02 = qG1([0.01 0.01], 0.2)
qG1_val_09 = qG1([0.01 0.01], 0.9)

log.(dG(qG1_val_02, cor_mat, 0.2))
log.(dG(qG1_val_09, cor_mat, 0.9))

log.(dG1(qG1_val_02, 0.2))
log.(dG1(qG1_val_09, 0.9))

log.(dG(qG1_val, Sigma, p)) .- sum(log.(dG1(qG1_val, p)), dims = 2)