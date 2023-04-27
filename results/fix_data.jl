using Serialization
using DelimitedFiles

pwd()
cd("C:/Users/aleen962/Dropbox/PhD/Forskning/Power exponential dist extremes/EpdExtremes/results")
pwd()

dat = deserialize("dim-5_lambda-0.5_nu-1.0_beta-0.2.dat")
writedlm("dim-5_lambda-0.5_nu-1.0_beta-0.2.csv", thetaEstMle, ',')



dimension = [5, 10, 15]
λ = [0.5, 1.0]
ν = 1.0
β = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
l = 1
join(["dim-", dimension, "_", "lambda-", λ, "_", "nu-", ν, "_", "beta-", β[l], ".dat"], "")

dat = deserialize(join(["dim-", dimension[1], "_", "lambda-", λ[1], "_", "nu-", ν, "_", "beta-", β[1], ".dat"], ""))
writedlm(join(["dim-", dimension[1], "_", "lambda-", λ[1], "_", "nu-", ν, "_", "beta-", β[1], ".csv"], ""), dat, ',')

for i in eachindex(dimension)
    for j in eachindex(λ)
        for l in eachindex(β)
            dat = deserialize(join(["dim-", dimension[i], "_", "lambda-", λ[j], "_", "nu-", ν, "_", "beta-", β[l], ".dat"], ""))
            writedlm(join(["dim", dimension[i], "_", "lambda", λ[j], "_", "nu", ν, "_", "beta", β[l], ".csv"], ""), dat, ',')
        end
    end
end