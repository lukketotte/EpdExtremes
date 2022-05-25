using Distributed, SharedArrays
@everywhere using LinearAlgebra, SpecialFunctions,  QuadGK, Cubature, Distributions, Roots
using Plots, StatsPlots, DataFrames, CSV

@everywhere g(x::Float64, p::Real) = exp(-abs(x)^p/2) / (π * gamma(1+1/p) * 2^(1/p))
@everywhere f(y::Float64, x::Float64, p::Real) = (y-x^2 + 1.e-20)^(-1/2) * g(y, p)
@everywhere depd(x::Float64, p::Real) = quadgk(y -> f(y, x, p), x^2, Inf)[1]
@everywhere pepd(x::Float64, p::Real) = 1/2 + quadgk(y -> depd(y, p), 0, x)[1]

## ignore this for now ##
K(x::Float64) = quadgk(t -> 0.5 * exp(-t-x^2/t)/t, 0, Inf)[1]
function dmvl(x::Real, y::Real, ρ::Real)
    a = √(2*(x^2 - 2*ρ*x*y + y^2)/(1-ρ^2))
    1/(π*√(1-ρ^2)) * K(a)
end

@everywhere function qrange(q::Real, p::Real)
    if p == 1
        q <= 0.95 ? range(0, 2, length =  1000) : range(1., 4, length =  2000)
    elseif p == 1/4
        if q < 0.9
            range(0, 79, length = 2000)
        elseif q <= 0.95
            range(78, 123, length = 2000)
        elseif q <= 0.96
            range(122, 139, length = 2000)
        elseif q <= 0.97
            range(138, 158.8, length = 2000)
        elseif q <= 0.98
            range(158.7, 189.5, length = 2000)
        elseif q <= 0.99
            range(189.4, 245.9, length = 2000)
        else
            range(245.9, 750, length = 4000)
        end
        #q <= 0.95 ? range(0, 125, length =  2000) : range(120, 748, length =  4000)
    elseif p == 1/2
        if q < 0.9
            range(0, 4.09, length = 2000)
        elseif q <= 0.95
            range(4.0, 6.1, length = 1000)
        elseif q <= 0.97
            range(6., 6.80, length = 1000)
        elseif q <= 0.99
            range(6.79, 9.215, length = 2000)
        else
            range(9.1, 20, length = 2000)
        end
        #q <= 0.95 ? range(0, 6, length =  1500) : range(5., 20, length =  2000)
    elseif p == 3/4
        q <= 0.95 ? range(0, 2.5, length =  1000) : range(2., 6.4, length =  1000)
    else
        q = range(0, 1000, length = 5000)
    end
end

@everywhere function qepd(quant::Float64, p::Real)
    quant < 1 && quant >= 0.5 || throw(DomainError("quant must be between 0.5, 1"))
    """if quant < 0.5
        x = range(-4, 0, length = 500)
    elseif quant < 0.95
        x = range(0, p <= 1/2 ? 6 : 3, length = 1000)
    else
        x = range(p <= 1/2 ? 5 : 1, p <= 1/2 ? 20 : 5, length = 2000)
    end"""
    x = qrange(quant, p)
    res = pepd.(x, p) .> quant
    x[findall(res .== 1)[1]]
end

##

# quantile function using roots
function qepd(q::Real, p::Real; x0::Real = 1)
    f(x) = pepd(x, p) - q
    find_zero(f, x0, Order16())
end


@everywhere function C(x::Float64, y::Float64, p::Real, ρ::Real)
    1/√(1-ρ^2) * g((x^2 + y^2 - 2*ρ*x*y)/(1-ρ^2), p)
end
@everywhere C1(x::Float64, p::Real, ρ::Real, q::Real) = quadgk(y -> C(y, x, p, ρ), -Inf, q)[1]
@everywhere C2(p::Real, ρ::Real, q::Real) = quadgk(y -> C1(y, p, ρ, q), -Inf, q; rtol=1e-12)[1]


#### Fig 6 of Dependence Measures for Extreme Value Analyses ####
ρ = [0., 0.3, 0.6, 0.9]
ρ = [0.9]
u = [range(0.9, 0.98, length = 5); range(0.99, 1, length = 6)]
#u = [range(0.9, 0.98, length = 10); range(0.99, 1, length = 20)]
p = [1/4, 1]

plt_dat = DataFrame(rho = repeat(ρ, inner = length(u)) |> x -> repeat(x, inner = length(p)),
    u = repeat(u, length(ρ)) |> x -> repeat(x, inner = length(p)),
    p = repeat(p, length(ρ)*length(u)), val = 0.)

cols = names(plt_dat)
plt_dat = SharedArray(Matrix(plt_dat))

@sync @distributed for i ∈ 1:size(plt_dat,1)
    println(i)
    if plt_dat[i, 2] != 1.
        a = qepd(plt_dat[i, 2], plt_dat[i, 3])
        val = C2(plt_dat[i, 3], plt_dat[i, 1], a)
        plt_dat[i, 4] = 2 - log(val)/log(plt_dat[i, 2])
    else
        plt_dat[i,4] = 0
    end
end

plt_dat = DataFrame(Tables.table(plt_dat)) |> x -> rename!(x, cols)
#CSV.write("plt_dat.csv", plt_dat)
#@df plt_dat plot(:u, :val, group = :rho, legend=:none)
@df plt_dat plot(:u, :val, group = (:rho, :p))

#### Compare with MVL
