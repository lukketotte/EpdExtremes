using SpecialFunctions, LinearAlgebra

function pFH(r::AbstractVector{<:Real}, par::AbstractVector{<:Real}; log::Bool = false)
  β,γ = par
  res = zeros(Float64, length(r))
  for i in eachindex(res)
    if r[i] > 1
      if β == 0.
        res[i] = log ? log(1 - r[i]^(-γ)) : (1 - r[i]^(-γ))
      else
        ret = 1 - exp(-γ*(r[i]^β - 1)/β)
        res[i] = log ? log(ret) : ret
      end
    end
  end
  res
end

pFH(r::Real, par::AbstractVector{<:Real}; log::Bool = false) = pFH([r], par; log=log)
  

pFH(1.1, [1.2, 0.5])



