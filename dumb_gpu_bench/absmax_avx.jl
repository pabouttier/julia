using LoopVectorization

function absmax_avx(x)
    result = zero(eltype(x))
    @avx for i in 1:length(x)
        result=max(abs(x[i]),result)
    end
    result
end
