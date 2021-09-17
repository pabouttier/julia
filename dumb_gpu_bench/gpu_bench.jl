using LinearAlgebra # for mul!
using BenchmarkTools
using CUDA
using Random
using PrettyTables
include("absmax_avx.jl")

function broadcast_expr!(x,y,z)
    @. z=x+exp(y)
end
#
function testexpr(n=2^23)
    @show n
    CUDA.allowscalar(false)

    xf=rand(Float32,n)
    yf=rand(Float32,n)
    zf=rand(Float32,n)

    xfc=CuArray(xf)
    yfc=CuArray(yf)
    zfc=CuArray(zf)

    broadcast_expr!(xf,yf,zf)
    broadcast_expr!(xfc,yfc,zfc)
    @show sum(zf),sum(zfc)

    tcpu=@belapsed CUDA.@sync broadcast_expr!($xf,$yf,$zf)
    tgpu=@belapsed CUDA.@sync broadcast_expr!($xfc,$yfc,$zfc)

    SpUp=tcpu/tgpu
    @show tcpu,tgpu,SpUp
end

function incircle(x,y)
    Int(x^2+y^2<one(typeof(x)))
end

function picalc(nchunks,frand,T,chunksize)
    count=0

    for i=1:nchunks
        x=frand(T,chunksize)
        y=frand(T,chunksize)

        count+=sum(incircle.(x,y))
    end
    return Float64(4*count/(nchunks*chunksize))
end

function testpicalc(n=2^22)
    CUDA.allowscalar(false)

    @show n
    @show picalc(100,rand,Float32,n)
    @show picalc(100,CUDA.rand,Float32,n)
    tcpu=@belapsed $picalc(100,$rand,Float32,$n)
    tgpu=@belapsed $picalc(100,$CUDA.rand,Float32,$n)
    SpUp=tcpu/tgpu
    @show tcpu,tgpu,SpUp
end

dopicalc=false
dopicalc && testpicalc()


gbs(n,t,T) = 2sizeof(T)*n/(1.e9*t)#R+W

function testbandwidth(n=2^22,T=Float64) 
    CUDA.allowscalar(false)
    a=rand(T,n)
    b=rand(T,n)
    tcopy_cpu=@belapsed copyto!($a,$b)
    ca=CuArray(a) 
    cb=CuArray(b) 
    tcopy_gpu=@belapsed CUDA.@sync copyto!($ca,$cb)
    tcopy_cpugpu=@belapsed CUDA.@sync copyto!($ca,$b)
    println("BW GPU<->VRAM: ",gbs(n,tcopy_gpu,T)," GB/s")
    println("BW CPU<->RAM:  ",gbs(n,tcopy_cpu,T)," GB/s")
    println("bW CPU<->GPU:  ",gbs(n,tcopy_cpugpu,T)," GB/s")
end


function gemmGPU(Cc,Ac,Bc)
    mul!(Cc,Ac,Bc)
    Array(Cc) #copy to CPU
end

gflops_gemm(n,ts)=round(Float64(2)*Float64(n)^3/(ts*1.e9),digits=2)
function testgemm(n=1024)
    CUDA.allowscalar(false)

    A=rand(Float32,n,n)
    B=rand(Float32,n,n)
    C=rand(Float32,n,n)

    Ac=CuArray(A)
    Bc=CuArray(B)
    Cc=CuArray(C)

    mul!(C,A,B)
    Cgpucpu=gemmGPU(Cc,Ac,Bc)
    println("L2 diff norm CPU/GPU :",norm(Cgpucpu-C)/norm(C))

    t_cpu=@belapsed mul!($C,$A,$B)
    t_gpu1=@belapsed CUDA.@sync mul!($Cc,$Ac,$Bc)
    t_gpu2=@belapsed CUDA.@sync gemmGPU($Cc,$Ac,$Bc)#With transfert

    println("CPU Float32 gemm GFlops n=$n  \t"," :",gflops_gemm(n,t_cpu))
    println("GPU Float32 gemm GFlops n=$n  \t"," :",gflops_gemm(n,t_gpu1))
    println("GPU Float32 gemm GFlops n=$n (+PCIE) \t"," :",gflops_gemm(n,t_gpu2))
end


function absmax(a)
    result=zero(eltype(a))
    for x in a
        abs(x)>result && (result=abs(x))
    end
    result
end

absmax_mapreduce(x) = mapreduce(abs,max,x)


function gcomps_absmax(f,n=2^22,T=Float64)
    #The following line is for reproducible test
    Random.seed!(1234);
    a=rand(T,n)
    #It can save a lot of time to check the result
    # continuously during the optimisation process...
    @show f,f(a),absmax(a)
    @assert f(a)==absmax(a) #test against a ref impl
    t=@belapsed $f($a) #Interpolation of $f too !
    result=n/(t*1.e9)
    println("$result GComps for $f  ! ")
    result
end

function gcomps_absmax_gpu(fgpu,n=2^22,T=Float64)
    #The following line is for reproducible test
    CUDA.allowscalar(false)
    Random.seed!(1234);
    a=rand(T,n)
    ac=CuArray(a)
    #It can save a lot of time to check the result
    # continuously during the optimisation process...
    @show fgpu,fgpu(ac),absmax(a)
    @assert fgpu(ac)==absmax(a) #test against a ref impl
    # @assert Array(fgpu(ac))[1]==absmax(a) #test against a ref impl
    t=@belapsed CUDA.@sync $fgpu($ac) #Interpolation of $f too !
    result=n/(t*1.e9)
    println("$result GComps for $fgpu  ! ")
    result
end


function testabsmax(n=2^23)
    CUDA.allowscalar(false)

    @show n

    gcomps=Dict{String,Float64}()
    gcomps["absmax"]=gcomps_absmax(absmax,n)
    gcomps["absmax_avx"]=gcomps_absmax(absmax_avx,n)
    gcomps["absmax_mapreduce"]=gcomps_absmax(absmax_mapreduce,n)
    gcomps["absmax_mapreduce_gpu"]=gcomps_absmax_gpu(absmax_mapreduce,n)
    gcomps["absmax_mapreduce_gpu_Float32"]=gcomps_absmax_gpu(absmax_mapreduce,n,Float32)
    #
    # gcomps["absmax_mrdim_gpu"]=gcomps_absmax_gpu(absmax_mrdim!,n)
    # gcomps["absmax_mrdim_gpu_Float32"]=gcomps_absmax_gpu(absmax_mrdim!,n,Float32)
    pretty_table(gcomps;formatters=ft_printf("%5.3f"),alignment=:l)
    gcomps
end


function __main__()
    # testexpr()
    testbandwidth()
    # testpicalc()
    # testabsmax()
    # for i in 1:3
    #    try
    #        testgemm()
    #        break
    #    catch e
    #        display(e)
    #        sleep(1)
    #        println("Retrying")
    #   end
    # end
end
__main__()
