# Installation de Julia 1.6.12 dans le $HOME

```
cd $HOME
wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.2-linux-x86_64.tar.gz
tar zxvf julia-1.6.2-linux-x86_64.tar.gz
export PATH=$PATH:$HOME/julia-1.6.2/bin
```

# Installation des dépendances du script 

Dans le dossier de ce dépôt (peut  être long) :
```
julia setup.jl
```

# Lancement du script
```
julia gpu_bench.jl
```

