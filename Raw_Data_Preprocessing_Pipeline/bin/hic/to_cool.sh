# conda install -c conda-forge -c bioconda cooler
cooler load -f coo \
    --assembly refence_name \
    --one-based \
    path/raw/10000/example_10000_abs.bed \
    path/raw/10000/example_10000.matrix \
    ./cool/10k_hic.cool

cooler info ./cool/10k_hic.cool > have_cool.log