echo "kernel,runtime,gpu,collapse,mem_to,mem_alloc,mem_from,mem_delete,num_var,N1,N2" >> dataset_laplace.csv
for i in `ls laplace_*.out`
do
  filename=$(basename -- "$i")
  filename="${filename%.*}"
  if [ ! -f output_${filename}.csv ]; then
    ./$i;
    cat output_${filename}.csv >> dataset_laplace.csv
  fi
done
