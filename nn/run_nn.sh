echo "kernel,runtime,gpu,collapse,mem_to,mem_alloc,mem_from,mem_delete,num_var,N1" >> dataset_nn.csv
for i in `ls nn_*.out`
do
  filename=$(basename -- "$i")
  filename="${filename%.*}"
  ./$i;
  cat output_${filename}.csv >> dataset_nn.csv
done
