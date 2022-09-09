echo "kernel,runtime,gpu,collapse,mem_to,mem_alloc,mem_from,mem_delete,num_var,N1,N2,N3" >> dataset_mm.csv
for i in `ls mm_*.out`
do
  filename=$(basename -- "$i")
  filename="${filename%.*}"
  ./$i;
  cat output_${filename}.csv >> dataset_mm.csv
done
