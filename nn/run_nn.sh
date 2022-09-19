DATASET=dataset_nn_v100_`date +'%m%d%y_%H%M%S%3N'`.csv
echo "kernel,runtime,gpu,collapse,num_teams,num_threads,mem_to,mem_alloc,mem_from,mem_delete,num_var,N1" > ${DATASET}
for i in `ls nn_*.out`
do
  filename=$(basename -- "$i")
  filename="${filename%.*}"
  if [ ! -f output_${filename}.csv ]; then
    ./$i;
  fi
  cat output_${filename}.csv >> ${DATASET}
done
