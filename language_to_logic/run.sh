# ./run.sh [seq2seq|seq2tree] [jobqueries|geoqueries|atis] [lstm|attention]

if [ -z $4 ] ; then
  GPU_ID=0
else
  GPU_ID=$4
fi

if [ $3 = "lstm" ] || [ $3 = "attention" ] ; then
  PWD_DIR=$(pwd)
  WORK_DIR=$(dirname "$(readlink -f "$0")")/$1/$2
  DUMP_DIR=$WORK_DIR/dump_$3
  cd $WORK_DIR
  julia data.jl --data_dir $WORK_DIR/data
  CUDA_VISIBLE_DEVICES=$GPU_ID julia $3/main.jl --data_dir $WORK_DIR/data --checkpoint_dir $DUMP_DIR 
  CUDA_VISIBLE_DEVICES=$GPU_ID julia $3/sample.jl --data_dir $WORK_DIR/data
  cd $PWD_DIR
fi


