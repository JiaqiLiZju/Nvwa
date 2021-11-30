# conda activate DeepMind
# python setup.py develop

export BASENJIDIR=/media/ggj/Files/Dev/basenji-master
export PATH=$BASENJIDIR/bin:$PATH
export PYTHONPATH=$BASENJIDIR/bin:$PYTHONPATH

basenji_train.py -o models/Benchmark/ models/params.json data/Benchmark/ &>log.train_basenji &
basenji_test.py --ai 0,1,2 --save -o output/Benchmark/ models/params.json models/Benchmark/model_best.h5 data/Benchmark
calculate_roc.py