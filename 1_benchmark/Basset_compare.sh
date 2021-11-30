export BASSETDIR=/media/ggj/Files/Dev/Basset-master

export PATH=$BASSETDIR/src:$PATH
export PYTHONPATH=$BASSETDIR/src:$PYTHONPATH
export LUA_PATH="$BASSETDIR/src/?.lua;$LUA_PATH"

## ./install_dependencies.py
# wget https://bitop.luajit.org/download/LuaBitOp-1.0.2.zip
# unzip LuaBitOp-1.0.2.zip
# make LUAINC=-I/home/ggj/torch/install/include/
# sudo ~/torch/install/bin/luarocks install luafilesystem
# sudo ~/torch/install/bin/luarocks install dpnn
# sudo ~/torch/install/bin/luarocks install inn 
# sudo ~/torch/install/bin/luarocks install torchx 
# sudo ~/torch/install/bin/luarocks install dp 
# sudo ~/torch/install/bin/luarocks install totem
# sudo ~/torch/install/bin/luarocks install hdf5 

basset_train.lua -cuda -job params_basset.txt -stagnant_t 10 Dataset.Basset.h5

{
  conv_filter_sizes :
    {
      1 : 19
      2 : 11
      3 : 7
    }
  weight_norm : 7
  momentum : 0.98
  learning_rate : 0.002
  hidden_units :
    {
      1 : 1000
      2 : 32
    }
  conv_filters :
    {
      1 : 300
      2 : 200
      3 : 200
    }
  hidden_dropouts :
    {
      1 : 0.3
      2 : 0.3
    }
  pool_width :
    {
      1 : 3
      2 : 4
      3 : 4
    }
}
seq_len: 20000, filter_size: 19, pad_width: 18
seq_len: 6667, filter_size: 11, pad_width: 10
seq_len: 1667, filter_size: 7, pad_width: 6
Running on GPU.
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> (14) -> (15) -> (16) -> (17) -> (18) -> (19) -> (20) -> (21) -> (22) -> (23) -> output]
  (1): nn.SpatialConvolution(4 -> 300, 19x1, 1,1, 9,0)
  (2): nn.SpatialBatchNormalization (4D) (300)
  (3): nn.ReLU
  (4): nn.SpatialMaxPooling(3x1, 3,1)
  (5): nn.SpatialConvolution(300 -> 200, 11x1, 1,1, 5,0)
  (6): nn.SpatialBatchNormalization (4D) (200)
  (7): nn.ReLU
  (8): nn.SpatialMaxPooling(4x1, 4,1)
  (9): nn.SpatialConvolution(200 -> 200, 7x1, 1,1, 3,0)
  (10): nn.SpatialBatchNormalization (4D) (200)
  (11): nn.ReLU
  (12): nn.SpatialMaxPooling(4x1, 4,1)
  (13): nn.Reshape(83400)
  (14): nn.Linear(83400 -> 1000)
  (15): nn.BatchNormalization (2D) (1000)
  (16): nn.ReLU
  (17): nn.Dropout(0.300000)
  (18): nn.Linear(1000 -> 32)
  (19): nn.BatchNormalization (2D) (32)
  (20): nn.ReLU
  (21): nn.Dropout(0.300000)
  (22): nn.Linear(32 -> 134557)
  (23): nn.Sigmoid
}
/home/ggj/torch/install/bin/luajit: /home/ggj/torch/install/share/lua/5.1/hdf5/ffi.lua:352: Reading data of class ENUM(50331748)is unsupported
stack traceback:
        [C]: in function 'error'
        /home/ggj/torch/install/share/lua/5.1/hdf5/ffi.lua:352: in function '_getTorchType'
        /home/ggj/torch/install/share/lua/5.1/hdf5/dataset.lua:88: in function 'getTensorFactory'
        /home/ggj/torch/install/share/lua/5.1/hdf5/dataset.lua:138: in function 'partial'
        /media/ggj/Files/Dev/Basset-master/src/batcher.lua:39: in function 'next'
        /media/ggj/Files/Dev/Basset-master/src/convnet.lua:1009: in function 'train_epoch'
        /media/ggj/Files/Dev/Basset-master/src/basset_train.lua:156: in main chunk
        [C]: in function 'dofile'
        .../ggj/torch/install/lib/luarocks/rocks/trepl/scm-1/bin/th:150: in main chunk
        [C]: at 0x00405d50


X_batch = batcher.Xf:partial({batcher.start,batcher.stop}, {1,batcher.init_depth}, {1,1}, {1,batcher.seq_len}):double()
Y_batch = batcher.Yf:partial({batcher.start,batcher.stop}, {1,batcher.num_targets}):double()