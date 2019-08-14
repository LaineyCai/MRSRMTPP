# MRSRMTPP
This python project implements the MRSRMTPP model proposed in the below paper which is submitted to Cognitive Computation in April 2019.
Hongyun Cai, Thanh Tung Nguyen, Yan Li, Vincent W. Zheng, Binbin Chen, Gao Cong, and Xiaoli Li. Modeling Marked Temporal Point Process UsingMulti-relation Structure RNN. 

The code has been tested under Ubuntu 14.04.5 LTS with Intel(R) Xeon(R) CPU E5-2667 v3 @ 3.20GHz*32, 128G memory, and NVIDIA Corporation GK210GL [Tesla K80].


## Dataset input
1. We use the two public datasets, Twitter[1] and Memes[2], as examples.
2. The input for MRSRMTPP is put in "data/twitter" and "data/memes"
3. For each dataset, there are four files.
	3.1 train.txt
	    Description: the sequence of (event-time) pairs for training. Each line is an event sequence.
	    Format: event_id_1 event_time_1 event_id_2 event_time_2 ...
	3.2 test.txt
	    Description: the sequence of (event-time) pairs for testing. Each line is an event sequence.
	    Format: event_id_1 event_time_1 event_id_2 event_time_2 ...
	3.3 seen_nodes.txt
	    Description: The list of all event ids observed in the dataset.
	    Format: a int per line, each line is a observed event id
	3.4 graph.txt
	    Description: The ontology dependence between events, each line indicates that there is a dependency relationship between the two events
	    Format: event_id_1 event_id_2
4. In our experiments, we use another SMRT dataset, however it cannot be public accessible due to the Confidentiality Agreement.

[1] N. O. Hodas and K. Lerman, “The simple rules of social contagion,” CoRR, vol. abs/1308.5015, 2013.
[2] J. Leskovec, L. Backstrom, and J. Kleinberg, “Meme-tracking and the dynamics of the news cycle,” in KDD, 2009, pp. 497–506.


## Program Running
1. Parameters Setting
	1.1 dim_proj: the dimension for event embedding, default 256
	1.2 dim_att: the dimension for attention, default 128.
	1.3 maxlen: the maximum length of event sequence during the modelling, default 30.
	1.4 batch_size: the size of batch SGD, default 256.
	1.5 keep_ratio: the ratio of kept data for training, default 1.
	1.6 shuffle_data: boolean value indicates whether the training data is shuffled, default True.
	1.7 learning_rate: the initial learning rate for adam, default 0.001.
	1.8 global_steps: the maximum number of iterations for training, default 50000.
	1.9 disp_freq, save_freq, test_freq: the frequency for display result, save result and test on the testing data, default 100.
	1.10 weight_decay: the weight for l2 norm of the model parameters, default 0.0005.
2. Command to run the program
```
python tpgru.py
```

## Program output
1. For testing，run
```
python tpgru_topk.py
```
2. The displayed results includes: hits@10, hits@50, hits@100, MAP@10, MAP@50, MAP@100 and RMSE.

