# Mnode2vec

This is a new network embedding algorithm, which generalized the classic [node2vec](https://snap.stanford.edu/node2vec/) algorithm. It can process multiple networks and assign a feature vector to each node in networks.

The codes of Mnode2vec are provided in the folder code. The file main.py contains the main function of Mnode2vec and the file node2vec.py is the program of node2vec, which is called in main.py.

A small example is provided in the folder network. It contains three files, corresponding to three networks. The network is unweighted and undirected as default. However, this can be changed by setting proper options. The format of network file is as follows:

```
node1_id_int node2_id_int <weight_float, optional>
```

To run Mnode2vec, please use the following command: 

```python
python Code/main.py --input graph --output output/result.emb
```

Please note that there are only network files in the input directory.

The output file of the example is provided in folder output. If the input networks contain n vertices, the output file will include n+1 lines.

The first line has the following format:

	num_of_nodes dim_of_representation

The next n lines have the same format, defined by
	node_id dim1 dim2 ... dimd

where dim1, ... , dimd are features.