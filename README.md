# mnode2vec

This repository references [node2vec](https://snap.stanford.edu/node2vec/) to implement multi-network wandering to obtain feature vectors for multiple networks

###Example

Run the following command in the project home directory to run mnode2vec on the three example networks:

```python
python src/main.py --input graph --output emb/result.emb
```

### Input

The files in the input path support edgelist files or txt files in the following formats,The graph defaults to a weighted undirected graph, and these options can be changed.

​	node1_id_int node2_id_int <weight_float, optional>

### Output

The output file is the same as node2vec, which has *n+1* lines for a graph with *n* vertices.

The first line has the following format:

	num_of_nodes dim_of_representation

The next *n* lines are as follows:
	
	node_id dim1 dim2 ... dimd

where dim1, ... , dimd is the *d*-dimensional representation .