'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
import os
from gensim.models import Word2Vec

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='graph',
						help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/result.emb',
						help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
						help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
						help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
						help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
						help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
					  help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
						help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
						help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
						help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
						help='Boolean specifying (un)weighted. Default is weighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=True)

	parser.add_argument('--directed', dest='directed', action='store_true',
						help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()

def read_graph():
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def learn_embeddings(walkpath):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	l=[]
	for line in open(walkpath+'/result.txt','r'):
		l.append(list(line.strip('\n').split(' ')))
	for j in range(len(l)):
		del l[j][args.walk_length]
	# walks = [map(str, walk) for walk in walks]
	model = Word2Vec(l, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.wv.save_word2vec_format(args.output)
	
	return

def find_graph(inputpath):
	filepath = os.listdir(inputpath)

	if '.DS-Store' in filepath:
		del filepath[filepath.index('.DS-Store')]
	for i in range(len(filepath)):
		filepath[i] = inputpath + '/' + filepath[i]
	return filepath


def get_name(inputpath):
	filepath = os.listdir(inputpath)
	if '.DS-Store' in filepath:
		del filepath[filepath.index('.DS-Store')]
	title={}
	type={}
	for i in range(len(filepath)):
		title[i],type[i]=os.path.splitext(filepath[i])
	return title
def record_walk(walks,path):
	with open(path, 'w') as f:
		for h in walks:
			for k in h:
				f.write(str(k))
				f.write(' ')
			f.write('\n')
		f.close()
def combine_walk(walk_path):
	filename=os.listdir(walk_path)
	if '.DS-Store' in filename:
		del filename[filename.index('.DS-Store')]
	resultname=walk_path+'/result.txt'
	f = open(resultname, 'w')
	for i in range(len(filename)):
		filepath=walk_path+'/'+filename[i]
		for line in open(filepath):
			f.writelines(line)
	f.close()

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	graph_path=find_graph(args.input)
	graph_title=get_name(args.input)
	walkpath=args.input+'/walk'
	os.makedirs(walkpath)
	for i in range(len(graph_path)):
		args.input=graph_path[i]
		walk_output=walkpath+'/'+str(i)+'.txt'
  
		nx_G = read_graph()
		G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
		G.preprocess_transition_probs()
		walks = G.simulate_walks(args.num_walks, args.walk_length)
		record_walk(walks,walk_output)
		del G,walks
	combine_walk(walkpath)
	learn_embeddings(walkpath)

if __name__ == "__main__":
	args = parse_args()

	main(args)
