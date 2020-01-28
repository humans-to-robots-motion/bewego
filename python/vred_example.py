import pybewego as bwgo
import numpy as np

np.random.seed(0)

dim = 9
dim_trans = 3
hidden_dim = 4
layers = 2
src_length = 2
pred_length = 5


l1W = np.random.uniform(-.1, .1, (dim*2-dim_trans, hidden_dim*3))
l1R = np.random.uniform(-.1, .1, (hidden_dim, hidden_dim*3))
l1b = np.random.uniform(-.1, .1, (2, hidden_dim*3))
l2W = np.random.uniform(-.1, .1, (hidden_dim, hidden_dim*3))
l2R = np.random.uniform(-.1, .1, (hidden_dim, hidden_dim*3))
l2b = np.random.uniform(-.1, .1, (2, hidden_dim*3))
Wdense = np.random.uniform(-.1, .1, (hidden_dim, dim))
bdense = np.random.uniform(-.1, .1, dim)

grucell1 = bwgo.GRUCell(np.transpose(l1W[:, :hidden_dim]), np.transpose(l1W[:, hidden_dim:hidden_dim*2]),  np.transpose(l1W[:, hidden_dim*2:]), np.transpose(l1R[:, :hidden_dim]), np.transpose(l1R[:, hidden_dim:hidden_dim*2]), np.transpose(l1R[:, hidden_dim*2:]), l1b[0, :hidden_dim], l1b[0, hidden_dim:hidden_dim*2], l1b[0, hidden_dim*2:], l1b[1, :hidden_dim], l1b[1, hidden_dim:hidden_dim*2], l1b[1, hidden_dim*2:])
grucell2 = bwgo.GRUCell(np.transpose(l2W[:, :hidden_dim]), np.transpose(l2W[:, hidden_dim:hidden_dim*2]),  np.transpose(l2W[:, hidden_dim*2:]), np.transpose(l2R[:, :hidden_dim]), np.transpose(l2R[:, hidden_dim:hidden_dim*2]), np.transpose(l2R[:, hidden_dim*2:]), l2b[0, :hidden_dim], l2b[0, hidden_dim:hidden_dim*2], l2b[0, hidden_dim*2:], l2b[1, :hidden_dim], l2b[1, hidden_dim:hidden_dim*2], l2b[1, hidden_dim*2:])
cells = [grucell1, grucell2]
cell = bwgo.StackedCoupledRNNCell(layers, hidden_dim, dim, cells, Wdense, bdense)
vred = bwgo.VRED(cell, dim_trans)
states = np.random.uniform(-.1, .1, hidden_dim*layers)
inp = np.random.uniform(-1., 1., (src_length, dim))
deltas = np.zeros((pred_length-1, dim))
print(vred.Forward(inp, deltas, src_length, pred_length))
