from graphviz import Source
path = '../build/computational_graph.dot'
s = Source.from_file(path)
s.view()