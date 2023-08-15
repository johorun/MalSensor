import networkx as nx
import os
from string import digits
import re

API = []
with open('API.txt', 'r') as f:
	for line in f:
		API.append(line.strip())


dir = ''	
exist_files = os.listdir(dir) #all file or filedir in this pathcd 

for file in exist_files:
	cg = nx.DiGraph()
	with open(dir + file, mode = 'r', encoding = 'utf-8', errors='ignore') as f:
		called = ''
		for line in f:
			if 'Function Name:' in line:
				called = line.split('Function Name:')[1].strip('\n')
				cg.add_node(called)
					

				# if '@' in called:
				# 	called = (called.split('@')[0]).strip('\n').translate(number).translate(char) 
				# 	cg.add_node(called)
				# elif '::' in called:
				# 	called = (called.split('::')[-1]).strip('\n').translate(number).translate(char)
				# 	cg.add_node(called)
				# elif 'sub_' in called:
				# 	called = called.strip('\n')
				# 	cg.add_node(called)
				# else:
				# 	called = called.strip('\n').translate(number).translate(char)
				# 	cg.add_node(called)
			else:
				#cg.add_edge(line.split('		')[1].strip('\n'),called)
				call = line.strip('\n')
				cg.add_edge(call, called)

				# if '@' in call:
				# 	call = (call.split('@')[0]).strip('\n').translate(number).translate(char) 
				# 	cg.add_edge(call, called)
				# elif '::' in call:
				# 	call = (call.split('::')[-1]).strip('\n').translate(number).translate(char)
				# 	cg.add_edge(call, called)
				# elif 'sub_' in call:
				# 	call = call.strip('\n')
				# 	cg.add_edge(call, called)
				# else:
				# 	call = call.strip('\n').translate(number).translate(char)
				# 	cg.add_edge(call, called)				

	nx.write_gexf(cg, file.split('.')[0] + '.gexf')
					

