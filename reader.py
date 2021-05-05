#! /usr/bin/python3

from forbiddenfruit import curse, reverse
import re

def isfloat(self):
	try:
		float(self)
		return True
	except ValueError:
		return False

def rSpecifications(filename):
	fd = open(filename, 'r')
	curse(str, 'isfloat', isfloat)
	spec = {}
	for line in fd:
		if re.search('^Autoencoder:',line):
			ae_layers = [int(s) for s in line.split() if s.isdigit()]
			spec['autoencoder'] = ae_layers
		elif re.search('\\b(D|d)ecod(e|ificado)r\\b',line):
			dn_layers = []
			for s in line.split():
				config = []
				if s[0] == '[':
					for val in s[1:-1].split(','):
						config.append(int(val))
					dn_layers.append(config)
			spec['decoder'] = dn_layers
		elif re.search('\\b(N|n)etwork\\b',line):
			for x in line.split():
				if re.search('=',x):
					key, value = x.split('=')
					spec[key] = value
		elif re.match('\\b(R|r)egulariz(ers?|ador(es)?)\\b',line):
			reg_aux = line.split(':')[1]
			regularizer = [s for s in reg_aux.split()]
			spec['ae_reg'] = regularizer
		elif re.search('\\b(D|d)ropout\\b', line):
			dropout = [float(s) for s in line.split() if s.isfloat() and float(s) < 1.]
			spec['dropout'] = dropout
		elif re.search('\\b(T|t)urns', line):
			spec['turns'] = int(line.split()[-1])
		elif re.search('\\b(S|s)ample (S|s)ize', line):
			spec['sample_size'] = int(line.split()[-1])
		elif re.search('\\b(C|c)ompressed', line):
			spec['compression_size'] = int(line.split()[-1])
		elif re.search('\\b(B|b)atch\\b',line):
			spec['batch_size'] = int(line.split()[-1])
		elif re.search('\\b((N|n)oise|(R|r)uído)\\b',line):
			noisePar = [float(s) for s in line.split() if s.isfloat() and not s.isdigit()]
			if re.search('\\b((S|s)|((E|e)s))pecifica(tion|ção)\\b',line):
				noisePar.append(line.split()[-1])
			spec['noise'] = noisePar
		elif re.search('\\b(W|w)indow\\b', line):
			if re.search('\\b(T|t)est\\b',line):
				spec['test_window'] = int(line.split()[-1])
			else:
				window = int(line.split()[-1])
				spec['train_style'] = 'window'
				spec['window_stride'] = window
		elif re.search('\\b(S|s)al?v(ed|o)\\s(em|at)',line):
			aux = []
			for line in fd:
				aux.append(line[1:])
			spec['ae_dirs'] = aux
		elif re.search('\\b((D|d)ataset|(F|f)(unction|unção))\\b',line):
			spec['dataset'] = line.split()[-1]
		elif re.search('\\b(E|e)arlystopping',line):
			spec['patience'] = int(line.split()[-1])
		elif re.search('\\b(C|c)hecking',line):
			vals = [int(s) for s in line.split() if s.isdigit()]
			spec['checking_range'] = vals[0]
			spec['start_check'] = vals[1]
		elif re.search('\\b(C|c)ross-validation',line):
			spec['train_style'] = 'cross-validation'
		elif re.search('\\b(U|u)pdate\\b',line):
			spec['update_recipes'] = line.split()[-1].split(',')
	reverse(str, 'isfloat')
	fd.close()
	return spec

def searchFiles(filenames,expression):
	if not isinstance(filenames,list):
		fd = open(filenames, 'r')
		for line in fd:
			if re.search(expression,line):
				fd.close()
				return True
		return False
	res = []
	for filename in filenames:
		fd = open(filename,'r')
		found = False
		for line in fd:
			if re.search(expression,line):
				res.append(True)
				fd.close()
				found = True
				break
		if not found:
			res.append(False)
	return res

def getFolders(group,filename='summary.txt'):
	filenames = []
	fd = open(filename, 'r')
	cont = True
	for line in fd:
		if re.search("^>>", line):
			if group == line[2:-1]:
				for l in fd:
					if re.search("(\-\-)+", l):
						break
					filenames.append(l[1:-1])
				cont = False
			if not cont:
				break
	fd.close()
	return filenames

#it is tightly related to the readme.txt files (assumes predetermined positions)
def readReadme(filename):
	fd = open(filename, 'r')
	info = {}
	curse(str, 'isfloat', isfloat)
	for line in fd:
		if re.search('\\btrain batche',line):
			info['batch_size'] = int(line.split()[-1])
		elif re.search('\\bsamples',line):
			info['sample_size'] = int(line.split()[-2])
		elif re.search('\\bGaussian Noise\\b',line):
			info['noise'] = [float(s) for s in line.split() if s.isfloat()]
		elif re.search('\\bActivation\\b',line):
			info['activation'] = line.split()[-1]
		elif re.search('\\bturns per configuration', line):
			info['turns'] = int(line.split()[1])
		elif re.search('\\bpatience\\b', line):
			info['patience'] = int(line.split()[-1])
		elif re.search('\\bTrain Style\\b',line):
			info['train_style'] = line.split(': ')[-1][:-1]	#remove \n at the end
		elif re.search('\\bWindow\\b',line):
			info['window'] = int(line.split()[-1])
		elif re.search('\\bBuffer',line):
			info['buffer'] = int(line.split()[-1])
		elif re.search('\\bonly in testing',line):	#aparentemente nao cai aqui
			info['test_only'] = line.split()[-1]
	reverse(str, 'isfloat')
	return info

def readPlReq(filename):
	fd = open(filename, 'r')
	desired_chars = []
	curse(str, 'isfloat', isfloat)
	for line in fd:
		if re.search('\\bGroup ID',line):
			group_id = line.split(': ')[1][:-1]
		elif re.search('\\bGaussian Noise',line):
			aux = [float(s) for s in line.split() if s.isfloat()]
			if not len(aux):
				desired_chars.append('Noise of (?!0.00 average and 0.00 standard)')
			else:
				desired_chars.append('Noise of {:.2f} average and {:.2f} standard'.format(
					aux[0],aux[1]))
		elif re.search('\\bActivation',line):
			desired_chars.append('Activation: '.format(line.split()[-1]))
		elif re.search('\\bTesting Buffer', line):
			desired_chars.append('Buffer Size: {}'.format(int(line.split()[-1])))
		elif re.search('\\bDatatype',line):
			datatype = line.split()[-1]
			if re.search('\(',datatype):
				datatype = re.escape(datatype)		#adds \ befor '(' and ')
			datatype += '$'
			desired_chars.append(datatype)
		elif re.search('\\bSample',line):
			desired_chars.append('made of {} samples'.format(line.split()[-1]))
		elif re.search('\\bBatch',line):
			desired_chars.append('batch size: {}'.format(line.split()[-1]))
#		elif re.search('\\bDatatype',line):
#			desired_chars.append(line.split()[-1])
		if re.search('\\bin testing',line):
			desired_chars.append('only in testing: True')
	reverse(str, 'isfloat')
	fd.close()
	return (desired_chars,group_id)

def get_lr(cmp_spec,out_act,hid_act,other_sp=None,filename='known_lr.txt'):
	cmp_id = '### {} dt {} ###'.format(cmp_spec[0],cmp_spec[1])
	act_id = '> {} && {}'.format(out_act,hid_act)
	if other_sp:
		act_id += ' && {}'.format(other_sp)
	try:
		lr_fd = open(filename,'r')
	except IOError:
		return None
	id_found = False
	block = list()
	for line in lr_fd:
		if not id_found:
			if line[:-1] == cmp_id: id_found = True
		if id_found:
			if re.search('^#+i\n$',line):
				break
			block.append(line[:-1])
	lr_fd.close()
	if not id_found:
		return None
	id_found = False
	lr_vals = dict()
	for line in block:
		if not id_found and line == act_id:
			id_found = True
		elif id_found:
			if line == '> END':
				break
			else:
				config,lr_id = line.split(' - ')
				config = config[2:]
				lr_mm = [float(i) for i in lr_id.split()]
				lr_vals[config] = (lr_mm[0],lr_mm[1])
#		if re.search('^>>AE',line):
#			config,lr_id = line.split(' - ')
#			config = config[2:]
#			lr_mm = [float(i) for i in lr_id.split()]
#			lr_vals[config] = (lr_mm[0],lr_mm[1])
#			print(lr_vals); input()
	if bool(lr_vals):
		return lr_vals
	return None

if __name__ == '__main__':
	lr_vals = get_lr((100,25),'sigmoid','selu', other_sp='nEp')
	for i in lr_vals.keys():
		print('{}:'.format(i))
		(min_lr,max_lr) = lr_vals[i]
		print('min_lr: {}; max_lr: {}'.format(min_lr,max_lr))
	lr_vals = get_lr((100,30),'c_sigmoid .5','selu','inexisente.txt')
	print(lr_vals)
#	specifications = rSpecifications('fExp.txt')
#	for key,s in specifications.items():
#		print('{}: {}'.format(key,s))
#	files = ['a.txt', 'c.txt', 'b.txt', 'd.txt']
#	expression = '\\babacate\\b'
#	print(searchFiles(files,expression))
