#! /usr/bin/python3

from os.path import exists
from os import replace
from reader import searchFiles, readReadme
from re import search
from numpy import loadtxt

#specific for my needs
def wTable(entries,labels=[],table='table.txt'):
	cols = len(entries)
	fd = open(table,'w')
	if len(labels):
		label_col = '#{}'.format(labels[0])
		for label in labels[1:]: label_col += ' {}'.format(label)
		fd.write(label_col)
	if isinstance(entries[0], list):
		num_entries = len(entries[0])
	else:
		num_entries = (entries[0]).shape[0]
	for i in range(num_entries):
		line = '\n{:.5E}'.format((entries[0])[i])
		for j in range(1,cols):
			line += ' {:.5E}'.format((entries[j])[i])
		fd.write(line)
	fd.write('\n')
	fd.close()

def wReadme(entries, exType, aeReg=None, lstm=None,saveAt='readme.txt'):
	fd = open(saveAt,'w')
	fd.write("Description: {:s} compressive system using Autoencoders".format(entries.pop()) if
		exType == 'ae' else "Description: {:s} compressive system using Autoencoder and\
 a Deeper Neural Net Decoder".format(entries.pop()))
	if aeReg and exType == 'ae': fd.write('\nwith {} regularization'.format(aeReg))
	if exType == 'dn':
		fd.write('\n\tAE config: {} down to {}'.format(entries.pop(),entries.pop()))
	fd.write('\n\tAE Cyclical Learning Rate Mode: {}'.format(entries.pop()))
	fd.write('\n\tAE CLR Stepsize Constant: {}'.format(entries.pop()))
	if aeReg: fd.write('\n\t\twith {} regularization'.format(aeReg))
	fd.write('\n')
	fd.write("\n\tDatatype: {:s}".format(entries.pop()))
	fd.write("\n\tEach compression batch is made of {:d} samples".format(entries.pop()))
	#fd.write("\n\tWith Gaussian Noise of {:.2f} average and {:.2f} standard deviation".format(
		#entries.pop(),entries.pop()))
	fd.write("\n\tActivation: {}; Threshold limit: {:.2f}%".format(entries.pop(),entries.pop()*100))
	enc_act = entries.pop()
	if enc_act:
		fd.write('\n\tEncoding Output Activation: {}'.format(enc_act))
	fd.write('\n\tDecoder Type: {}'.format(entries.pop()))
	if lstm:
		lstm_config = [str(cell) for cell in lstm]
		fd.write(" with LSTM: {} (lstm config)".format('-'.join(lstm_config)))
	fd.write('\n\tDropout Values: {}'.format(entries.pop()))
	fd.write("\nTraining:");
	fd.write("\n\tUsed {:d} turns per configuration".format(entries.pop()))
	fd.write('\n\tTraining Model used {}'.format(entries.pop()))
	fd.write("\n\tTraining batch size: {}".format(entries.pop()))
	fd.write("\n\tEarlystopping patience: {}".format(entries.pop()))
	window_stride = entries.pop()
	fd.write("\n\tTrain Style: {}".format('Fixed with Sliding Windows' if window_stride else 'Cross-Validation'))
	if window_stride:
		fd.write("\n\tWindow stride: {}".format(window_stride))
	window_stride = entries.pop()
	if window_stride:
		for val in window_stride:
			fd.write(' and {}'.format(val))
	fd.write('\n\tCyclical Learning Rate Mode: {}'.format(entries.pop()))
	fd.write('\n\tCLR Stepsize Constant: {}'.format(entries.pop()))
	fd.write('\n\tData Augmentation: {}'.format(';'.join(entries.pop())))
	fd.write("\nTesting:")
	fd.write("\n\tBuffer Size: {}".format(entries.pop()))
	fd.write("\n\tNoise only in testing: {}".format(entries.pop()))
	#if exType == 'dn':
		#directories = entries.pop()
		#fd.write('\nAEs from:')
		#for dirs in directories:
			#fd.write('\n\t{}'.format(dirs))
	fd.write('\n')
	fd.close()

def wReadme2(entries, wait=None, saveAt='readme.txt'):
	fd = open(saveAt,'w')
	fd.write("Description: {:s} compressive system using Autoencoders - A Taxonomic Study".format(entries.pop()))
	fd.write('\n\tCompression: {} down to {}'.format(entries.pop(),entries.pop()))
	fd.write('\n\tAE Cyclical Learning Rate Mode: {}'.format(entries.pop()))
	fd.write('\n\tAE CLR Stepsize Constant: {}'.format(entries.pop()))
	fd.write('\n')
	fd.write("\n\tDatatype: {:s}".format(entries.pop()))
	fd.write("\n\tEach compression batch is made of {:d} samples".format(entries.pop()))
	fd.write("\n\tActivation: {}; Threshold limit: {:.2f}%".format(entries.pop(),entries.pop()*100))
	enc_act = entries.pop()
	if enc_act:
		fd.write('\n\tEncoding Output Activation: {}'.format(enc_act))
	fd.write('\n\tDeep Config. Activation: {}'.format(entries.pop()))
	fd.write('\n\tDropout Values: {}'.format(entries.pop()))
	fd.write("\nTraining:");
	fd.write("\n\tUsed {:d} turns per configuration".format(entries.pop()))
	fd.write('\n\tTraining Model used {}'.format(entries.pop()))
	fd.write("\n\tTraining batch size: {}".format(entries.pop()))
	fd.write("\n\tEarlystopping patience: {}".format(entries.pop()))
	if wait:
		fd.write(' after waiting {} epochs'.format(wait))
	window_stride = entries.pop()
	fd.write("\n\tTrain Style: {}".format('Fixed with Sliding Windows' if window_stride else 'Cross-Validation'))
	if window_stride:
		fd.write("\n\tWindow stride: {}".format(window_stride))
	window_stride = entries.pop()
	if window_stride:
		for val in window_stride:
			fd.write(' and {}'.format(val))
	fd.write('\n\tData Augmentation: {}'.format(';'.join(entries.pop())))
	fd.write("\nTesting:")
	fd.write("\n\tBuffer Size: {}".format(entries.pop()))
	fd.write('\n')
	fd.close()

def summarizer(directories, expressions, identifier, readme='readme.txt', saveAt='summary.txt'):
	match_dirs = []
	for directory in directories:
		filename = directory + '/' + readme
		include = True
		fd = open(filename,'r')
		f_line = fd.readline()
		fd.close()
		#regex explanation: line that does not contain
		if not search('^((?!(B|b)lacklist).)*$',f_line):
			include = False
		else:
			for expression in expressions:
				if not searchFiles(filename, expression):
					include = False
					break;
		if include:
			match_dirs.append(directory)
	if not len(match_dirs):
		return 'No matching files found'
	fd = open(saveAt,'a' if exists(saveAt) else 'w')
	fd.write('>>' + identifier + "\n")
	for dirs in match_dirs:
		fd.write('\t{}\n'.format(dirs))
	fd.write('-'*40 + "\n")
	fd.close()
	return 'Success'

def updatePlotID(filename, dont_add=[]):
#dont_add keeps the specifying terms present in every
#elements of the plot (therefore commun to all)
	in_fd = open('tmp.txt', 'w')
	out_fd = open(filename, 'r')
	for line in out_fd:
		dir_name = line.split(': ')[-1][:-1]	#remove \n at the end
		readme = dir_name + '/readme.txt'
		info = readReadme(readme)
		in_fd.write(line)
		for key in info.keys():
			ignore = False
			for val in dont_add:
				if key == val: ignore = True
			if not ignore:
				in_fd.write('\t' + key + ': {}\n'.format(info[key]))
	in_fd.close()
	out_fd.close()
	replace('tmp.txt',filename)
	return 'Success'

def printTable(filename, hpFile=None, return_vals=False):
	x_labels = []
	if hpFile:
		fd = open(hpFile,'r')
		for line in fd:
			if search('^AE Config', line):##ajeitar para o t5_taxo
				config = '[' + ','.join(line.split()[-1].split('-')) + ']'
				x_labels.append(config)
			elif search('^(C?AE|HIB)\-[0-9]+',line) or search('^(S|R)A?AE(\-[0-9]+)?',line):
				config = line.split()[0]
				x_labels.append(config)
			elif search('^[0-9]+(\.[0-9]+)?',line):
				config = line.split()[0]
				x_labels.append(config)
		fd.close()
	table = loadtxt(filename, comments='#', delimiter=' ')#.tolist()
	if not len(x_labels):
		x_labels = ['(#{})'.format(i) for i in range(table.shape[0])]
	with open(filename,'r') as reader:
		header = reader.readline()
	print(header[:-1])
	for label,vals in zip(x_labels,table.tolist()):
		entries = [str(i) for i in vals]
		line = label + ' ' + ' '.join(entries)
		print(line)
	if return_vals:
		if not hpFile:
			return None, table, header[1:].split()
		return x_labels, table, header[1:].split()

if __name__ == '__main__':
	printTable('./Ad_Plot/table.txt',hpFile='./Ad_Plot/table_helper.txt')
#	entries = ['Temporal', 'weather report from Rio', 12, 1., .3, 100]
#	entries.reverse()
#	wReadme(entries.copy(), exType='ae', aeReg='l2')
#	entries = ['Temporal', 100, 20, 'weather report from Rio', 100, 1., .3, 100]
#	entries.reverse()
#	wReadme(entries.copy(), exType='dn', saveAt='readme2.txt')
#	import os
#	dirname = '/home/mateusgilbert/trabalhos/IC/Prog/Proj2020/m0dev07/'
#	dirs = [dirname + name for name in os.listdir(dirname) if os.path.isdir(os.path.join(dirname,name))]
#	print(dirs)
#	expressions = ["Buffer Size: [1-9]{1}[0-9]*", 'Noise only in training: True']
#	identifier = '>>Directories with noisy test samples using sliding windows'
	#print(summarizer(dirs, expressions,identifier,saveAt='This_is_an_example.txt'))
	#updatePlotID('results_ID.txt', dont_ad=['test_only', 'buffer', 'noise', 'batch_size', 'activation'])
