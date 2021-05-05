#! /usr/bin/python3

from matplotlib import pyplot as pl, dates as dt_mp
import matplotlib as mpl
from numpy import loadtxt, arange
from reader import getFolders
from os.path import exists
from re import search
from writter import printTable

setInit = False

def init():
	mpl.rcParams['agg.path.chunksize'] = 10000
	return True

def plot2D(curves, svFormat='png', saveAt='fig', figsize=(20,10), title='Plot',
	labelPos='upper right', addLabels=False, axisLabel=[], dpi=200):
	global setInit
	if not setInit:
		setInit = init()
	pl.figure(figsize=figsize)
	pl.title(title,fontsize=40)
	if addLabels:
		for x,y,style,label in curves:
			pl.plot(x,y, style, label=label)
		pl.legend(loc=labelPos,fontsize=20)
	else:
		for x,y,style in curves: pl.plot(x,y, style)
	pl.tick_params(labelsize=20)
	if len(axisLabel):
		pl.ylabel(axisLabel.pop(), fontsize=25)
		pl.xlabel(axisLabel.pop(), fontsize=25)
	saveAt += '.' + svFormat
	pl.tight_layout()
	pl.savefig(saveAt, format=svFormat, dpi=dpi)
	pl.close()

#tentar limpar o código abaixo
def plfromTab(filenames, cols, style='-.', svFormat='png', saveAt='fig', figsize=(20,10),
	title='plot', labelPos='upper right', labels=[], id_curve=False, dpi=200):
	global setInit
	if not setInit:
		setInit = init()
	pl.figure(figsize=figsize)
	pl.title(title,fontsize=40)
	entries = []
	if id_curve: count = 1
	for filename in filenames:
		table = loadtxt(filename, comments="#", delimiter=" ")
		x = table[:,cols[0]]
		y = table[:,cols[1]]
		if len(cols) == 3:
			err = table[:,cols[2]]
			entries.append([x,y,err])
		else:
			entries.append([x,y])
		if id_curve:
			i = -1
			while filename[i] != '/': i -= 1
			dir_name = filename[:i]
			fd = open(saveAt + '_ID.txt','a' if exists(saveAt + '_ID.txt') else 'w')
			fd.write('curve_{}: {}\n'.format(count,dir_name))
			fd.close()
			count += 1
	if id_curve: count = 1
	if not len(labels):
		fd = open(filename[0], 'r')
		header = fd.readline()
		header = header[1:].split()
		fd.close()
		x_label = header[cols[0]]
		y_label = header[cols[1]]
		if len(cols) == 3:
			err_label = header[cols[2]]
	else:
		if len(cols) == 3:
			err_label = labels.pop()
		y_label = labels.pop()
		x_label = labels.pop()
	pl.xlabel(x_label, fontsize=25)
	pl.ylabel(y_label, fontsize=25)
	if len(cols) == 3:
		for x,y,err in entries:		#nao implementei o acrescimo de id's
			pl.errorbar(x,y,yerr=err,fmt=style, solid_capstyle='projecting', capsize=5,
				label=err_label)
		pl.legend(loc=labelPos,fontsize=20)
	else:
		for x,y in entries:
			if id_curve:
				pl.plot(x,y,style,label='curve_{}'.format(count))
				count += 1
			else:
				pl.plot(x,y,style)
	if id_curve:
		pl.legend(loc=labelPos,fontsize=20)
	saveAt += '.' + svFormat
	pl.grid(True)
	pl.tight_layout()
	pl.savefig(saveAt, format=svFormat, dpi=dpi)
	pl.close()

def barFromTab(filename, cols, svFormat='png', saveAt='fig', figsize=(20,10), title='plot',
	labelPos='upper left', labels=[], ignoreAE=False, allAE=False,dpi=200, hpFile=None,
	addT_line=False, maxBar=None, rotate=False, err=None):
	global setInit
	if not setInit:
		setInit = init()
	x_labels = []
	if hpFile:
		fd = open(hpFile,'r')
		for line in fd:
			if not allAE:
				if search('^AE Config', line) and (not ignoreAE):
					x_labels = ['AE']
				if search('^Dec_Config',line):
					config = '[' + ','.join(line.split()[-1].split('-')) + ']'
					x_labels.append(config)
			else:
				if search('^AE Config', line):##ajeitar para o t5_taxo
					config = '[' + ','.join(line.split()[-1].split('-')) + ']'
					x_labels.append(config)
				elif search('^(C?AE|HIB)\-[0-9]+',line) or search('^(S|R)?A?AE(\-[0-9]+\.?[0-9]?)?',line):
					config = line.split()[0]
					x_labels.append(config)
		fd.close()
	table = loadtxt(filename, comments='#', delimiter=' ')
	if not len(x_labels):
		x_labels = [int(i) if i.isdigit() else i for i in table[:,cols[0]].tolist()]
		y_vals = table[:,cols[1]].tolist()
	else:
		y_vals = table[:,cols[0]].tolist()
	if ignoreAE:
		y_vals = y_vals[1:]
	if err:
		err_vals = table[:,err].squeeze().tolist()
	if maxBar:
		all_labels = x_labels
		for i in range(1,int(len(all_labels)/maxBar)+1):
			x_labels = all_labels[(i-1)*maxBar:i*maxBar]
			put_ys = y_vals[(i-1)*maxBar:i*maxBar]
			if err:
				put_err = err_vals[(i-1)*maxBar:i*maxBar]
			x_pos = arange(len(x_labels))
			pl.figure(figsize=figsize)
			pl.title(title,fontsize=40)
			if err:
				pl.errorbar(x_pos,put_ys,yerr=put_err,fmt='k.',capsize=10)
			pl.bar(x_pos,put_ys,color='b')
			pl.grid()
			if rotate:
				pl.xticks(x_pos,x_labels,fontsize=25, rotation=45)#17.5)
			else:
				pl.xticks(x_pos,x_labels,fontsize=25)#17.5)
			pl.yticks(fontsize=25)#17.5)
			if not allAE:
				if (not ignoreAE) and addT_line:
					pl.plot([-x_pos[1],x_pos[-1]+x_pos[1]],[y_vals[0],y_vals[0]],'k--', linewidth=2)
					pl.xlim([-x_pos[1]/2,x_pos[-1]+x_pos[1]/2])
			if isinstance(labels,list):
				pl.xlabel(labels[0],fontsize=35)
				pl.ylabel(labels[1],fontsize=35)
			saveAtAux = saveAt + '{}.'.format(i) + svFormat
			pl.tight_layout()
			pl.savefig(saveAtAux,format=svFormat,dpi=dpi)
			pl.close()
	else:
		x_pos = arange(len(x_labels))
		pl.figure(figsize=figsize)
		pl.title(title,fontsize=40)
		if err:
			pl.errorbar(x_pos,y_vals,yerr=err_vals,fmt='k.',capsize=10)
		pl.bar(x_pos,y_vals,color='b')
		pl.grid()
		if rotate:
			pl.xticks(x_pos,x_labels,fontsize=25, rotation=45)#17.5)
		else:
			pl.xticks(x_pos,x_labels,fontsize=25)#17.5)
		pl.yticks(fontsize=25)#17.5)
#		pl.xticks(x_pos,x_labels,fontsize=17.5)
#		pl.yticks(fontsize=17.5)
		if not allAE:
			if (not ignoreAE) and addT_line:
				pl.plot([-x_pos[1],x_pos[-1]+x_pos[1]],[y_vals[0],y_vals[0]],'k--', linewidth=2)
				pl.xlim([-x_pos[1]/2,x_pos[-1]+x_pos[1]/2])
		if isinstance(labels,list):
			pl.xlabel(labels[0],fontsize=35)
			pl.ylabel(labels[1],fontsize=35)
		saveAt += '.' + svFormat
		pl.tight_layout()
		pl.savefig(saveAt,format=svFormat,dpi=dpi)
		pl.close()

def itBar(filename, svFormat='png', saveAt='fig', figsize=(20,10), title='plot', bar_wid=None,#set for three bars
	labelPos='upper right', labels=[], ignoreAE=False, allAE=False, dpi=200, hpFile=None, dontSpread=False):
	x_labels, table, cols_lab = printTable(filename,hpFile,return_vals=True)
	if not x_labels:
		x_labels = [str(i+1) for i in range(table.shape[0])]
	print('Select columns:')
	cols = []
	print('Select up to 3 columns')
	while(True):
		i = int(input('Type column value (0 through {}, type any value outside range to exit): '.format(table.shape[1]-1)))
		if i < 0 or i > len(table)-1:
			print('Exiting column selection')
			break
		else:
			if i in cols:
				print('Column is already selected!!!')
			else:
				op = input('You selected {} column. Do you want to change? [Y]es/[N]o: '.format(cols_lab[i])).upper()
				if op[0] == 'N':
					cols.append(i)
		print('You\'ve alerady selected {} columns'.format(len(cols)))
		if len(cols) == 3:
			print('Exiting column selection')
			break
	x_pos = arange(len(x_labels))
	pl.figure(figsize=figsize)
	pl.title(title,fontsize=40)
	pl.xlabel(labels[0], fontsize=35)
	pl.ylabel(labels[1], fontsize=35)
	pl.grid()
	pl.xticks(x_pos,x_labels,fontsize=25)#17.5)
	pl.yticks(fontsize=25)#17.5)
	bar_left = table[:,cols[0]].tolist()
	if len(cols) > 2:
		bar_middle = table[:,cols[1]].tolist()
	bar_right = table[:,cols[-1]].tolist()
	if len(cols) == 1:
		bar_right = bar_middle = None
	elif len(cols) == 2:
		bar_middle = None
	r_label = None
	if bar_right:
		r_label = input('Input right bar label: ')
	if bar_middle:
		m_label = input('Input middle bar label: ')
		if not bar_wid:
			bar_wid = .3
	if r_label:
		l_label = input('Input left bar label: ')
		if not bar_wid:
			bar_wid = .4
	if dontSpread:
		if bar_right:
			pl.bar(x_pos, bar_right, color='navy', align='center', label=r_label)
		if bar_middle:
			pl.bar(x_pos, bar_middle, color='b', align='center', label=m_label)
		pl.bar(x_pos,bar_left, color='c', align='center', label=l_label)
	else:
		if bar_middle:
			pl.bar(x_pos-bar_wid, bar_left, width=bar_wid, color='g', align='center', label=l_label)
			pl.bar(x_pos, bar_middle, width=bar_wid, color='b', align='center', label=m_label)
			pl.bar(x_pos+bar_wid, bar_right, width=bar_wid, color='r', align='center', label=r_label)
		elif bar_right:
			pl.bar(x_pos-bar_wid/2, bar_left, width=bar_wid, color='b', align='center', label=l_label)
			pl.bar(x_pos+bar_wid/2, bar_right, width=bar_wid, color='r', align='center', label=r_label)
		else:
			pl.bar(x_pos,bar_left)
	saveAt += '.' + svFormat
	pl.tight_layout()
	pl.legend(loc=labelPos, fontsize=25)
	pl.savefig(saveAt,format=svFormat,dpi=dpi)
	pl.close()

def time_table(table, cols, start, inc, style, svFormat='png', saveAt='fig',
	figsize=(20,10), title='Plot', labelPos='upper right', labels=[], axisLabels=[],
	dpi=200, split=None, pt=False):
	global setInit
	if not setInit:
		setInit = init()
	tab = loadtxt(table,comments='#', delimiter=' ')
	#print(tab)
	curves = []
	for i in cols:
		curves.append(tab[:,i])
	time_vals = [start + i*inc for i in range(tab.shape[0])]
	if not split:
		split = 1
	for i in range(1,int(len(time_vals)/split)):
		pl.figure(figsize=figsize)
		pl.title(title,fontsize=40)
		st = (i-1)*split; fn = i*split
		#print(time_vals[st:st+10]); input()
		for j,curve in enumerate(curves):
			if len(labels):
				pl.plot_date(time_vals[st:fn], curve[st:fn], style[j], label=labels[j])
			else:
				pl.plot_date(time_vals[st:fn], curve[st:fn], style[j])
		pl.xlim(time_vals[st],time_vals[fn])
		pl.gcf().autofmt_xdate()
		if pt:
			date_format = dt_mp.DateFormatter('%d %b %Y')
		else:
			date_format = dt_mp.DateFormatter('%b, %d %Y')
		pl.gca().xaxis.set_major_formatter(date_format)
		pl.legend(loc=labelPos,fontsize=20)
		pl.tick_params(labelsize=20)
		if len(axisLabels):
			#pl.xlabel(axisLabels[0], fontsize=25)
			pl.ylabel(axisLabels[1], fontsize=25)
		pl.grid()
		pl.tight_layout()
		pl.savefig(saveAt+'_{}'.format(i) + '.' + svFormat, format=svFormat, dpi=dpi)
		pl.close()

if __name__ == '__main__':
	table = './Ad_Plot/tableyc.txt'; helper = './Ad_Plot/table_helperyc.txt'

	print('Generating plots...')
#	import datetime
#	z_range=500
#	filename = './Ad_Plot/AE_5.txt'
#	saveAt = './Aux/AE_5_zoom'
#	time_table(filename,[1,2,3,4],datetime.datetime(year=2016,month=7,day=1),datetime.timedelta(minutes=15),['k--','g-','r-','b-'],
#		labels=['Original', 'Best', 'Worst', 'Random'], dpi=500, split=z_range, title=None,#'Signal Reconstruction',
#		axisLabels=['Data','Temperature (ᵒC)'], saveAt=saveAt)#, pt=True)

#	itBar(table,hpFile=helper,title=None,#'Error per Configuration',
#		labels=['Network Configuration', 'Mean Error (MSE)'], saveAt='yc_res',dpi=500)#,dontSpread=True)

	barFromTab(table, [0], saveAt='SAAEy_res2',title=None,#'Erro por Configuração',
		labels=['Network Configuration', 'Mean Error (MSE)'],
		hpFile=helper,dpi=500,addT_line=False, allAE=True)#, rotate=True)#, maxBar=6)

#	filename = '/home/mateusgilbert/trabalhos/IC/Prog/Proj2020/m0dev05/AEEx_2020_07_24_10_47_44/table.txt'
#	filenames = getFolders('noisy test samples using sliding windows', 'This_is_an_example.txt')
#	print(filenames)
#	for i in range(len(filenames)):
#		filenames[i] += '/table.txt'
#	print(filenames)
#	plfromTab(filenames, [0, 1], labels=['num. neurons', 'mean loss'],
#		title='Reconstruction Loss vs. Number of Neurons',saveAt='results',id_curve=True,dpi=500)
#	import numpy as np
#
#	x = np.arange(-5,5)
#	y_1 = x**1
#	y_2 = x**2
#	y_3 = x**3
#	y_4 = x**4
#	curves = [(x,y_1,'r-','x'),(x,y_2,'g--','x²'),(x,y_3,'yo','x³'),(x,y_4,'b^','x⁴')]
#	curves2 = [(x,y_1,'r-'),(x,y_2,'g--'),(x,y_3,'yo'),(x,y_4,'b^')]
#	axisLabel = [('x axis'),('y axis')]
#
#	plot2D(curves, title='Polynomials', saveAt='test', addLabels=True, axisLabel=axisLabel)
#	plot2D(curves2)
#	plfromTab("tab1.txt",cols = [0,1,2],saveAt='fromTab')
print('...done')
