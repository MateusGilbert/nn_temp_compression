#! /usr/bin/python3

from numpy import cos, sin, arctan, linspace, array, loadtxt
from sys import maxsize
import csv

csv.field_size_limit(maxsize)

def f1(x,rLiteral=True):
	if not (rLiteral): return 2*(sin(2.*x)+sin(x/2.))
	return (2*(sin(2.*x)+sin(x/2.)), "2[sin(2x)+sin(x/2)]")

def f2(x,rLiteral=True):
	if not (rLiteral): return 2*(cos(x)+1.)*sin(2*x)
	return (2*(cos(x)+1.)*sin(2*x), "2(cos(x)+1)sin(2x)")

def f3(x,rLiteral=True):
	if not (rLiteral): return 2*(cos(x+1.)*sin(2*x))+arctan(.5*x)
	return (2*(cos(x+1.)*sin(2*x))+arctan(.5*x),'2(cos(x)+1)sin(2x) + arctan(.5*x)')

def readtxt(filename,cols=None):
	with open(filename, newline='\n') as txtfile:
		if cols:
			table = loadtxt(txtfile, delimiter=' ')[:,cols]
		else:
			table = loadtxt(txtfile, delimiter=' ')
	return table

#mudar essa função (ver o cols)
def csvFl(filename,delimiter=',',cols=None):						#very specific to my needs
	with open(filename, newline='\n') as csvfile:
		table = csv.reader(csvfile, delimiter=delimiter)
		dts_raw = []
		dts_treated = []
		count = 0
		for row in table:
			line_1 = []
			if count:
				line_2 = [count-1]
			for i in cols:
				line_1.append(row[i])
				if count:
					if i:
						line_2.append(float(row[i]))
			dts_raw.append(line_1.copy())
			if count:
				dts_treated.append(line_2.copy())
			count += 1
	labels = dts_raw[0]
	dts_raw = dts_raw[1:]
	return (labels,dts_raw,array(dts_treated))

def genDataset(function, interval=[0.,50.], numPts=int(2e3)):
	x = linspace(interval[0], interval[1], numPts)
	if function == 'f1':
		return x,f1(x)
	elif function == 'f2':
		return x,f2(x)
	elif function == 'f3':
		return x,f3(x)

if __name__ == '__main__':
	import matplotlib.pyplot as pl
	x,(y,func) = genDataset('f2',interval=[0.,20.])
	pl.figure(figsize=(20,10))
	pl.plot(x,y)
	pl.title(func, fontsize=40)
	pl.grid(True)
	pl.savefig('f2.png',format='png',dpi=400)
	pl.figure(figsize=(20,10))
	x,(y,func) = genDataset('f3',interval=[0.,20.])
	pl.plot(x,y)
	pl.title(func, fontsize=40)
	pl.grid(True)
	pl.savefig('f3.png',format='png',dpi=400)
