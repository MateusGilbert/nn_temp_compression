#! /usr/bin/python3

import os
import errno
from shutil import move as mv
import datetime as dttm
from testImports import *
from training import aeTrain, dnetTrain
from tensorflow.keras.activations import relu, sigmoid, tanh
from timeSerDataAug import jittering, flipping, scaling
from testNet import ae_test_dts, dec_test_dts

#Sample/Compression##########################
smp_size = 100
cmp_size = 25
#AE specifications###########################
out_func = 'sigmoid'#'tanh'
hid_act = 'selu'#'relu'
hid_act2 = 'relu'
enc_act = None#'c_sigmoid'
#addBatchNorm = True if hid_act == 'relu' else False
#addBatchNorm = (addBatchNorm,False)
func_range = (0,1)		#(-1,1)
perc_th = 0#5e-4 #.05% dos limites das funções de ativação
round_range = func_range#(perc_th,func_range[1]*(1.-perc_th))
ae_clr_const = 5			#clr stepsize constant
ae_mode = 'exp_range' #'triangular2'
ae_patience = 15		#for an ae_clr_const == 5, we have a cycle and a half
ae_fix = None#100
tempAE = 'tmpAE.h5'
bestAE = 'bestAE.h5'
worstAE = 'worstAE.h5'
randomAE = 'randomAE.h5'
#Models to be trained####################
models = [
	('AE-1', [('in',(smp_size,),None), ('dl',cmp_size,hid_act), ('dl',smp_size,out_func)]),
	('AE-2', [('in',(smp_size,),None), ('dl',60,hid_act), ('dl',cmp_size,hid_act), ('dl',60,hid_act), ('dl',smp_size,out_func)]),
	('AE-3', [('in',(smp_size,),None), ('dl',65,hid_act), ('dl',cmp_size,hid_act), ('dl',65,hid_act), ('dl',smp_size,out_func)]),
	('AE-4', [('in',(smp_size,),None), ('dl',75,hid_act), ('dl',50,hid_act), ('dl',cmp_size,hid_act), ('dl',50,hid_act), ('dl',75,hid_act), ('dl',smp_size,out_func)]),
	('AE-5', [('in',(smp_size,),None), ('dl',85,hid_act), ('dl',65,hid_act), ('dl',45,hid_act), ('dl',cmp_size,hid_act),
				('dl',45,hid_act), ('dl',65,hid_act), ('dl',85,hid_act), ('dl',smp_size,out_func)]),
	('AAE-1', [('in',(smp_size,),None), ('dl',cmp_size,hid_act), ('dl',60,hid_act), ('dl',smp_size,out_func)]),
	('AAE-2', [('in',(smp_size,),None), ('dl',cmp_size,hid_act), ('dl',65,hid_act), ('dl',smp_size,out_func)]),
	('AAE-3', [('in',(smp_size,),None), ('dl',cmp_size,hid_act), ('dl',50,hid_act), ('dl',75,hid_act), ('dl',smp_size,out_func)]),
	('AAE-4', [('in',(smp_size,),None), ('dl',cmp_size,hid_act), ('dl',45,hid_act), ('dl',65,hid_act), ('dl',85,hid_act), ('dl',smp_size,out_func)])
]
#General#####################################
wnd_stride = 10
wnd_str_comp = [17,23]
test_stride = smp_size
turns_per_config = 10
batch_size = 20
test_size = .2
usingNest = True
compType = 'temporal'
dts_name = ['YandHalfChallenge/' + name for name in ['Caples_Lake_N7_2014_20162.csv', 'Caples_Lake_N7_2016_2017.csv']] #'Caples_Lake_N7_2014_2017.csv'
cols = [0,1]
if isinstance(dts_name,list):
	dts_loc = ['/home/mateusgilbert/trabalhos/IC/Prog/Proj2020/Datasets/' + name for name in dts_name]
	rFolder = '{}_results'.format(dts_name[0][:-6])
else:
	dts_loc = '/home/mateusgilbert/trabalhos/IC/Prog/Proj2020/Datasets/' + dts_name
	rFolder = '{}_results'.format(dts_name[:-6])
table_name = 'table.txt'
conf_table = '{}_helper.txt'.format(table_name[:-4])
#the file above keeps of the entries in table.txt
filename = 'readme.txt'
#Configuration settings for LR finder
start_lr = 1e-4
end_lr = 1.
lr_epochs = 10
#for a better vizualization of the plots
zoom = True
z_range = 1000
#Data Augmentation
jitt = None#([.25,.5], [.6,.4],.333)
flip = None#.005
scal = None#.125
combine = None
#############################################

def t2_taxonomy(lr_list=None,dir_id='tx_results_',verbose=False):
	try:
		os.makedirs(rFolder)
	except OSError as err:
		if err.errno != errno.EEXIST:
			raise

	os.chdir(rFolder)
	execDirname = dir_id + dttm.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
	os.mkdir(execDirname)
	os.chdir(execDirname)

	#generate datasets -- specific to my dataset
	(train_dts,num_batches),(test_dts,num_ts_batches) = initDataset(dts_loc,
																						cols,
																						smp_size,
																						test_size,
																						wnd_stride,
																						test_stride,
																						wnd_str_comp=wnd_str_comp)#,
																						#combine=combine)
	aug_tec = []
	#scaling
	if scal:
		train_dts,num_batches = scaling(train_dts,scale_y=True,addProb=scal)
		aug_tec.append('Scaling ({:.3f})'.format(scal))
	#flipping
	if flip:
		train_dts,num_batches = flipping(train_dts,addProb=flip)
		aug_tec.append('Flipping ({:.3f})'.format(flip))
	#jittering
	if jitt:
		train_dts,num_batches = jittering(train_dts, sigma=jitt[0],
			prob=jitt[1], addProb=jitt[2])
		aug_tec.append('Jittering ({:.3f}) with σ = {} with probabilities {}, respectively'.format(
			jitt[2],jitt[1], jitt[0]))
	if not len(aug_tec):
		aug_tec = ['None']

	unscalled_dts = train_dts
	#scale each training batch inside function output range and shuffle them
	train_dts = scaleBatches(
		train_dts,dRange=round_range,s_ipts=True,s_tgs=True).shuffle(buffer_size=num_batches)

	inputs = []
	targets = []
	for x,y in train_dts:
		inputs.append(x.numpy()[nax,:])
		targets.append(y.numpy()[nax,:])
	inputs = concat(inputs,0)
	targets = concat(targets,0)

	if usingNest:
		optFunc = Nadam
	else:
		optFunc = Adam

	results_dir = os.getcwd()
	fd = open(table_name,'w')
	fd.write("#mean stdD max min meanIt\n")
	fd2 = open(conf_table,'w')
	fd2.write('This file keeps track of each AE  configuration.\n')
	fd2.write('Each line presents the number of neurons per layers.\n')
	fd2.write('Each line is correspond to each line of {}, in order.\n'.format(table_name))
	fd2.write('-'*20 + '\n')
	fd2.write('From {} down to {}\n'.format(smp_size,cmp_size))
	fd2.write('-'*20 + '\n')

	for net_id,model in models:
		fd2.write('{} (model list): {}\n'.format(net_id,model))
		config_dirname = '_'.join(net_id.split('-'))
		print('Training Config.: {}'.format(net_id))
		os.makedirs(config_dirname)
		os.chdir(config_dirname)

		#define ae
		if lr_list and net_id in lr_list.keys():
			(min_lr,max_lr) = lr_list[net_id]
		else:
			ae = buildNN2(model)
			ae.compile(loss='mse', optimizer=optFunc(), metrics=['mse','mae'])

			lr_finder = LRFinder(min_lr=start_lr, max_lr=end_lr)
			ae.fit(inputs,targets,batch_size=batch_size,callbacks=[lr_finder],epochs=lr_epochs)
			min_lr = float(input('Insert minimum value for Learning Rate: '))
			max_lr = float(input('Insert maximum value for Learning Rate: '))
			while(True):
				print('min_lr = {:.2g}; max_lr = {:.2g}'.format(min_lr,max_lr))
				op = input("Want to change one of the limits? [Y]es/[N]o: ")
				if op[0].upper() == 'N':
					break
				op = int(input('\t[0] for min_lr\n\t[1] for max_lr\n\t[2]Both\nOption:'))
				if not op % 2:
					min_lr = float(input('Insert minimum value for Learning Rate: '))
					if op == 2:
						max_lr = float(input('Insert maximum value for Learning Rate: '))
				else:
					max_lr = float(input('Insert maximum value for Learning Rate: '))
			del ae

		#metrics initialization
		best_outputs = []
		worst_outputs = []
		random_outputs = []
		test_loss_avg = Mean()
		iteration_avg = Mean()
		test_losses = []
		for i in range(turns_per_config):
			print('>>>{:03d} Turn'.format(i))
			iteration = aeTrain(model,#layers.copy(),
										batch_size,
										train_dts,
										#spec,
										(min_lr,max_lr,ae_mode),
										#out_act=out_func,
										#dropout = config_dropout.copy(),
										k=ae_clr_const,
										patience=ae_patience,
										saveAt=tempAE,
										verbose=verbose,
										wait=ae_fix,
										optimizer=optFunc)#,
										#addBatchNorm=addBatchNorm))

			iteration_avg.update_state(iteration)
			print('It took {} iterations'.format(iteration))
			#load best weight configuration
			ae = load_model(tempAE)
			os.remove(tempAE)

			#test
			cur_outputs = []				#parei aqui
			it_test_loss_avg = Mean()
			if isinstance(test_dts,list):
				for name,dataset in test_dts:
					if verbose:
						print('-'*30)
						print('Results from: {}'.format(name))
						print('-'*30)
					aux_outputs, res = ae_test_dts(ae,dataset,func_range=round_range,test_stride=test_stride,verbose=verbose)
					#it_test_loss_avg.update_state(res)
					cur_outputs.append(aux_outputs)
			else:
				aux_outputs, res = ae_test_dts(ae,test_dts,func_range=round_range,test_stride=test_stride,verbose=verbose)
			it_test_loss_avg.update_state(res)
			cur_outputs = aux_outputs

			#update results
			curTestLoss = it_test_loss_avg.result().numpy()
			if len(best_outputs):
				if minLoss > curTestLoss:
					best_outputs = cur_outputs.copy()
					minLoss = curTestLoss
					ae.save(bestAE)
				elif maxLoss < curTestLoss:
					worst_outputs = cur_outputs.copy()
					maxLoss = curTestLoss
					ae.save(worstAE)
				elif not np.random.randint(0,100) % 5:			#.2 prob of selecting results
					random_outputs = cur_outputs.copy()
					ae.save(randomAE)
			else:
				best_outputs = random_outputs = worst_outputs = cur_outputs.copy()
				minLoss = maxLoss = curTestLoss
				ae.save(bestAE); cp(bestAE,worstAE); cp(bestAE,randomAE)
			test_losses.append(curTestLoss)
			test_loss_avg.update_state(curTestLoss)
			del ae

		##save results
		if isinstance(best_outputs[0],list):
			saveOg = os.getcwd()
			for i, (name, dataset) in enumerate(test_dts):
				os.makedirs(name)
				os.chdir(name)
				x_vals = []; y_vals = []
				for x,y in dataset:
					x_vals += x.numpy().squeeze().tolist()
					y_vals += y.numpy().squeeze().tolist()
				b_outputs = best_outputs[i]
				rem_samples = len(b_outputs)
				a_outputs = ae_outputs[i][:rem_samples]
				x_vals = x_vals[:rem_samples]
				x_vals = [i-x_vals[0] for i in x_vals]
				y_vals = y_vals[:rem_samples]
				if turns_per_config == 1:
					plots = [(x_vals, y_vals, 'k--', 'original'),
								(x_vals, b_outputs, 'b-', 'd-decompressor')]
				else:
					w_outputs = worst_outputs[i]
					r_outputs = random_outputs[i]
					plots = [(x_vals, y_vals, 'k--', 'original'),
								(x_vals, b_outputs, 'b-', 'best'),
								(x_vals, r_outputs, 'g-', 'random'),
								(x_vals, w_outputs, 'r-', 'worst')]
				title = 'AE Compression (from {:d} down to {:d})'.format(smp_size,cmp_size)
				saveAt = '_'.joint(net_id.split('-'))
				#for layer in layers: saveAt += '-{}'.format(layer)
				plot2D(plots,title=title,saveAt=saveAt,addLabels=True,dpi=500)
				if turns_per_config == 1:
					entriesT = (x_vals,y_vals, b_outputs)
					tLabels = ('instant','original','d-decompressor')
				else:
					entriesT = (x_vals, y_vals,b_outputs, w_outputs, r_outputs)
					tLabels = ('instant','original','best', 'worst', 'random')
				wTable(entriesT, labels=tLabels, table=saveAt+'.txt')
				if zoom and rem_samples > z_range:
					zoom_dir = 'Zoom_plots'
					os.mkdir(zoom_dir)
					os.chdir(zoom_dir)
					for i in range(1,int(rem_samples/z_range) + 1):
						if turns_per_config == 1:
							plots = [(x_vals[(i-1)*z_range:i*z_range], y_vals[(i-1)*z_range:i*z_range], 'k--', 'original'),
										(x_vals[(i-1)*z_range:i*z_range], b_outputs[(i-1)*z_range:i*z_range], 'b-', 'd-decompressor')]
						else:
							plots = [(x_vals[(i-1)*z_range:i*z_range], y_vals[(i-1)*z_range:i*z_range], 'k--', 'original'),
										(x_vals[(i-1)*z_range:i*z_range], b_outputs[(i-1)*z_range:i*z_range], 'g-', 'best'),
										(x_vals[(i-1)*z_range:i*z_range], w_outputs[(i-1)*z_range:i*z_range], 'r-', 'worst'),
										(x_vals[(i-1)*z_range:i*z_range], r_outputs[(i-1)*z_range:i*z_range], 'b-', 'random')]
						saveAt_zoom = saveAt + '_zoom_{}'.format(i)
						plot2D(plots,title=title,saveAt=saveAt_zoom,addLabels=True,dpi=500)
				os.chdir('..')
		else:
			rem_samples = len(best_outputs)	#needed because when using test windows, as some
														#samples are droped
			x_vals = []; y_vals = []
			for x,y in test_dts:
				x_vals += x.numpy().squeeze().tolist()
				y_vals += y.numpy().squeeze().tolist()
			x_vals = x_vals[:rem_samples]
			y_vals = y_vals[:rem_samples]
			#plot curves and save tables
			if turns_per_config == 1:
				plots = [(x_vals, y_vals, 'k--', 'original'),
							(x_vals, best_outputs, 'b-', 'd-decocompressor')]
			else:
				plots = [(x_vals, y_vals, 'k--', 'original'),
							(x_vals, best_outputs, 'g-', 'best'),
							(x_vals, worst_outputs, 'r-', 'worst'),
							(x_vals, random_outputs, 'b-', 'random')]
			title = 'AE Compression (from {:d} down to {:d})'.format(smp_size,cmp_size)
			saveAt = '_'.join(net_id.split('-'))
			#for layer in layers: saveAt += '-{}'.format(layer)
			plot2D(plots,title=title,saveAt=saveAt,addLabels=True,dpi=500)
			if turns_per_config == 1:
				entriesT = (x_vals, y_vals,best_outputs)
				tLabels = ('instant','original','d-decompressor')
			else:
				entriesT = (x_vals, y_vals,best_outputs, worst_outputs, random_outputs)
				tLabels = ('instant','original','best', 'worst', 'random')
			wTable(entriesT, labels=tLabels, table=saveAt+'.txt')
			if zoom and rem_samples > z_range:
				zoom_dir = 'Zoom_plots'
				os.mkdir(zoom_dir)
				os.chdir(zoom_dir)
				for i in range(1,int(rem_samples/z_range) + 1):
					if turns_per_config == 1:
						plots = [(x_vals[(i-1)*z_range:i*z_range], y_vals[(i-1)*z_range:i*z_range], 'k--', 'original'),
									(x_vals[(i-1)*z_range:i*z_range], best_outputs[(i-1)*z_range:i*z_range], 'b-', 'd-decompressor')]
					else:
						plots = [(x_vals[(i-1)*z_range:i*z_range], y_vals[(i-1)*z_range:i*z_range], 'k--', 'original'),
									(x_vals[(i-1)*z_range:i*z_range], best_outputs[(i-1)*z_range:i*z_range], 'g-', 'best'),
									(x_vals[(i-1)*z_range:i*z_range], worst_outputs[(i-1)*z_range:i*z_range], 'r-', 'worst'),
									(x_vals[(i-1)*z_range:i*z_range], random_outputs[(i-1)*z_range:i*z_range], 'b-', 'random')]
					saveAt_zoom = saveAt + '_zoom_{}'.format(i)
					plot2D(plots,title=title,saveAt=saveAt_zoom,addLabels=True,dpi=500)
				os.chdir('..')
		meanLoss = test_loss_avg.result().numpy()
		os.chdir('..')
		fd.write("{:.5E} {:.5E} {:.5E} {:.5E} {:.2f}\n".format(meanLoss, stdDev(test_losses,meanLoss),
			maxLoss, minLoss, iteration_avg.result().numpy()))
	fd.write("\n")
	fd.close()
	fd2.close()
	os.chdir(results_dir)
	if len(models) > 8:
		num = len(models)
		if num % 7 > 3 or num % 7 == 0:
			num = 7
		elif num % 6 or num % 6 == 0:
			num = 6
		elif num % 5 > 2 or num %  5 == 0:
			num = 5
		elif num % 2 == 0:
			num = 4 if (num % 4) == (num % 8) else 8
		else:
			num = None
		if num:
			barFromTab(table_name,[0],saveAt='results_MSE_s',hpFile=conf_table,title='Loss per Configuration',
						labels=['Network config.', 'mean loss (MSE)'],ignoreAE=False,dpi=500,allAE=True,maxBar=num)
	barFromTab(table_name,[0],saveAt='results_MSE',hpFile=conf_table,title='Loss per Configuration',
		labels=['Network config.', 'mean loss (MSE)'],ignoreAE=False,dpi=500,allAE=True)

	##create a readme with description
	entries = [compType,								#if it is Temporal or Spatial
					smp_size,							#samples per compression
					cmp_size,							#compressed size
					ae_mode,								#ae clr mode
					ae_clr_const,						#ae clr stepsize const.
					dts_name[0][:-6],					#what is the dataset
					smp_size,							#batch_size (samples per batch)
					out_func + custom_sigmoid('get const') if out_func == 'c_sigmoid' else out_func,
					perc_th,								#function limits
					enc_act + custom_sigmoid('get const') if hid_act == 'c_sigmoid' else hid_act,
					hid_act,								#hidden layers activation
					'Non specified',					#dropout values
					turns_per_config,					#turns per configuration
					'Nadam' if usingNest else 'Adam',
					batch_size,							#training batch size
					ae_patience,						#earlystopping patience
					wnd_stride,							#testing window stride
					wnd_str_comp,
					#dd_mode,								#clr mode
					#dd_clr_const,						#clr stepsize constant
					aug_tec,								#data augmentation techniques
					smp_size-test_stride]#,			#number of samples that are
															#retained at the buffers
					#False]#noise_spec]							#noise
	entries.reverse()
	#if wait:
		#wReadme2(entries, 'dn', wait=100, saveAt=filename)		#adicionar variável
	#else:
		#wReadme(entries, 'dn', saveAt=filename)
	wReadme2(entries, wait=ae_fix, saveAt=filename)
	return results_dir

if __name__ == '__main__':
	print('Checking if models are viable')
	print('There are {} models'.format(len(models)))
	for net_id,model in models:
		print('\nNet id {}'.format(net_id))
		try:
			ae = buildNN2(model)
		except:
			for i in range(1,len(model)+1):
				ae = buildNN2(model[:i])
				ae.summary()
		ae.summary()
		del ae
	print('Success!!!')
