import tensorflow as tf
import numpy as np
import sys, os
import getopt 
import random

from nn import * 
from func import *

#whether the data is stored in the sparse format 
if FLAGS.sparse:
	import scipy.sparse as sparse

NUM_ITERATIONS_PER_DECAY = 100

def train(the_nn, sess, train_step, D, I_valid, D_test, logfile, i_exp):

	n = D['x'].shape[0]
	I = range(n)
	I_train = list(set(I)-set(I_valid))
	n_train = len(I_train)
	p_treated = np.mean(D['t'][I_train,:])

	#train loss feed_dicts
	dict_factual = {the_nn.x: D['x'][I_train,:], the_nn.t: D['t'][I_train,:], the_nn.y_: D['yf'][I_train,:], \
	  the_nn.drop_in: 1.0, the_nn.drop_out: 1.0, the_nn.r_alpha: FLAGS.p_alpha, \
	  the_nn.r_lambda: FLAGS.p_lambda, the_nn.p_t: p_treated}
	#vali loss feed_dicts
	if FLAGS.val_part > 0:
		dict_valid = {the_nn.x: D['x'][I_valid,:], the_nn.t: D['t'][I_valid,:], the_nn.y_: D['yf'][I_valid,:], \
		  the_nn.drop_in: 1.0, the_nn.drop_out: 1.0, the_nn.r_alpha: FLAGS.p_alpha, \
		  the_nn.r_lambda: FLAGS.p_lambda, the_nn.p_t: p_treated}
	#counterfactual loss feed_dicts if exist
	if D['HAVE_TRUTH']:
		dict_cfactual = {the_nn.x: D['x'][I_train,:], the_nn.t: 1-D['t'][I_train,:], the_nn.y_: D['ycf'][I_train,:], \
		  the_nn.drop_in: 1.0, the_nn.drop_out: 1.0}

	sess.run(tf.global_variables_initializer())
	preds_train = []
	preds_test = []
	losses = []
	obj_loss, f_error, imb_err = sess.run([the_nn.tot_loss, the_nn.pred_loss,the_nn.imb_dist],\
	feed_dict=dict_factual)
	cf_error = np.nan
	if D['HAVE_TRUTH']:
		cf_error = sess.run(the_nn.pred_loss, feed_dict=dict_cfactual)
	valid_obj = np.nan
	valid_imb = np.nan
	valid_f_error = np.nan 
	if FLAGS.val_part > 0:
		valid_obj, valid_f_error, valid_imb = sess.run([the_nn.tot_loss, the_nn.pred_loss,the_nn.imb_dist],\
		  feed_dict=dict_valid)
	losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])

	objnan = False
	reps = []
	reps_test = []

	for i in range(FLAGS.iterations):
		#batch train 
		I = random.sample(range(0, n_train), FLAGS.batch_size)
		x_batch = D['x'][I_train,:][I,:]
		t_batch = D['t'][I_train,:][I]
		y_batch = D['yf'][I_train,:][I]

		#one gradient descent 
		if not objnan:
			sess.run(train_step, feed_dict={the_nn.x: x_batch, the_nn.t: t_batch, \
				the_nn.y_: y_batch, the_nn.drop_in: FLAGS.dropout_in, the_nn.drop_out: FLAGS.dropout_out, \
				the_nn.r_alpha: FLAGS.p_alpha, the_nn.r_lambda: FLAGS.p_lambda, the_nn.p_t: p_treated})

		#loss every N iterations
		if i % FLAGS.output_delay == 0 or i==FLAGS.iterations-1:
			obj_loss,f_error,imb_err = sess.run([the_nn.tot_loss, the_nn.pred_loss, the_nn.imb_dist],
				feed_dict=dict_factual)

			rep = sess.run(the_nn.h_rep_norm, feed_dict={the_nn.x: D['x'], the_nn.drop_in: 1.0})
			rep_norm = np.mean(np.sqrt(np.sum(np.square(rep), 1)))
			cf_error = np.nan
			if D['HAVE_TRUTH']:
				cf_error = sess.run(the_nn.pred_loss, feed_dict=dict_cfactual)
			valid_obj = np.nan
			valid_imb = np.nan
			valid_f_error = np.nan
			if FLAGS.val_part > 0:
				valid_obj, valid_f_error, valid_imb = sess.run([the_nn.tot_loss, the_nn.pred_loss, the_nn.imb_dist], feed_dict=dict_valid)
			losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])
			loss_str = str(i) + '\tObj: %.3f,\tF: %.3f,\tCf: %.3f,\tImb: %.2g,\tVal: %.3f,\tValImb: %.2g,\tValObj: %.2f' \
						% (obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj)

			if FLAGS.loss == 'log':
				y_pred = sess.run(the_nn.output, feed_dict={the_nn.x: x_batch, \
					the_nn.t: t_batch, the_nn.drop_in: 1.0, the_nn.drop_out: 1.0})
				y_pred = 1.0*(y_pred > 0.5)
				acc = 100*(1 - np.mean(np.abs(y_batch - y_pred)))
				loss_str += ',\tAcc: %.2f%%' % acc
			log(logfile, loss_str)
			if np.isnan(obj_loss):
				log(logfile,'Experiment %d: Objective is NaN. Skipping.' % i_exp)
				objnan = True

		#predictions every M iterations 
		if (FLAGS.pred_output_delay > 0 and i % FLAGS.pred_output_delay == 0) or i==FLAGS.iterations-1:

			y_pred_f = sess.run(the_nn.output, feed_dict={the_nn.x: D['x'], \
				the_nn.t: D['t'], the_nn.drop_in: 1.0, the_nn.drop_out: 1.0})
			y_pred_cf = sess.run(the_nn.output, feed_dict={the_nn.x: D['x'], \
				the_nn.t: 1-D['t'], the_nn.drop_in: 1.0, the_nn.drop_out: 1.0})
			preds_train.append(np.concatenate((y_pred_f, y_pred_cf),axis=1))

			#predict on the test portion of the data
			if D_test is not None:
				y_pred_f_test = sess.run(the_nn.output, feed_dict={the_nn.x: D_test['x'], \
					the_nn.t: D_test['t'], the_nn.drop_in: 1.0, the_nn.drop_out: 1.0})
				y_pred_cf_test = sess.run(the_nn.output, feed_dict={the_nn.x: D_test['x'], \
					the_nn.t: 1-D_test['t'], the_nn.drop_in: 1.0, the_nn.drop_out: 1.0})
				preds_test.append(np.concatenate((y_pred_f_test, y_pred_cf_test),axis=1))

			if FLAGS.save_rep and i_exp == 1:
				reps_i = sess.run([the_nn.h_rep], feed_dict={the_nn.x: D['x'], \
					the_nn.drop_in: 1.0, the_nn.drop_out: 0.0})
				reps.append(reps_i)
				if D_test is not None:
					reps_test_i = sess.run([the_nn.h_rep], feed_dict={the_nn.x: D_test['x'], \
						the_nn.drop_in: 1.0, the_nn.drop_out: 0.0})
					reps_test.append(reps_test_i)
	return losses, preds_train, preds_test, reps, reps_test

def run(outdir, dataset_train, dataset_test, dataset_vali):
	npzfile = outdir+'result'
	npzfile_test = outdir+'result.test'
	repfile = outdir+'reps'
	repfile_test = outdir+'reps.test'
	outform = outdir+'y_pred'
	outform_test = outdir+'y_pred.test'
	lossform = outdir+'loss'
	
	has_test = False
	if dataset_test != None: 
		has_test = True

	random.seed(FLAGS.seed)
	tf.set_random_seed(FLAGS.seed)
	np.random.seed(FLAGS.seed) 
	save_config(outdir+'config.txt')
	log(logfile, 'Training with hyperparameters: alpha=%.2g, lambda=%.2g' % (FLAGS.p_alpha,FLAGS.p_lambda))
	log(logfile, 'Loaded data with shape [%d,%d]' % (dataset_train['n'], dataset_train['dim']))

	#session
	sess = tf.Session()
	x  = tf.placeholder("float", shape=[None, dataset_train['dim']], name='x') 
	t  = tf.placeholder("float", shape=[None, 1], name='t')   
	y_ = tf.placeholder("float", shape=[None, 1], name='y_') 
	r_alpha = tf.placeholder("float", name='r_alpha')
	r_lambda = tf.placeholder("float", name='r_lambda')
	drop_in = tf.placeholder("float", name='dropout_in')
	drop_out = tf.placeholder("float", name='dropout_out')
	p = tf.placeholder("float", name='p_treated')

	#graph
	log(logfile, 'Defining graph...\n')
	dims = [dataset_train['dim'], FLAGS.dim_rep, FLAGS.dim_reg]
	the_nn = nn(x, t, y_, p, FLAGS, r_alpha, r_lambda, drop_in, drop_out, dims)
	global_step = tf.Variable(0, trainable=False)
	lr = tf.train.exponential_decay(FLAGS.lrate, global_step, \
		NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)
	opt = None
	if FLAGS.optimizer == 'Adagrad':
		opt = tf.train.AdagradOptimizer(lr)
	elif FLAGS.optimizer == 'GradientDescent':
		opt = tf.train.GradientDescentOptimizer(lr)
	elif FLAGS.optimizer == 'Adam':
		opt = tf.train.AdamOptimizer(lr)
	else:
		opt = tf.train.RMSPropOptimizer(lr, FLAGS.decay)

	train_step = opt.minimize(the_nn.tot_loss, global_step=global_step)

	#set up before training
	all_losses = []
	all_preds_train = []
	all_preds_test = []
	all_valid = []
	if FLAGS.varsel:
		all_weights = None
		all_beta = None
	all_preds_test = []


	n_experiments = FLAGS.experiments
	for i_exp in range(1, n_experiments+1):
		log(logfile, 'Training on experiment %d/%d...' % (i_exp, n_experiments))
		if i_exp==1 or FLAGS.experiments>1:
			D_exp_test = None
			D_exp = None
			if npz_input:
				D_exp = {}
				D_exp['x']  = dataset_train['x'][:,:,i_exp-1] 
				D_exp['t']  = dataset_train['t'][:,i_exp-1:i_exp]
				D_exp['yf'] = dataset_train['yf'][:,i_exp-1:i_exp]
				D_exp['HAVE_TRUTH'] = dataset_train['HAVE_TRUTH']
				if dataset_train['HAVE_TRUTH']:
					D_exp['ycf'] = dataset_train['ycf'][:,i_exp-1:i_exp]
				else:
					D_exp['ycf'] = None



				D_exp_test = {}
				if has_test: 
					D_exp_test['x']  = dataset_test['x'][:,:,i_exp-1]
					D_exp_test['t']  = dataset_test['t'][:,i_exp-1:i_exp]
					D_exp_test['yf'] = dataset_test['yf'][:,i_exp-1:i_exp]
					D_exp_test['HAVE_TRUTH'] = dataset_test['HAVE_TRUTH']
					if dataset_test['HAVE_TRUTH']:
						D_exp_test['ycf'] = dataset_test['ycf'][:,i_exp-1:i_exp]
					else:
						D_exp_test['ycf'] = None
				
		I_train, I_valid = validation_split(D_exp, FLAGS.val_part)
		losses, preds_train, preds_test, reps, reps_test = train(the_nn, sess, train_step, D_exp, I_valid, \
																D_exp_test, logfile, i_exp)

		#Collect all reps 
		all_preds_train.append(preds_train)
		all_preds_test.append(preds_test)
		all_losses.append(losses)

		#Fix shape for output (n_units, dim, n_reps, n_outputs) 
		out_preds_train = np.swapaxes(np.swapaxes(all_preds_train,1,3),0,2)
		if  has_test:
			out_preds_test = np.swapaxes(np.swapaxes(all_preds_test,1,3),0,2)
		out_losses = np.swapaxes(np.swapaxes(all_losses,0,2),0,1)

		# Store predictions 
		log(logfile, 'Saving result to %s...\n' % outdir)
		if FLAGS.output_csv:
			np.savetxt('%s_%d.csv' % (outform,i_exp), preds_train[-1], delimiter=',')
			np.savetxt('%s_%d.csv' % (outform_test,i_exp), preds_test[-1], delimiter=',')
			np.savetxt('%s_%d.csv' % (lossform,i_exp), losses, delimiter=',')

		# Compute weights if doing variable selection 
		if FLAGS.varsel:
			if i_exp == 1:
				all_weights = sess.run(the_nn.weights_in[0])
				all_beta = sess.run(the_nn.weights_pred)
			else:
				all_weights = np.dstack((all_weights, sess.run(the_nn.weights_in[0])))
				all_beta = np.dstack((all_beta, sess.run(the_nn.weights_pred)))

		# Save results and predictions 
		all_valid.append(I_valid)
		if FLAGS.varsel:
			np.savez(npzfile, pred=out_preds_train, loss=out_losses, w=all_weights, beta=all_beta, val=np.array(all_valid))
		else:
			np.savez(npzfile, pred=out_preds_train, loss=out_losses, val=np.array(all_valid))

		if has_test:
			np.savez(npzfile_test, pred=out_preds_test)

		# Save representations 
		if FLAGS.save_rep and i_exp == 1:
			np.savez(repfile, rep=reps)

			if has_test:
				np.savez(repfile_test, rep=reps_test)

'''the part that starts the program'''
outdir = './results/'
dataset_file_train = FLAGS.datadir+'ihdp_npci_1-100.train.npz'
dataset_file_test = FLAGS.datadir+'ihdp_npci_1-100.test.npz'
logfile = outdir+'log.txt'
f = open(logfile,'w')
f.close()

percentage = [0.63, 0.27, 0.1]
dataset_train = None 
dataset_test = None 
dataset_vali = None

#Load Data 
npz_input = False
if dataset_file_train[-3:] == 'npz':
	npz_input = True
if npz_input:
	dataset_train = load_data(dataset_file_train)
	dataset_test = load_data(dataset_file_test)

run(outdir, dataset_train, dataset_test, dataset_vali)

