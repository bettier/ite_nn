import tensorflow as tf
import numpy as np
from func import *

class nn(object):

	def __init__(self, x, t, y_ , p_t, FLAGS, r_alpha, r_lambda, drop_in, drop_out, dims):
		self.variables = {}
		self.wd_loss = 0

		if FLAGS.nonlin.lower() == 'elu':
			self.nonlin = tf.nn.elu
		else:
			self.nonlin = tf.nn.relu

		self._build_graph(x, t, y_ , p_t, FLAGS, r_alpha, r_lambda, drop_in, drop_out, dims)

	def _add_variable(self, var, name):

		basename = name
		i = 0
		while name in self.variables:
			name = '%s_%d' % (basename, i) 
			i += 1

		self.variables[name] = var

	def _create_variable(self, var, name):

		var = tf.Variable(var, name=name)
		self._add_variable(var, name)
		return var

	def _create_variable_with_weight_decay(self, initializer, name, wd):
		var = self._create_variable(initializer, name)
		self.wd_loss += wd*tf.nn.l2_loss(var)
		return var

	def _build_graph(self, x, t, y_ , p_t, FLAGS, r_alpha, r_lambda, drop_in, drop_out, dims):

		self.x = x
		self.t = t
		self.y_ = y_
		self.p_t = p_t
		self.r_alpha = r_alpha
		self.r_lambda = r_lambda
		self.drop_in = drop_in
		self.drop_out = drop_out

		dim_input = dims[0]
		dim_rep = dims[1]
		dim_reg = dims[2]

		weights_in = []
		biases_in = []

		if FLAGS.batch_norm:
			bn_biases = []
			bn_scales = []
		h_in = [x]
		for i in range(0, FLAGS.n_in):
			if i==0:

				if FLAGS.varsel:
					weights_in.append(tf.Variable(1.0/dim_input*tf.ones([dim_input])))
				else:
					weights_in.append(tf.Variable(tf.random_normal([dim_input, dim_rep], stddev=FLAGS.weight_init/np.sqrt(dim_input))))
			else:
				weights_in.append(tf.Variable(tf.random_normal([dim_rep,dim_rep], stddev=FLAGS.weight_init/np.sqrt(dim_rep))))

			if FLAGS.varsel and i==0:
				biases_in.append([])
				h_in.append(tf.mul(h_in[i],weights_in[i]))
			else:
				biases_in.append(tf.Variable(tf.zeros([1,dim_rep])))
				z = tf.matmul(h_in[i], weights_in[i]) + biases_in[i]

				if FLAGS.batch_norm:
					batch_mean, batch_var = tf.nn.moments(z, [0])

					if FLAGS.normalization == 'bn_fixed':
						z = tf.nn.batch_normalization(z, batch_mean, batch_var, 0, 1, 1e-3)
					else:
						bn_biases.append(tf.Variable(tf.zeros([dim_rep])))
						bn_scales.append(tf.Variable(tf.ones([dim_rep])))
						z = tf.nn.batch_normalization(z, batch_mean, batch_var, bn_biases[-1], bn_scales[-1], 1e-3)

				h_in.append(self.nonlin(z))
				h_in[i+1] = tf.nn.dropout(h_in[i+1], drop_in)

		h_rep = h_in[len(h_in)-1]


		if FLAGS.normalization == 'divide':
			h_rep_norm = h_rep / safe_sqrt(tf.reduce_sum(tf.square(h_rep), axis=1, keep_dims=True))
		else:
			h_rep_norm = 1.0*h_rep
		y, weights_out, weights_pred = self._build_output_graph(h_rep_norm, t, dim_rep, dim_reg, drop_out, FLAGS) 
		if FLAGS.reweight_sample:
			w_t = t/(2*p_t)
			w_c = (1-t)/(2*(1-p_t))
			sample_weight = w_t + w_c
		else:
			sample_weight = 1.0

		self.sample_weight = sample_weight

		if FLAGS.loss == 'l1':
			risk = tf.reduce_mean(sample_weight*tf.abs(y_-y))
			pred_error = -tf.reduce_mean(res)
		elif FLAGS.loss == 'log':
			y = 0.995/(1.0+tf.exp(-y)) + 0.0025
			res = y_*tf.log(y) + (1.0-y_)*tf.log(1.0-y)

			risk = -tf.reduce_mean(sample_weight*res)
			pred_error = -tf.reduce_mean(res)
		else:
			risk = tf.reduce_mean(sample_weight*tf.square(y_ - y))
			pred_error = tf.sqrt(tf.reduce_mean(tf.square(y_ - y)))

		if FLAGS.p_lambda>0 and FLAGS.rep_weight_decay:
			for i in range(0, FLAGS.n_in):
				if not (FLAGS.varsel and i==0): # No penalty on W in variable selection
					self.wd_loss += tf.nn.l2_loss(weights_in[i])

		if FLAGS.use_p_correction:
			p_ipm = self.p_t
		else:
			p_ipm = 0.5

		if FLAGS.imb_fun == 'mmd2_rbf':
			imb_dist = mmd2_rbf(h_rep_norm,t,p_ipm,FLAGS.rbf_sigma)
			imb_error = r_alpha*imb_dist
		elif FLAGS.imb_fun == 'mmd2_lin':
			imb_dist = mmd2_lin(h_rep_norm,t,p_ipm)
			imb_error = r_alpha*mmd2_lin(h_rep_norm,t,p_ipm)
		elif FLAGS.imb_fun == 'mmd_rbf':
			imb_dist = tf.abs(mmd2_rbf(h_rep_norm,t,p_ipm,FLAGS.rbf_sigma))
			imb_error = safe_sqrt(tf.square(r_alpha)*imb_dist)
		elif FLAGS.imb_fun == 'mmd_lin':
			imb_dist = mmd2_lin(h_rep_norm,t,p_ipm)
			imb_error = safe_sqrt(tf.square(r_alpha)*imb_dist)
		elif FLAGS.imb_fun == 'wass':
			imb_dist, imb_mat = wasserstein(h_rep_norm,t,p_ipm,lam=FLAGS.wass_lambda,its=FLAGS.wass_iterations,sq=False,backpropT=FLAGS.wass_bpt)
			imb_error = r_alpha * imb_dist
			self.imb_mat = imb_mat 
		elif FLAGS.imb_fun == 'wass2':
			imb_dist, imb_mat = wasserstein(h_rep_norm,t,p_ipm,lam=FLAGS.wass_lambda,its=FLAGS.wass_iterations,sq=True,backpropT=FLAGS.wass_bpt)
			imb_error = r_alpha * imb_dist
			self.imb_mat = imb_mat 
		else:
			imb_dist = lindisc(h_rep_norm,p_ipm,t)
			imb_error = r_alpha * imb_dist

		tot_error = risk

		if FLAGS.p_alpha>0:
			tot_error = tot_error + imb_error

		if FLAGS.p_lambda>0:
			tot_error = tot_error + r_lambda*self.wd_loss 


		if FLAGS.varsel:
			self.w_proj = tf.placeholder("float", shape=[dim_input], name='w_proj')
			self.projection = weights_in[0].assign(self.w_proj)

		self.output = y
		self.tot_loss = tot_error
		self.imb_loss = imb_error
		self.imb_dist = imb_dist
		self.pred_loss = pred_error
		self.weights_in = weights_in
		self.weights_out = weights_out
		self.weights_pred = weights_pred
		self.h_rep = h_rep
		self.h_rep_norm = h_rep_norm
		print("Finished Constructing the NN")

	def _build_output(self, h_input, dim_rep, dim_reg, drop_out, FLAGS):
		h_out = [h_input]
		dims = [dim_rep] + ([dim_reg]*FLAGS.n_out)

		weights_out = []
		biases_out = []

		for i in range(0, FLAGS.n_out):

			wo = self._create_variable_with_weight_decay( tf.random_normal([dims[i], dims[i+1]], \
						stddev=FLAGS.weight_init/np.sqrt(dims[i])), 'w_out_%d' % i, 1.0)
			
			weights_out.append(wo)
			biases_out.append(tf.Variable(tf.zeros([1,dim_reg])))
			z = tf.matmul(h_out[i], weights_out[i]) + biases_out[i]

			h_out.append(self.nonlin(z))
			h_out[i+1] = tf.nn.dropout(h_out[i+1], drop_out)

		
		weights_pred = self._create_variable(tf.random_normal([dim_reg,1], \
			stddev=FLAGS.weight_init/np.sqrt(dim_reg)), 'w_pred')
		bias_pred = self._create_variable(tf.zeros([1]), 'b_pred')

		if FLAGS.varsel or FLAGS.n_out == 0:
			self.wd_loss += tf.nn.l2_loss(tf.slice(weights_pred,[0,0],[dim_out-1,1])) #don't penalize treatment coefficient
		else:
			self.wd_loss += tf.nn.l2_loss(weights_pred)
		h_pred = h_out[-1]
		y = tf.matmul(h_pred, weights_pred)+bias_pred 
		return y, weights_out, weights_pred

	def _build_output_graph(self, rep, t, dim_rep, dim_reg, drop_out, FLAGS):
		if FLAGS.split_output:
			i0 = tf.to_int32(tf.where(t < 1)[:,0])
			i1 = tf.to_int32(tf.where(t > 0)[:,0])

			rep0 = tf.gather(rep, i0)
			rep1 = tf.gather(rep, i1)

			y0, weights_out0, weights_pred0 = self._build_output(rep0, dim_rep, dim_reg, drop_out, FLAGS)
			y1, weights_out1, weights_pred1 = self._build_output(rep1, dim_rep, dim_reg, drop_out, FLAGS)

			y = tf.dynamic_stitch([i0, i1], [y0, y1])
			weights_out = weights_out0 + weights_out1
			weights_pred = weights_pred0 + weights_pred1
		else:
			h_input = tf.concat(1,[rep, t])
			y, weights_out, weights_pred = self._build_output(h_input, dim_rep+1, dim_reg, drop_out, FLAGS)

		return y, weights_out, weights_pred
