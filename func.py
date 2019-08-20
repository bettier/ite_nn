import tensorflow as tf
import numpy as np

SQRT_CONST = 1e-10
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('outdir', '../results/result.npz', """Output directory. """)
tf.app.flags.DEFINE_integer('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
tf.app.flags.DEFINE_float('weight_init', 0.1, """Weight initialization scale. """)
tf.app.flags.DEFINE_string('save_datadir', '../data/', """Data directory. """)
tf.app.flags.DEFINE_integer('seed', 1, """Seed. """)
tf.app.flags.DEFINE_string('datadir', './data/', """Data directory. """)
tf.app.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
tf.app.flags.DEFINE_integer('dim_rep', 200, """Pre-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_reg', 200, """Post-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('n_in', 3, """Number of representation layers. """)
tf.app.flags.DEFINE_integer('n_out', 3, """Number of regression layers. """)
tf.app.flags.DEFINE_string('loss', 'l2', """Which loss function to use (l1/l2/log)""")
tf.app.flags.DEFINE_float('dropout_in', 1, """Input layers dropout keep rate. """)
tf.app.flags.DEFINE_float('dropout_out', 1, """Output layers dropout keep rate. """)
tf.app.flags.DEFINE_string('nonlin', 'elu', """Kind of non-linearity. Default relu. """)

tf.app.flags.DEFINE_float('p_alpha', 1, """Imbalance regularization param. """)
tf.app.flags.DEFINE_float('p_lambda', 1e-4, """Weight decay regularization parameter. """)
tf.app.flags.DEFINE_integer('rep_weight_decay', 0, """Whether to penalize representation layers with weight decay""")
tf.app

tf.app.flags.DEFINE_float('lrate', 1e-3, """Learning rate. """)
tf.app.flags.DEFINE_float('decay', 0.3, """RMSProp decay. """)
tf.app.flags.DEFINE_integer('batch_size', 100, """Batch size. """)

tf.app.flags.DEFINE_integer('batch_norm', 0, """Whether to use batch normalization. """)
tf.app.flags.DEFINE_string('normalization', 'none', """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
tf.app.flags.DEFINE_float('rbf_sigma', 0.1, """RBF MMD sigma """)

tf.app.flags.DEFINE_integer('experiments', 100, """Number of experiments. """)

tf.app.flags.DEFINE_integer('iterations', 3000, """Number of iterations. """)

tf.app.flags.DEFINE_float('lrate_decay', 0.97, """Decay of learning rate every 100 iterations """)
tf.app.flags.DEFINE_integer('wass_iterations', 10, """Number of iterations in Wasserstein computation. """)
tf.app.flags.DEFINE_float('wass_lambda', 10.0, """Wasserstein lambda. """)
tf.app.flags.DEFINE_integer('wass_bpt', 1, """Backprop through T matrix? """)


tf.app.flags.DEFINE_string('data_test', '', """Test data filename form. """)


tf.app.flags.DEFINE_integer('use_p_correction', 0, """Whether to use population size p(t) in mmd/disc/wass.""")
tf.app.flags.DEFINE_string('optimizer', 'Adam', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
tf.app.flags.DEFINE_string('imb_fun', 'wass', """Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
tf.app.flags.DEFINE_integer('output_csv',0,"""Whether to save a CSV file with the results""")
tf.app.flags.DEFINE_integer('output_delay', 100, """Number of iterations between log/loss outputs. """)
tf.app.flags.DEFINE_integer('pred_output_delay', 200, """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
tf.app.flags.DEFINE_integer('debug', 0, """Debug mode. """)
tf.app.flags.DEFINE_integer('save_rep', 0, """Save representations after training. """)
tf.app.flags.DEFINE_float('val_part', 0.3, """Validation part. """)
tf.app.flags.DEFINE_boolean('split_output', 1, """Whether to split output layers between treated and control. """)
tf.app.flags.DEFINE_boolean('reweight_sample', 1, """Whether to reweight sample for prediction loss with average treatment probability. """)
def cf_nn(x, t):
    It = np.array(np.where(t==1))[0,:]
    Ic = np.array(np.where(t==0))[0,:]

    x_c = x[Ic,:]
    x_t = x[It,:]

    D = pdist2(x_c, x_t)

    nn_t = Ic[np.argmin(D,0)]
    nn_c = It[np.argmin(D,1)]

    return nn_t, nn_c

def pehe_nn(yf_p, ycf_p, y, x, t, nn_t=None, nn_c=None):
    if nn_t is None or nn_c is None:
        nn_t, nn_c = cf_nn(x,t)

    It = np.array(np.where(t==1))[0,:]
    Ic = np.array(np.where(t==0))[0,:]

    ycf_t = 1.0*y[nn_t]
    eff_nn_t = ycf_t - 1.0*y[It]
    eff_pred_t = ycf_p[It] - yf_p[It]

    eff_pred = eff_pred_t
    eff_nn = eff_nn_t
    pehe_nn = np.sqrt(np.mean(np.square(eff_pred - eff_nn)))
    return pehe_nn

def validation_split(D_exp, val_fraction):
	n = D_exp['x'].shape[0]
	if val_fraction > 0:
		n_valid = int(val_fraction*n)
		n_train = n-n_valid
		I = np.random.permutation(range(0,n))
		I_train = I[:n_train]
		I_valid = I[n_train:]
	else:
		I_train = range(n)
		I_valid = []

	return I_train, I_valid

def log(logfile,str):
	with open(logfile,'a') as f:
		f.write(str+'\n')
	print(str)

def save_config(fname):
	flagdict =  FLAGS.__dict__['__flags']
	s = '\n'.join(['%s: %s' % (k,str(flagdict[k])) for k in sorted(flagdict.keys())])
	f = open(fname,'w')
	f.write(s)
	f.close()

def load_data(fname):
	if fname[-3:] == 'npz':
		data_in = np.load(fname)
		data = {'x': data_in['x'], 't': data_in['t'], 'yf': data_in['yf']}
		try:
			data['ycf'] = data_in['ycf']
		except:
			data['ycf'] = None
	else:
		if FLAGS.sparse>0:
			data_in = np.loadtxt(open(fname+'.y',"rb"),delimiter=",")
			x = load_sparse(fname+'.x')
		else:
			data_in = np.loadtxt(open(fname,"rb"),delimiter=",")
			x = data_in[:,5:]

		data['x'] = x
		data['t'] = data_in[:,0:1]
		data['yf'] = data_in[:,1:2]
		data['ycf'] = data_in[:,2:3]

	data['HAVE_TRUTH'] = not data['ycf'] is None

	data['dim'] = data['x'].shape[1]
	data['n'] = data['x'].shape[0]
	return data

def load_sparse(fname):
	E = np.loadtxt(open(fname,"rb"),delimiter=",")
	H = E[0,:]
	n = int(H[0])
	d = int(H[1])
	E = E[1:,:]
	S = sparse.coo_matrix((E[:,2],(E[:,0]-1,E[:,1]-1)),shape=(n,d))
	S = S.todense()
	return S

def safe_sqrt(x, lbound=SQRT_CONST):
	return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))

#linear MMD 
def lindisc(X,p,t):

	it = tf.where(t>0)[:,0]
	ic = tf.where(t<1)[:,0]
	Xc = tf.gather(X,ic)
	Xt = tf.gather(X,it)

	mean_control = tf.reduce_mean(Xc,reduction_indices=0)
	mean_treated = tf.reduce_mean(Xt,reduction_indices=0)

	c = tf.square(2*p-1)*0.25
	f = tf.sign(p-0.5)

	mmd = tf.reduce_sum(tf.square(p*mean_treated - (1-p)*mean_control))
	mmd = f*(p-0.5) + safe_sqrt(c + mmd)

	return mmd

#linear MMD 
def mmd2_lin(X,t,p):

	it = tf.where(t>0)[:,0]
	ic = tf.where(t<1)[:,0]

	Xc = tf.gather(X,ic)
	Xt = tf.gather(X,it)

	mean_control = tf.reduce_mean(Xc,reduction_indices=0)
	mean_treated = tf.reduce_mean(Xt,reduction_indices=0)

	mmd = tf.reduce_sum(tf.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control))

	return mmd

#12-RBF MMD 
def mmd2_rbf(X,t,p,sig):
	it = tf.where(t>0)[:,0]
	ic = tf.where(t<1)[:,0]

	Xc = tf.gather(X,ic)
	Xt = tf.gather(X,it)

	Kcc = tf.exp(-pdist2sq(Xc,Xc)/tf.square(sig))
	Kct = tf.exp(-pdist2sq(Xc,Xt)/tf.square(sig))
	Ktt = tf.exp(-pdist2sq(Xt,Xt)/tf.square(sig))

	m = tf.to_float(tf.shape(Xc)[0])
	n = tf.to_float(tf.shape(Xt)[0])

	mmd = tf.square(1.0-p)/(m*(m-1.0))*(tf.reduce_sum(Kcc)-m)
	mmd = mmd + tf.square(p)/(n*(n-1.0))*(tf.reduce_sum(Ktt)-n)
	mmd = mmd - 2.0*p*(1.0-p)/(m*n)*tf.reduce_sum(Kct)
	mmd = 4.0*mmd

	return mmd

# the squared Euclidean distance between all pairs x in X, y in Y
def pdist2sq(X,Y):
	C = -2*tf.matmul(X,tf.transpose(Y))
	nx = tf.reduce_sum(tf.square(X),1,keep_dims=True)
	ny = tf.reduce_sum(tf.square(Y),1,keep_dims=True)
	D = (C + tf.transpose(ny)) + nx
	return D

def pdist2(X,Y):
	return safe_sqrt(pdist2sq(X,Y))

#distance matrix 
def pop_dist(X,t):
	it = tf.where(t>0)[:,0]
	ic = tf.where(t<1)[:,0]
	Xc = tf.gather(X,ic)
	Xt = tf.gather(X,it)
	nc = tf.to_float(tf.shape(Xc)[0])
	nt = tf.to_float(tf.shape(Xt)[0])
	M = pdist2(Xt,Xc)
	return M

# the Wasserstein distance between treatment groups
def wasserstein(X,t,p,lam=10,its=10,sq=False,backpropT=False):

	it = tf.where(t>0)[:,0]
	ic = tf.where(t<1)[:,0]
	Xc = tf.gather(X,ic)
	Xt = tf.gather(X,it)
	nc = tf.to_float(tf.shape(Xc)[0])
	nt = tf.to_float(tf.shape(Xt)[0])

	#distance matrix 
	if sq:
		M = pdist2sq(Xt,Xc)
	else:
		M = safe_sqrt(pdist2sq(Xt,Xc))
	#lamda and delta 
	M_mean = tf.reduce_mean(M)
	M_drop = tf.nn.dropout(M,10/(nc*nt))
	delta = tf.stop_gradient(tf.reduce_max(M))
	eff_lam = tf.stop_gradient(lam/M_mean)
	#new distance matrix 
	Mt = M
	row = delta*tf.ones(tf.shape(M[0:1,:]))
	col = tf.concat(0,[delta*tf.ones(tf.shape(M[:,0:1])),tf.zeros((1,1))])
	Mt = tf.concat(0,[M,row])
	Mt = tf.concat(1,[Mt,col])
	#marginal vector 
	a = tf.concat(0,[p*tf.ones(tf.shape(tf.where(t>0)[:,0:1]))/nt, (1-p)*tf.ones((1,1))])
	b = tf.concat(0,[(1-p)*tf.ones(tf.shape(tf.where(t<1)[:,0:1]))/nc, p*tf.ones((1,1))])
	#kernel matrix'''
	Mlam = eff_lam*Mt
	K = tf.exp(-Mlam) + 1e-6 
	U = K*Mt
	ainvK = K/a
	u = a
	for i in range(0,its):
		u = 1.0/(tf.matmul(ainvK,(b/tf.transpose(tf.matmul(tf.transpose(u),K)))))
	v = b/(tf.transpose(tf.matmul(tf.transpose(u),K)))
	T = u*(tf.transpose(v)*K)

	if not backpropT:
		T = tf.stop_gradient(T)

	E = T*Mt
	D = 2*tf.reduce_sum(E)

	return D, Mlam
#Projects a vector x onto the k-simplex 
def simplex_project(x,k):
	d = x.shape[0]
	mu = np.sort(x,axis=0)[::-1]
	nu = (np.cumsum(mu)-k)/range(1,d+1)
	I = [i for i in range(0,d) if mu[i]>nu[i]]
	theta = nu[I[-1]]
	w = np.maximum(x-theta,0)
	return w
