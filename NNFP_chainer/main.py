class Main(Chain):
	def __init__(self):
		#self.fp = FP()
		self.dnn = DNN()
	
	def __call__(self):
		return rmse(self.fwd(x), y)
	
	def fwd(x):
		finger_print = fp(x)
		return dnn(finger_print)


model = Main()
optimizer = optimizer.Adam()
optimizer.setup(model)

