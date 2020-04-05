X = [0.5, 2.5] #input
Y = [0.2, 0.9] # prediction

def sigmoid_function(w, b, x):
  return 1.0/(1.0 + np.exp(-(w * x +b)))

def calculate_error(w, b):
  err = 0.0
  for x, y in zip(X, Y):
    y_predicted =sigmoid_function(w, b, x)
    err += -[(1-y) * math.log(1 - y_predicted, 2) + y * math.log(y_predicted, 2)]
  return err

def gradient_b(w, b, x, y):
  fx = sigmoid_function(w, b, x)
  return (fx - y )

def gradient_w(w, b, x, y):
  fx = sigmoid_function(w, b, x)
  return (fx - y) * x

def calc_gradient_descent():
  w,b,eta = -2, -2, 1.0 #eta is the learning rate
  max_epochs = 1000 # run this code for 1000 iterations
  for i in range (max_epochs):
    derivative_w , derivative_b = 0, 0
    for x, y in zip(X, Y):
      derivative_w += gradient_w(w, b, x, y)
      derivative_b += gradient_b(w, b, x, y)
      
      w = w - eta * derivative_w # new  w  = w - eta * derivative_w
      b = b - eta * derivative_b
    
    
  
