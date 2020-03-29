X = [0.5, 2.5] #input
Y = [0.2, 0.9] # prediction

def sigmoid_function(w, b, x):
  return 1.0/(1.0 + np.exp(-(w * x +b)))

def calculate_error(w, b):
  err = 0.0
  for x, y in zip(X, Y):
    y_predicted =sigmoid_function(w, b, x)
    err += 0.5 * (y - y_predicted) ** 2 # Root Mean Square Error = 1/2 (y-predictedy)squared
  return err

def gradient_b(w, b, x, y):
  fx = sigmoid_function(w, b, x)
  return (fx - y ) * fx * (1 - fx)

def gradient_w(w, b, x, y):
  fx = sigmoid_function(w, b, x)
  return (fx - y) * fx * (1 - fx ) * x

def calc_gradient_descent():
  w,b,eta = -2, -2, 1.0 #eta is the learning rate
  max_epochs = 1000 # run this code for 1000 iterations
  for i in range (max_epochs):
    derivative_w , derivative_b = 0, 0
    for x, y in zip(X, Y):
      derivative_w += gradient_w(w, b, x, y)
      derivative_b += gradient_b(w, b, x, y)
      
      w = w - eta * derivative_w
      b = b - eta * derivative_b
    
    
  
