import numpy as np

class ok:
    def __init__(self,a,b,sigma,my):
        self.a = a
        self.b = b
        self.sigma = sigma
        self.my = my

    def f(self):
        "Calculates the integral of f, using the midtpoint method"
        N = 10000 #number of steps
        x_array = np.linspace(self.a,self.b,N) #x-values
        dx = (self.b-self.a)/N
        int_array = np.zeros(len(x_array))
        for i,j in zip(x_array,range(len(int_array))):
            "Calculates the area of each rectangle and adds it to int_array"
            f_start = 1/(np.sqrt(2*np.pi)*self.sigma)*np.exp(-(1/2)*((i-self.my)/self.sigma)**2)
            f_end = 1/(np.sqrt(2*np.pi)*self.sigma)*np.exp(-(1/2)*(((i+dx)-self.my)/self.sigma)**2)
            int_array[j] = ((f_start+f_end)/2)*dx
        return np.sum(int_array) #sums up all rectangles to approach integral

my = 50
sigma = 5
a_1 = my - sigma; b_1 = my + sigma
a_2 = my - 2*sigma; b_2 = my + 2*sigma
a_3 = my - 3*sigma; b_3 = my + 3*sigma

"P(−1σ ≤ x − µ ≤ 1σ)"
h_1 = ok(a_1,b_1,sigma,my)
print(r"P(−1σ ≤ x − µ ≤ 1σ):")
print(h_1.f())

"P(−2σ ≤ x − µ ≤ 2σ)"
h_2 = ok(a_2,b_2,sigma,my)
print(r"P(−2σ ≤ x − µ ≤ 2σ):")
print(h_2.f())

"P(−3σ ≤ x − µ ≤ 3σ)"
h_3 = ok(a_3,b_3,sigma,my)
print("P(−3σ ≤ x − µ ≤ 3σ):")
print(h_3.f())
