from hw_1 import DataSet, PAL
import numpy as np
import matplotlib.pyplot as plt

"""
The LinearRegression class get a dataset (DataSet or TrainingSet class) object for an input and uses
linear regression method to fit linear line that classifies the points into two groups, above and below the line.

class instances:
w: weight vector
w_t: weight vector for nonlinear transformation
dataset: The input and the target function on which the algorithm is executed
y_left: (-1, h(x)), h(x) is the hypothesis function 
y_right: (1, h(x)), h(x) is the hypothesis function
misclassified_points: List of all the points that the algorithm misclassified
classified_points: List of all the points that the algorithm classified correctly
"""


class LinearRegression:
    def __init__(self, dataset):
        self.w = np.array([0.0, 0.0, 0.0])
        self.w_t = []
        self.dataset = dataset
        self.y_left = 0
        self.y_right = 0
        self.misclassified_points = []
        self.classified_points = []

    # Liner regression fit for regular input [1, X1, X2]
    def fit(self):
        self.w = np.dot(np.linalg.pinv(self.dataset.input), self.dataset.output)
        self.y_left = (self.w[1] - self.w[0]) / self.w[2]
        self.y_right = (-self.w[1] - self.w[0]) / self.w[2]

        self.misclassified_points = [i for i in np.arange(0, len(self.dataset.input), 1) if
                                     np.sign(np.dot(self.w, self.dataset.input[i]))
                                     != self.dataset.output[i]]
        self.classified_points = [i for i in np.arange(0, len(self.dataset.input), 1) if
                                  np.sign(np.dot(self.w, self.dataset.input[i]))
                                  == self.dataset.output[i]]

    def plot(self):
        if self.dataset.m != 0 and self.dataset.b != 2:
            plt.fill_between(self.dataset.x, self.dataset.m * self.dataset.x + self.dataset.b, 1,
                             color="lightsteelblue")
            plt.fill_between(self.dataset.x, self.dataset.m * self.dataset.x + self.dataset.b, -1,
                             color="lightcoral")
            plt.plot((-1, 1), (-self.dataset.m + self.dataset.b, self.dataset.m + self.dataset.b),
                     color="black", label="Target function")

        plt.scatter([self.dataset.input[i][1] for i in self.misclassified_points]
                    , [self.dataset.input[i][2] for i in self.misclassified_points], color="black", marker="x")

        for class_point in self.classified_points:
            plt.scatter(self.dataset.input[class_point][1], self.dataset.input[class_point][2],
                        color=self.dataset.color_s[class_point], marker=self.dataset.marker_s[class_point])

        plt.plot((-1.0, 1.0), (self.y_left, self.y_right), color="g", label="hypothesis function")
        plt.gca().set_aspect(1)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.legend()
        plt.show()

    # Return the fraction of misclassified points / total num of points
    def disagreement_probability(self):
        return len(self.misclassified_points) / len(self.dataset.input)

    """ 
        # The disagreement function gets number of points as an input and generate points for checking
        # The disagreement probability between f(x) and g(x)
        # If you want to check the disagreement of nonlinear transformation --> please make sure to change the default
          non_linear to True' when you call this method.
        # You have the option to make some noise to date by change the noise_percentage default to another num [1-100]
        # Return the fraction of misclassified points / total num of points 
    """
    def disagreement(self, num_of_points, non_linear=False, noise_percentage=0):
        points = np.random.uniform(-1, 1, (num_of_points, 3))
        points[:, 0] = 1

        if non_linear:
            non_linear_vector = nonlinear_feature_vector(points)
            tf = self.dataset.target_function_output(f(points), non_linear)

        if noise_percentage != 0:
            i_flip = np.arange(0, len(points))
            np.random.shuffle(i_flip)
            i_flip = i_flip[0:int(len(points) * (noise_percentage / 100))]
            if non_linear:
                tf[i_flip] *= -1

        misclassified_points = 0
        # for p in points:
        for i in range(len(points)):
            if not non_linear:
                if noise_percentage == 0:
                    if np.sign(np.dot(self.w, points[i])) != self.dataset.target_function(points[i]):
                        misclassified_points += 1
                if noise_percentage != 0:
                    if i in i_flip:
                        if np.sign(np.dot(self.w, points[i])) != -1*self.dataset.target_function(points[i]):
                            misclassified_points += 1

            if non_linear:
                if np.sign(np.dot(self.w_t, non_linear_vector[i])) != tf[i]:
                    misclassified_points += 1

        return misclassified_points / num_of_points

    def non_linear_fit(self, non_linear_vector):
        self.w_t = np.dot(np.linalg.pinv(non_linear_vector), self.dataset.output)

"""
The TrainingSet class creates randomly set of points from uniform distribution in range [-1, 1] x [-1, 1]
The class gets the number of points as an input.
N --> num_of_points: Number of points generated

class instances:
x: Sets of points between -1 to 1, for plot
X --> input (2D): Attributes of point in the shape (1,x1,x2) 

I don't want so many changes between DataSet class and TrainingSet class so I keep m' b even though TrainingSet 
doesn't create target function (for plot)
m: The slope of the target function 
b:  The y-intercept of the target function 
y --> output: Vector of the outputs in the shape [1, -1, 1, ...]
color_s: List of the colors of the points. "red" --> -1 , "blue" --> 1
marker_s: List of the markers of the points. "-" --> -1 , "+" --> 1
"""

class TrainingSet:
    def __init__(self, num_of_points):
        self.x = np.arange(-1, 1.01, 0.01)

        # Generate points/ input
        self.input = np.random.uniform(-1, 1, (num_of_points, 3))
        self.input[:, 0] = 1
        self.output = []
        self.m = 0
        self.b = 2
        self.color_s = []
        self.marker_s = []
    """
    If the data cannot be classified linearly, create a target function that gets the input and return a vector 
    of the solutions f(ts.input). this function updates the self.output (starts with []).
    If non_linear = True that mean you use lr.disagreement on non linear transformation. 
    """
    def target_function_output(self, tf, non_linear=False):
        if not non_linear:
            self.output = tf
        else:
            return tf

    # Generate random noise randomly from noise percentage input
    def random_noise(self, noise_percentage):
        i_flip = np.arange(0, len(self.input))
        np.random.shuffle(i_flip)
        i_flip = i_flip[0:int(len(self.input) * (noise_percentage/100))]
        self.output[i_flip] *= -1

    def plot(self):
        # Organize colors and markers for plot
        self.color_s = ["blue" if y > 0 else "red" for y in self.output]
        self.marker_s = ["+" if y > 0 else "_" for y in self.output]

        for i in range(len(self.input)):
            plt.plot(self.input[i][1], self.input[i][2], color=self.color_s[i], marker=self.marker_s[i], markersize=10)

        plt.gca().set_aspect(1)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.show()


def linear_regression_part():
    iterations = 1000
    dis_prob = 0
    dis_prob_fresh = 0
    total_num_of_iterations = 0
    for i in range(iterations):
        print(i)
        ds = DataSet(50)
        # ds.plot()
        lr = LinearRegression(ds)
        lr.fit()
        # lr.plot()
        pal = PAL(ds, lr.w)
        pal.fit()
        # pal.plot()
        total_num_of_iterations += pal.num_of_iter
        dis_prob += lr.disagreement_probability()
        dis_prob_fresh += lr.disagreement(1000)

    print('{}: {}'.format("Disagreement Probability", dis_prob / iterations))
    print('{}: {}'.format("Disagreement Probability 1000 fresh points", dis_prob_fresh / iterations))
    print('{}: {}'.format("Number of iterations", total_num_of_iterations / iterations))


def f(x):
    return np.sign(x[:,1]**2 + x[:,2]**2 - 0.6)

# Returns non linear input vector for the transformation [1, X1, X2, X1*X2, X1^2, X3^2]
def nonlinear_feature_vector(x):
    vector = np.random.uniform(-1, 1, (len(x), 6))
    vector[:, 0] = 1.0
    vector[:, 1] = x[:, 1]
    vector[:, 2] = x[:, 2]
    vector[:, 3] = x[:, 1] * x[:, 2]
    vector[:, 4] = x[:, 1] ** 2
    vector[:, 5] = x[:, 2] ** 2
    return vector


def nonlinear_transformation_part():
    iterations = 1000
    dis_prob = 0
    dis_prob_fresh = 0
    w_0, w_1, w_2, w_3, w_4, w_5 = 0, 0, 0, 0, 0, 0
    for i in range(iterations):
        ts = TrainingSet(1000)
        ts.target_function_output(f(ts.input))
        ts.random_noise(10)  # Flip 10% to simulate noise
        # ts.plot()
        lr = LinearRegression(ts)
        nlfv = nonlinear_feature_vector(ts.input)
        lr.non_linear_fit(nlfv)
        w_0 += lr.w_t[0]
        w_1 += lr.w_t[1]
        w_2 += lr.w_t[2]
        w_3 += lr.w_t[3]
        w_4 += lr.w_t[4]
        w_5 += lr.w_t[5]

        # lr.fit()
        # lr.plot()
        # dis_prob += lr.disagreement_probability()
    # print('{}: {}'.format("Disagreement Probability", dis_prob / iterations))
    print("({}) + ({})X1 +{}X2 + ({})X1*X2 + ({})X1^2 + ({})X2^2".format(w_0/iterations, w_1/iterations, w_2/iterations,
                                                               w_3/iterations, w_4/iterations, w_5/iterations))

    for i in range(iterations):
        print(i)
        dis_prob_fresh += lr.disagreement(1000, non_linear=True, noise_percentage=10)

    print('{}: {}'.format("Disagreement Probability (nl) 1000 fresh points", dis_prob_fresh / iterations))


def main():
    # linear_regression_part()
    nonlinear_transformation_part()

if __name__ == '__main__':
    main()
