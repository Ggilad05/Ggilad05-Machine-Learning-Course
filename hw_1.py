import numpy as np
import random
import matplotlib.pyplot as plt


"""
The DataSet class creates randomly set of points from uniform distribution in range [-1, 1] x [-1, 1]
and create target function which classifies the points into two groups: 1 - above the target function
and -1 - below the target function.
The class gets the number of points as an input.
N --> num_of_points: Number of points generated

class instances:
x: Sets of points between -1 to 1, for plot
X --> input (2D): Attributes of point in the shape (1,x1,x2) 
m: The slope of the target function
b:  The y-intercept of the target function
y --> output: Vector of the outputs in the shape [1, -1, 1, ...]
color_s: List of the colors of the points. "red" --> -1 , "blue" --> 1
marker_s: List of the markers of the points. "-" --> -1 , "+" --> 1
"""


class DataSet:
    def __init__(self, num_of_points):
        self.x = np.arange(-1, 1.01, 0.01)

        # Generate points/ input
        self.input = np.random.uniform(-1, 1, (num_of_points, 3))
        self.input[:, 0] = 1

        # Create f(X)
        p1 = random_point()
        p2 = random_point()
        while p2[0] - p1[0] == 0:
            p2 = random_point()
        self.m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        self.b = p1[1] - self.m * p1[0]

        # Generate output
        self.output = []
        for n in range(num_of_points):
            y = self.target_function(self.input[n])
            self.output.append(y)

        # Organize colors and markers for plot
        self.color_s = ["blue" if y > 0 else "red" for y in self.output]
        self.marker_s = ["+" if y > 0 else "_" for y in self.output]

    """ The target_function gets a x = (1, x1, x2) for an input and return the y/ outputs of her."""
    def target_function(self, x):
        if self.m * x[1] + self.b == x[2]:
            while self.m * x[1] + self.b == x[2]:
                p = random_point()
                x[1] = p[0]
                x[2] = p[1]
        if self.m * x[1] + self.b > x[2]:
            return -1
        else:
            return 1

    def plot(self):
        for i in range(len(self.input)):
            plt.plot(self.input[i][1], self.input[i][2], color=self.color_s[i], marker=self.marker_s[i], markersize=10)

        plt.plot((-1, 1),(-self.m + self.b, self.m + self.b), color="black",label="Target function")
        plt.fill_between(self.x, self.m * self.x + self.b, 1, color="lightsteelblue")
        plt.fill_between(self.x, self.m * self.x + self.b, -1, color="lightcoral")
        plt.gca().set_aspect(1)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.legend()
        plt.show()

"""
The PAL class get a DataSet object for an input and uses "Perceptron Learning Algorithm" to fit linear line 
that classifies the points into two groups, above and below the line.

class instances:
w: weight vector
num_of_iter: number of iterations
dataset: The input and the target function on which the algorithm is executed
y_left: (-1, h(x)), h(x) is the hypothesis function 
y_right: (1, h(x)), h(x) is the hypothesis function
"""
class PAL:
    def __init__(self, dataset):
        self.w = np.array([0.0, 0.0, 0.0])
        self.num_of_iter = 0
        self.dataset = dataset
        self.y_left = 0
        self.y_right = 0

    def fit(self, plot_iter=False):

        check = True
        while check:
            misclassified_points = [i for i in np.arange(0, len(self.dataset.input), 1) if np.sign(np.dot(self.w, self.dataset.input[i]))
                                    != self.dataset.output[i]]
            classified_points = [i for i in np.arange(0, 10, 1) if np.sign(np.dot(self.w, self.dataset.input[i]))
                                    == self.dataset.output[i]]
            if not misclassified_points:
                check = False
            else:
                if plot_iter:
                    self.plot_iterates(misclassified_points, classified_points)

                misclassified_point = np.random.choice(misclassified_points)
                self.w += self.dataset.output[misclassified_point] * self.dataset.input[misclassified_point]
                self.y_left = (self.w[1] - self.w[0]) / self.w[2]
                self.y_right = (-self.w[1] - self.w[0]) / self.w[2]

                self.num_of_iter += 1

    def plot(self):
        plt.plot((-1.0, 1.0), (self.y_left, self.y_right), color="g", label="hypothesis function")
        self.dataset.plot()

    def plot_iterates(self, misclassified_points, classified_points):
        plt.fill_between(self.dataset.x, self.dataset.m * self.dataset.x + self.dataset.b, 1,
                         color="lightsteelblue")
        plt.fill_between(self.dataset.x, self.dataset.m * self.dataset.x + self.dataset.b, -1,
                         color="lightcoral")

        plt.scatter([self.dataset.input[i][1] for i in misclassified_points]
                    , [self.dataset.input[i][2] for i in misclassified_points], color="black", marker="x")

        for class_point in classified_points:
            plt.scatter(self.dataset.input[class_point][1], self.dataset.input[class_point][2],
                        color=self.dataset.color_s[class_point], marker=self.dataset.marker_s[class_point])

        plt.plot((-1, 1), (-self.dataset.m + self.dataset.b, self.dataset.m + self.dataset.b),
                 color="black", label="Target function")
        plt.plot((-1.0, 1.0), (self.y_left, self.y_right), color="g", label="hypothesis function")
        plt.gca().set_aspect(1)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.legend()
        plt.show()
        plt.close()

    # The disagreement function gets number of points as an input and generate points for checking
    # the disagreement probability between f(x) and g(x)
    def disagreement(self, num_of_points):
        points = np.random.uniform(-1, 1, (num_of_points, 3))
        points[:, 0] = 1
        misclassified_points = 0
        for p in points:
            if np.sign(np.dot(self.w, p)) != self.dataset.target_function(p):
                misclassified_points += 1
        return misclassified_points / num_of_points


def random_point():
    x, y = random.uniform(-1, 1), random.uniform(-1, 1)
    return (x, y)


def main():
    total_num_of_iterations = 0
    iterations = 1000
    dis_prob = 0
    for i in range(iterations):
        ds = DataSet(100)
        pal = PAL(ds)
        pal.fit()
        dis_prob += pal.disagreement(100)
        # print('{}: {}'.format("Number of iterations", pal.num_of_iter))
        # print(" ")
        total_num_of_iterations += pal.num_of_iter

    print('{}: {}'.format("Number of iterations", total_num_of_iterations / iterations))
    print(" ")
    print('{}: {}'.format("Probability", dis_prob / iterations))


if __name__ == '__main__':
    main()