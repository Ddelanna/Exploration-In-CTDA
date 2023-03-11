# to open toolbar, press cmd+','
# tixz diagram
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import gudhi
import scipy
from matplotlib import collections as mc
import itertools as iter


class SampleData:  # TODO: create more sample shapes
    """ Generate 2D sample shaped data with noise. """
    def __init__(self, shape="circle", num_data_pts=100):
        self.data = pd.DataFrame()
        self.num_data_pts = num_data_pts
        self.make_sample_data(shape)

    def make_sample_data(self, shape):
        self.data['x noise'], self.data['y noise'] = self.generate_noise(), self.generate_noise()
        shape_dict = {
            'circle': self.generate_unit_circle
        }
        try:
            shape_dict[shape](self.data)
        except KeyError:
            print('KeyError: Shape keyword input not found.')

    def generate_noise(self):
        noise_scaling_factor = .50
        return np.random.rand(self.num_data_pts) * noise_scaling_factor

    def generate_unit_circle(self, data):
        radian_scaling_factor = 2 * math.pi
        data['radian'] = np.random.rand(self.num_data_pts) * radian_scaling_factor
        data['x coord'] = np.cos(data['radian']) + data['x noise']
        data['y coord'] = np.sin(data['radian']) + data['y noise']


class CechComplexes:
    def __init__(self, data_points: pd.DataFrame, time: float):
        self.data_pts = data_points[['x coord', 'y coord']].values
        self.time = time
        self.complex = self.generate_cech_complex()

    def generate_cech_complex(self):
        cech_complex = []
        DIAMETER = 2 * self.time
        distance_matrix = scipy.spatial.distance_matrix(self.data_pts, self.data_pts, p=2)  # TODO: p=np.inf
        for (i, j) in iter.combinations(range(len(self.data_pts)), 2):  # all lower-triangular indices
            if distance_matrix[i][j] <= DIAMETER:
                cech_complex.append([self.data_pts[i], self.data_pts[j]])
        return cech_complex

    def graph_complex(self):
        COLORS = ['r', 'g', 'b', 'm']
        lc = mc.LineCollection(self.complex, linewidths=1, color='black')
        fig, ax = plt.subplots()
        ax.set_aspect(1)
        for i, (x_coord, y_coord) in enumerate(self.data_pts[['x coord', 'y coord']].values):
            plt.scatter(x_coord, y_coord, color='black', marker='o', s=10)
            ball = plt.Circle((x_coord, y_coord), radius=self.time, color=COLORS[i % len(COLORS)], alpha=0.2)
            ax.add_artist(ball)
            ax.add_collection(lc)
        plt.show()


class RipsComplexes:
    def __init__(self, data_points: pd.DataFrame, time: float):
        self.data_pts = data_points[['x coord', 'y coord']].values
        self.time = time
        self.complex = self.generate_rips_complex()

    def generate_rips_complex(self):
        rips_complex = gudhi.RipsComplex(points=self.data_pts, max_edge_length=2*self.time)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        simplex_tree.compute_persistence()
        return simplex_tree.get_filtration()

    def plot_data_points(self, ax):
        COLORS, OPACITY = ['r', 'g', 'b', 'm'], 0.25
        for i, (x_coord, y_coord) in enumerate(self.data_pts):
            plt.scatter(x_coord, y_coord, color='black', marker='o', s=10)
            ball = plt.Circle((x_coord, y_coord), radius=self.time, color=COLORS[i % len(COLORS)], alpha=OPACITY)
            ax.add_artist(ball)
        # new_data = SampleData(num_data_pts=20).data[['x coord', 'y coord']].values
        # for i, (x_coord, y_coord) in enumerate(new_data):
        #     plt.scatter(x_coord, y_coord, color='red', marker='o', s=10)

    def plot_simplices(self):
        OPACITY = 0.20
        for (simplex_idx, _) in self.complex:
            simplex_coords = [self.data_pts[idx] for idx in simplex_idx]
            simplex = plt.Polygon(simplex_coords, color='black', alpha=OPACITY, linewidth=2)
            plt.gca().add_patch(simplex)

    def graph_complex(self):
        fig, ax = plt.subplots()
        ax.set_aspect(1)
        ax.autoscale(enable=True)
        self.plot_data_points(ax)
        self.plot_simplices()
        plt.show()

class RipsComplexes:
    def __init__(self, data_points: pd.DataFrame, time: float):
        self.data_pts = data_points[['x coord', 'y coord']].values
        self.time = time
        self.complex = self.generate_rips_complex()

    def generate_rips_complex(self):
        rips_complex = gudhi.RipsComplex(points=self.data_pts, max_edge_length=2*self.time)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        simplex_tree.compute_persistence()
        return simplex_tree.get_filtration()

    def plot_data_points(self, ax):
        COLORS, OPACITY = ['r', 'g', 'b', 'm'], 0.25
        for i, (x_coord, y_coord) in enumerate(self.data_pts):
            plt.scatter(x_coord, y_coord, color='black', marker='o', s=10)
            ball = plt.Circle((x_coord, y_coord), radius=self.time, color=COLORS[i % len(COLORS)], alpha=OPACITY)
            ax.add_artist(ball)
        # new_data = SampleData(num_data_pts=20).data[['x coord', 'y coord']].values
        # for i, (x_coord, y_coord) in enumerate(new_data):
        #     plt.scatter(x_coord, y_coord, color='red', marker='o', s=10)

    def plot_simplices(self):
        OPACITY = 0.20
        for (simplex_idx, _) in self.complex:
            simplex_coords = [self.data_pts[idx] for idx in simplex_idx]
            simplex = plt.Polygon(simplex_coords, color='black', alpha=OPACITY, linewidth=2)
            plt.gca().add_patch(simplex)

    def graph_complex(self):
        fig, ax = plt.subplots()
        ax.set_aspect(1)
        ax.autoscale(enable=True)
        self.plot_data_points(ax)
        self.plot_simplices()
        plt.show()


class Complexes:
    def __init__(self, data_points: pd.DataFrame, time: float, complex_type='Rips', graph_results=False):
        self.complex_type = self.get_complex_type(data_points, time, complex_type)
        self.complex = self.complex_type.complex
        if graph_results:
            self.graph_complex()

    @staticmethod
    def get_complex_type(data_points, time, complex_type):
        complex_type_dict = {
            'Rips': RipsComplexes,
            'Cech': CechComplexes
        }
        try:
            return complex_type_dict[complex_type](data_points, time)
        except KeyError:
            print('KeyError: Shape keyword input not found.')

    def graph_complex(self):
        self.complex_type.graph_complex()


def main():
    sample_data = SampleData().data
    TIME = 0.15
    rips_complex = Complexes(sample_data, TIME, complex_type='Rips', graph_results=True)
    new_data = SampleData(num_data_pts=20).data


if __name__ == '__main__':
    main()

