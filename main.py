# to open toolbar, press cmd+','
# tixz diagram

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
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


class RipsComplexes:
    def __init__(self, data_points: pd.DataFrame, time: float, graph_results=True, p=np.inf):
        self.data_pts = data_points[['x coord', 'y coord']].values
        self.time, self.p = time, p
        self.complex = self.generate_rips_complex()
        if graph_results:
            self.graph_complex()
            plt.show()

    def generate_rips_complex(self):
        distance_matrix = scipy.spatial.distance_matrix(self.data_pts, self.data_pts, p=self.p)
        rips_complex = gudhi.RipsComplex(distance_matrix=distance_matrix, max_edge_length=2*self.time)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        simplex_tree.compute_persistence()
        return simplex_tree.get_filtration()

    def point_cloud_type(self, i, x_coord, y_coord):
        COLORS, OPACITY = ['r', 'g', 'b', 'm'], 0.25
        if self.p == np.inf:
            return RegularPolygon((x_coord, y_coord), numVertices=4, orientation=math.pi/4, radius=self.time,
                                  color=COLORS[i % len(COLORS)], alpha=OPACITY)
        return plt.Circle((x_coord, y_coord), radius=self.time, color=COLORS[i % len(COLORS)], alpha=OPACITY)

    def plot_data_points(self, ax):
        for i, (x_coord, y_coord) in enumerate(self.data_pts):
            plt.scatter(x_coord, y_coord, color='black', marker='o', s=10)
            ball = self.point_cloud_type(i, x_coord, y_coord)
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


def main():
    sample_data = SampleData().data
    TIME = 0.15
    rips_complex = RipsComplexes(sample_data, TIME, graph_results=True, p=np.inf)


if __name__ == '__main__':
    main()

