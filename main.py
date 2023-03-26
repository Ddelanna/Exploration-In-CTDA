# to open toolbar, press cmd+','
# tixz diagram

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import math
import gudhi
import scipy


class SampleData:  # TODO: create more sample shapes
    """ Generate 2D sample data modeled after given shape with noise. """
    def __init__(self, shape="circle", num_data_pts=100, p=np.inf):
        self.num_data_pts = num_data_pts
        dataframe = self.__make_sample_data(shape)
        self.data = dataframe[['x coord', 'y coord']]
        self.distance_mat = scipy.spatial.distance_matrix(self.data, self.data, p=p)

    def __generate_noise(self):
        NOISE_SCALING_FACTOR = .50
        return np.random.rand(self.num_data_pts) * NOISE_SCALING_FACTOR

    def __generate_unit_circle(self, dataframe):
        radian_scaling_factor = 2 * math.pi
        dataframe['radian'] = np.random.rand(self.num_data_pts) * radian_scaling_factor
        dataframe['x coord'] = np.cos(dataframe['radian']) + dataframe['x noise']
        dataframe['y coord'] = np.sin(dataframe['radian']) + dataframe['y noise']
        return dataframe

    def __make_sample_data(self, shape):
        dataframe = pd.DataFrame()
        dataframe['x noise'], dataframe['y noise'] = self.__generate_noise(), self.__generate_noise()
        shape_dict = {
            'circle': self.__generate_unit_circle
        }
        try:
            return shape_dict[shape](dataframe)
        except KeyError:
            print('KeyError: Shape keyword input not found.')


class GraphData:
    """ Graphs data points, their balls of given radius, and given complex.
        :key p: any valid numpy distance metric
    """
    def __init__(self, data_pts: np.ndarray, radius: float, simplices: gudhi.SimplexTree, p):
        self.data_pts = data_pts
        self.radius = radius
        self.simplices__ = simplices
        self.p = p
        self.__graph_complex()

    def __plot_ball(self, i, x_coord, y_coord):
        COLORS, OPACITY = ['r', 'g', 'b', 'm'], 0.25
        if self.p == np.inf:
            return RegularPolygon((x_coord, y_coord), numVertices=4, orientation=math.pi/4, radius=self.radius,
                                  color=COLORS[i % len(COLORS)], alpha=OPACITY)
        return plt.Circle((x_coord, y_coord), radius=self.radius, color=COLORS[i % len(COLORS)], alpha=OPACITY)

    def __plot_data_points(self, axes):
        for i, (x_coord, y_coord) in enumerate(self.data_pts):
            # PLOT DATA POINT
            plt.scatter(x_coord, y_coord, color='black', marker='o', s=10)
            # PLOT BALL CENTERED AT DATA POINT
            ball = self.__plot_ball(i, x_coord, y_coord)
            axes.add_artist(ball)

    def __plot_simplices(self):
        OPACITY = 0.20
        for (simplex_idx, _) in self.simplices__:
            # each data point is given a number (called its vertex name)
            # simplex_idx is the tuple of the vertex names that make the simplex up
            simplex_coords = [self.data_pts[idx] for idx in simplex_idx]
            # simplex_coords is the list (x, y)-coords of each vertex
            simplex = plt.Polygon(simplex_coords, color='black', alpha=OPACITY, linewidth=2)
            plt.gca().add_patch(simplex)

    def __graph_complex(self):
        figure, axes = plt.subplots()
        axes.set_aspect(1)
        axes.autoscale(enable=True)
        self.__plot_data_points(axes)
        self.__plot_simplices()


class RipsComplexes:
    """
    :key distance_matrix: <type> scipy distance matrix: if None, compute the distance between all of the data points
    """
    def __init__(self, data_points: pd.DataFrame, time: float, distance_matrix=None, p=np.inf, graph_results=False):
        self.data_pts = data_points
        self.diameter, self.p = 2 * time, p
        self.distance_matrix = self.__get_distance_matrix(distance_matrix)
        self.complex = self.__generate_rips_complex()

        if graph_results:
            GraphData(self.data_pts.values, time, self.complex, self.p)
            plt.show()

    def __get_distance_matrix(self, distance_matrix):
        if distance_matrix is None:
            return scipy.spatial.distance_matrix(self.data_pts, self.data_pts, p=self.p)
        return distance_matrix

    def __generate_rips_complex(self):
        rips_complex = gudhi.RipsComplex(distance_matrix=self.distance_matrix, max_edge_length=self.diameter)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        return simplex_tree.get_filtration()

    def __repr__(self):
        results = []
        print_format = '%s -> %.2f'
        for filtered_value in self.complex.get_filtration():
            results.append(print_format % tuple(filtered_value) + '\n')
        return ''.join(results)


class UpdateRipsComplex:
    """ Computes the Rips Complex of the relevant vertices (sufficiently close data points to the new_data_pt)."""
    def __init__(self, new_data_pt, data_pts, current_cplx, distance_mat, time, p=np.inf):
        self.new_data_pt = new_data_pt
        self.distances_to_new_data_pt = scipy.spatial.distance_matrix(data_pts.values, new_data_pt, p=p)
        # filtered_values = (filtered vertex names, filtered data points, filtered distance matrix)
        self.filtered_values = self.__get_filtered_values(data_pts, distance_mat, 2*time)
        self.updated_distance_mat = self.__update_distance_mat()
        self.new_simplices = self.__compute_rips_complex(time, p)

    def __get_filtered_values(self, data_pts, distance_mat, diameter):
        """ Get the relevant data points and their related values.
            filtered_vertex_names : indices of the relevant vertices
            filtered_data_pts : the (x, y)-coords of the relevant vertices
            filtered_distances : submatrix of distance_mat of the rows/columns of the relevant vertices
        """
        filtered_vertex_names = np.where(self.distances_to_new_data_pt <= diameter)[0]
        filtered_data_pts = data_pts.loc[filtered_vertex_names]
        filtered_distances = distance_mat[np.ix_(filtered_vertex_names, filtered_vertex_names)]
        return filtered_vertex_names, filtered_data_pts, filtered_distances

    def __update_distance_mat(self):
        """ Adding the new_data_pt distance values to filtered distance_mat. """
        updated_distance_mat = self.filtered_values[2].copy()
        filtered_dist_to_new_data_pt = self.distances_to_new_data_pt[np.ix_(self.filtered_values[0])]
        row_to_append = filtered_dist_to_new_data_pt.T
        col_to_append = np.append(filtered_dist_to_new_data_pt, [[0]], axis=0)

        updated_distance_mat = np.concatenate((updated_distance_mat, row_to_append), 0)
        updated_distance_mat = np.concatenate((updated_distance_mat, col_to_append), 1)
        return updated_distance_mat

    def __compute_rips_complex(self, time, p):
        """ Compute Rips Complex of the relevant vertices. """
        updated_data_pts = pd.concat((self.filtered_values[1], self.new_data_pt))
        print(updated_data_pts)
        new_simplices = RipsComplexes(updated_data_pts, time, distance_matrix=self.updated_distance_mat,
                                      p=p, graph_results=True).complex
        return new_simplices

    def __repr__(self):
        results = []
        print_format = '%s -> %.2f'
        for filtered_value in self.new_simplices:
            results.append(print_format % tuple(filtered_value) + '\n')
        return ''.join(results)


# TODO: the vertex names no longer match


def main():
    TIME = 0.15
    sample_data_class = SampleData(num_data_pts=100)
    data, distance_mat = sample_data_class.data, sample_data_class.distance_mat

    rips_complex = RipsComplexes(data, TIME, graph_results=True, p=np.inf).complex

    new_data_pt = SampleData(num_data_pts=1).data
    new_rips_cplx = UpdateRipsComplex(new_data_pt, data, rips_complex, distance_mat, TIME, p=np.inf)


if __name__ == '__main__':
    main()

