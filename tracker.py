import numpy as np
from scipy.optimize import linear_sum_assignment


class Tracker:

    def __init__(self, threshold, wait):

        self.i = 0
        self.tracks = []
        self.wait = wait
        self.threshold = threshold

    def update(self, positions, descriptors):
        """
        Arguments:
            positions: a numpy float array with shape [n, *].
            descriptors: a numpy float array with shape [n, c].
        Returns:
            a list of dicts.
        """

        num_detections = len(descriptors)
        data = zip(positions, descriptors)

        if len(self.tracks) == 0:

            self.tracks = [
                {'i': i, 'x': x, 'd': d, 'u': 0, 'd_ma': d}
                for i, (x, d) in enumerate(data, self.i)
            ]

            """
            The meaning of the keys:
            i - identifier,
            x - position,
            d - descriptor,
            u - number of steps since the last update,
            d_ma - moving average of descriptors.
            """

            self.i += num_detections
            return self.tracks

        if num_detections == 0:

            self.delete_old_tracks()
            return self.tracks

        previous_descriptors = np.stack([t['d'] for t in self.tracks])
        # it has shape [m, c]

        distances = cosine_distance(descriptors, previous_descriptors)
        # it has shape [n, m]

        matched = []
        distances_clipped = np.clip(distances, 0.0, self.threshold)

        row_ind, col_ind = linear_sum_assignment(distances_clipped)
        # they have shape [min(n, m)]

        for i, j in zip(row_ind, col_ind):

            if distances[i, j] > self.threshold:
                continue

            t = self.tracks[j]
            t['x'] = positions[i]
            t['d'] = descriptors[i]
            t['u'] = 0
            t['d_ma'] = 0.7 * t['d_ma'] + 0.3 * t['d']

            matched.append(i)

        for i, (x, d) in enumerate(data):
            if i not in matched:
                self.tracks.append({'i': self.i, 'x': x, 'd': d, 'u': 0, 'd_ma': d})
                self.i += 1

        self.delete_old_tracks()
        return self.tracks

    def delete_old_tracks(self):
        to_remove = []

        for i, t in enumerate(self.tracks):
            t['u'] += 1
            if t['u'] > self.wait:
                to_remove.append(i)

        for i in reversed(to_remove):
            self.tracks.pop(i)


def cosine_distance(x, y):
    """
    Arguments:
        x: a numpy float array with shape [n, c].
        y: a numpy float array with shape [m, c].
    Returns:
        a numpy float array with shape [n, m].
    """
    epsilon = 1e-8

    x_norm = np.sqrt((x**2).sum(1, keepdims=True))
    y_norm = np.sqrt((y**2).sum(1, keepdims=True))

    x = x / (x_norm + epsilon)
    y = y / (y_norm + epsilon)

    product = np.expand_dims(x, 1) * y  # shape [n, m, c]
    cos = product.sum(2)  # shape [n, m]
    return 1.0 - cos
