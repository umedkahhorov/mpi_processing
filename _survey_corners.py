import numpy as np
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

def survey_corners(df, x, y, qhull_options='QG4'):
    """
    Computes the corners of a 2D dataset based on the convex hull and PCA.

    Parameters:
    - df: pandas.DataFrame containing the dataset with at least two columns.
    - x: str, the name of the column in `df` representing the x-coordinates.
    - y: str, the name of the column in `df` representing the y-coordinates.
    - qhull_options: str, options passed to the Qhull algorithm for computing the convex hull.

    Returns:
    - final_corners: numpy.ndarray, coordinates of the four corners of the dataset.
    - angle_deg: float, angle in degrees of the main principal direction with respect to the x-axis.
    """
    # Extract x and y coordinates from the DataFrame
    points = df[[x, y]].values

    # Compute the convex hull using Qhull options
    hull = ConvexHull(points, qhull_options=qhull_options)
    hull_points = points[hull.vertices]

    # Apply PCA to determine the orientation of the hull
    pca = PCA(n_components=2)
    pca.fit(hull_points)

    # Calculate the angle of the first principal component
    angle_rad = np.arctan2(*pca.components_[0][::-1])
    angle_deg = np.degrees(angle_rad) % 360

    # Transform hull points to the PCA coordinate system
    transformed_points = pca.transform(hull_points)

    # Find the extreme points along the principal components
    extreme_indices = np.array([transformed_points.argmin(axis=0), transformed_points.argmax(axis=0)]).flatten()
    extreme_points = transformed_points[extreme_indices]

    # Map the extreme points back to the original coordinate system
    final_corners = pca.inverse_transform(extreme_points)

    # Return the corners and the angle of the main principal direction
    return final_corners, angle_deg
