import numpy as np
import pandas as pd

def rotate_and_shift_polygon(df, x_col, y_col, degree_angle, new_corner_x, new_corner_y):
    """
    Rotate and shift a polygon defined by X and Y coordinates in a DataFrame.

    Parameters:
    - df: pandas.DataFrame containing the dataset with at least two columns for X and Y coordinates.
    - x_col: str, the name of the column in `df` representing the x-coordinates.
    - y_col: str, the name of the column in `df` representing the y-coordinates.
    - degree_angle: float, angle in degrees for the rotation.
    - new_corner_x: float, X coordinate of the corner from the original polygon.
    - new_corner_y: float, Y coordinate of the corner from the original polygon.

    Returns:
    - df_rotated: pandas.DataFrame with columns 'X_original', 'Y_original', 'X_rotated', 'Y_rotated'.
    """
    # Convert degree angle to radians
    radian_angle = np.deg2rad(degree_angle)

    # Calculate the centroid of the original polygon
    centroid_x = df[x_col].mean()
    centroid_y = df[y_col].mean()

    # Translate the original polygon so that its centroid is at the origin
    df_translated = df[[x_col, y_col]].subtract([centroid_x, centroid_y])

    # Apply the rotation transformation to the original polygon
    c, s = np.cos(radian_angle), np.sin(radian_angle)
    df_rotated = pd.DataFrame({
        'X_rotated': df_translated[x_col] * c - df_translated[y_col] * s,
        'Y_rotated': df_translated[x_col] * s + df_translated[y_col] * c
    })

    # Translate the rotated polygon back to its original position (center)
    df_rotated = df_rotated.add([centroid_x, centroid_y])

    # Find the corner coordinate in the original polygon
    corner_row = df[(df[x_col] == new_corner_x) & (df[y_col] == new_corner_y)]
    if corner_row.empty:
        raise ValueError("Corner coordinate not found in the original polygon.")

    # Calculate the shift needed to move the specified corner to the new position
    shift_x = new_corner_x - df_rotated.loc[corner_row.index, 'X_rotated']
    shift_y = new_corner_y - df_rotated.loc[corner_row.index, 'Y_rotated']

    # Shift the rotated polygon
    df_rotated = df_rotated.add([shift_x.values[0], shift_y.values[0]])

    # Combine the original and rotated coordinates into one DataFrame
    df_rotated['X_original'] = df[x_col]
    df_rotated['Y_original'] = df[y_col]

    # Reorder columns
    df_rotated = df_rotated[['X_original', 'Y_original', 'X_rotated', 'Y_rotated']]

    return df_rotated
