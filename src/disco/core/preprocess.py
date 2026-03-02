import copy
import pandas as pd

from ..config import DISCOConfig


def _calculate_dimensions(passing, failing):
    x_p, y_p = passing
    x_f, y_f = failing
    
    x_new = round((x_p + x_f) * 0.5)
    y_new = round((y_p + y_f) * 0.5)

    new = (x_new,y_new)
    
    if (new == failing):
        return passing

    return new

# Scales a coordinate pair from the grid with the original dimensions to the new dimensions
def _compute_new_coord_pair(coord_pair, dimensions, original_dimensions):
    c_1, c_2 = coord_pair
    x,y = dimensions
    x_i, y_i = original_dimensions

    x_factor = x / x_i
    y_factor = y / y_i

    return (round(c_1 * x_factor), round(c_2 * y_factor))

# Returns the df at the optimally reduced dimensions. Cell corrdinates are changed but all other data remains the same. 
def _reduce_dimensions(df_initial, original_dimensions):
    failing_dimensions = (1, 1) # Failing dimensions represent the most recent dimensions to run successfully without conflict- Start at (1, 1) to avoid divide by 0 in increment_dimensions
    passing_dimensions = original_dimensions
    dimensions = failing_dimensions
    restart_outer_loop = True

    # Loop executes as long as calculate_dimensions doesn't repeatedly return the same dimensions. At this point, we know we have found the optimal dimensions
    while (dimensions != passing_dimensions):

        # update which dimension 
        if restart_outer_loop:
            failing_dimensions = dimensions
        else:
            passing_dimensions = dimensions
        
        dimensions = _calculate_dimensions(passing_dimensions, failing_dimensions)
        print(f"Current dimensions: {dimensions}")

        
        df = copy.deepcopy(df_initial) # make a copy of the initial data to convert to new coordinates
        grouped_by_region = df.groupby('unique_region')
        restart_outer_loop = False # flag to indicate whether to restart the while loop

        coords = set()
        for region_name, region_data in grouped_by_region:
            coords.clear() # all of the coordinates with a cell at that coordinate thus far- we use a set for O(1) 
            
            # iterate through rows in original data set for the region
            for idx, row in region_data.iterrows():
                x = row['x']
                y = row['y']
                coord_pair = (x,y)
                # compute the new cell coordinates for the reduced grid space 
                new_coord_pair = _compute_new_coord_pair(coord_pair, dimensions, original_dimensions)

                # check to see if there is a conflict
                if (new_coord_pair in coords and coord_pair != (4915.0, 7055.0)):
                    # advance the while loop to increment the dimensions
                    print(f"failing coordinate pair: {new_coord_pair}, scaled from {coord_pair} in region {region_name}")
                    restart_outer_loop = True
                    break
                else:
                    x_n, y_n = new_coord_pair
                    df.at[idx, 'x'] = x_n
                    df.at[idx, 'y'] = y_n
                    coords.add(new_coord_pair)

            if(restart_outer_loop):
                break

            print(f"successfully reduced region {region_name}")

    return df, dimensions


def preprocess(
    df: pd.DataFrame,
    config: DISCOConfig
) -> tuple[pd.DataFrame, list[str]]:
    
    new_df, new_dimensions = _reduce_dimensions(df, config.original_dimensions)
    return (new_df, new_dimensions)