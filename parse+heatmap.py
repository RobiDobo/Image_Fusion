
import hashlib
import os
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import re
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

def rssi_to_percentage(rssi, min_rssi, max_rssi):
    """
    Map RSSI values to a percentage scale (0-100).
    
    Args:
        rssi (float): The RSSI value (in dBm).
        min_rssi (float): The minimum RSSI value (e.g., -100 dBm).
        max_rssi (float): The maximum RSSI value (e.g., 0 dBm).
    
    Returns:
        float: The mapped percentage value (0-100).
    """
    if rssi < min_rssi:
        return 0
    elif rssi > max_rssi:
        return 100
    else:
        return round((rssi - min_rssi) / (max_rssi - min_rssi) * 100)
    
def parse_wifi_file(file_path):
    """Read and parse WiFi file with a comma-separated format"""
    #print(f"Parsing file: {file_path}")
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Process the lines to extract SSID, BSSID, and Signal
    data = []
    for line in lines[1:]:  # Skip header line
        parts = line.strip().split(',')  # Split by commas
        if len(parts) >= 3:
            ssid = parts[0].strip()
            bssid = parts[1].strip()
            signal = parts[2].strip('%')  # Remove % sign
            
            try:
                signal_value = float(signal)
                data.append({
                    'ssid': ssid,
                    'bssid': bssid,
                    'signal': signal_value
                })
            except ValueError:
                print(f"Invalid signal value in line: {line.strip()}")
                continue
    
    df = pd.DataFrame(data)
    #print(f"Parsed {len(df)} rows from {file_path}")
    return df

def extract_coordinates(folder_name):
    """Extract X and Y coordinates from folder name (e.g., 'X=1,Y=1')"""
    #print(f"Extracting coordinates from folder name: {folder_name}")
    match = re.match(r'X=(\d+),Y=(\d+)', folder_name)
    if match:
        x, y = int(match.group(1)), int(match.group(2))
        #print(f"Extracted coordinates: X={x}, Y={y}")
        return x, y
    #print("No valid coordinates found.")
    return None

def read_measurements(base_folder):
    """Read all measurements from the nested folder structure"""
    print(f"Reading measurements from base folder: {base_folder}")
    all_measurements_vertical = []
    all_measurements_horizontal = []
    
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        
        # Skip if not a directory or doesn't match our naming pattern
        if not os.path.isdir(folder_path):
            #print(f"Skipping non-directory: {folder_path}")
            continue
            
        coords = extract_coordinates(folder_name)
        if not coords:
            continue
            
        x, y = coords
        
        # Read vertical polarization (Wi-Fi 2 folder)
        wifi_folder = os.path.join(folder_path, 'Realtek RTL8188FTV Wireless LAN 802.11n USB 2.0 Network Adapter')#or Realtek RTL8188FTV Wireless LAN 802.11n USB 2.0 Network Adapter
        if os.path.exists(wifi_folder):
            #print(f"Processing vertical polarization in: {wifi_folder}")
            for file in os.listdir(wifi_folder):
                if file.endswith('.csv'):
                    df = parse_wifi_file(os.path.join(wifi_folder, file))
                    df['x'] = x
                    df['y'] = y
                    all_measurements_vertical.append(df)
        
        # Read horizontal polarization (Wi-Fi 3 folder)
        wifi2_folder = os.path.join(folder_path, 'Realtek RTL8188FTV Wireless LAN 802.11n USB 2.0 Network Adapter #2')#or Realtek RTL8188FTV Wireless LAN 802.11n USB 2.0 Network Adapter #2
        if os.path.exists(wifi2_folder):
            #print(f"Processing horizontal polarization in: {wifi2_folder}")
            for file in os.listdir(wifi2_folder):
                if file.endswith('.csv'):
                    df = parse_wifi_file(os.path.join(wifi2_folder, file))
                    df['x'] = x
                    df['y'] = y
                    all_measurements_horizontal.append(df)
    
    
    
    # Combine measurements
    vertical_df = pd.concat(all_measurements_vertical, ignore_index=True) if all_measurements_vertical else pd.DataFrame()
    horizontal_df = pd.concat(all_measurements_horizontal, ignore_index=True) if all_measurements_horizontal else pd.DataFrame()
    
    print(f"Total vertical measurements: {len(vertical_df)}")
    print(f"Total horizontal measurements: {len(horizontal_df)}")
    
    
    
    return vertical_df, horizontal_df

def estimate_bssid_locations(df, ssid_filter):
    # Filter the DataFrame for the specific SSID
    df_filtered = df[df['ssid'] == ssid_filter]
    df_filtered = df_filtered[df_filtered['signal'] < -10]  # Filter out erroneous RSSI values > -10 dBm 
    
    # Keep only the max signal per (x, y, bssid)
    df_cleaned = df_filtered.loc[df_filtered.groupby(['x', 'y', 'bssid'])['signal'].idxmax()]
    df_cleaned = df_cleaned.sort_values(by=['x', 'y'])

    # Convert RSSI to percentage
    df_cleaned['signal_percentage'] = df_cleaned['signal'].apply(lambda rssi: rssi_to_percentage(rssi, -100, -20))

    # Initialize list to store estimated locations
    bssid_locations = []

    # Loop through each unique BSSID
    for bssid in df_cleaned['bssid'].unique():
        # Filter by each BSSID
        df_bssid = df_cleaned[df_cleaned['bssid'] == bssid]
        
        # Get the signal grid and coordinates for the current BSSID
        x_coords = df_bssid['x'].values
        y_coords = df_bssid['y'].values
        signal_percentages = df_bssid['signal_percentage'].values

        # Flatten the arrays for easier calculation
        x_coords_flat = x_coords.flatten()
        y_coords_flat = y_coords.flatten()
        signal_flat = signal_percentages.flatten()

        # Ensure the sum of signal_flat is not zero to prevent division by zero
        if np.sum(signal_flat) == 0:
            print(f"Warning: Sum of signals is zero for BSSID: {bssid}, skipping.")
            continue

        # Calculate weighted sums of coordinates
        weighted_x = round(np.sum(x_coords_flat * signal_flat) / np.sum(signal_flat))
        weighted_y = round(np.sum(y_coords_flat * signal_flat) / np.sum(signal_flat))

        # Print the estimated signal origin (weighted center of mass)
        #print(f"Estimated signal origin (center of mass): ({weighted_x}, {weighted_y})")



        # Append the result to the list
        bssid_locations.append({
            'ssid': ssid_filter,
            'bssid': bssid,
            'x_marker': weighted_x,
            'y_marker': weighted_y
        })

    # Convert the list to a DataFrame
    bssid_location_df = pd.DataFrame(bssid_locations)

    # Print the DataFrame with estimated locations
    #print(bssid_location_df)

    # Save this DataFrame to a CSV file for review
    #bssid_location_df_filename = f"bssid_locations_{ssid_filter}.csv"
    #bssid_location_df.to_csv(bssid_location_df_filename, index=False)
    #print(f"Estimated BSSID locations saved as {bssid_location_df_filename}.")

    return bssid_location_df

def process_all_ssids(df,output_folder="bssid_locations"):
    os.makedirs(output_folder, exist_ok=True)
    # Extract unique SSIDs from the dataframe
    ssids = df['ssid'].unique()

    # Initialize a list to store all the results
    all_bssid_locations = []

    # Loop through each SSID and apply the estimate_bssid_locations function
    for ssid in ssids:
        #print(f"Processing SSID: {ssid}")
        bssid_location_df = estimate_bssid_locations(df, ssid)  # Call the function for each SSID
        all_bssid_locations.append(bssid_location_df)  # Append the result to the list

    # Concatenate all the results into one DataFrame
    combined_bssid_locations = pd.concat(all_bssid_locations, ignore_index=True)

    # Define the file path
    combined_filename = os.path.join(output_folder, "combined_bssid_locations.csv")
    combined_bssid_locations.to_csv(combined_filename, index=False)
    #print(f"All estimated BSSID locations saved as {combined_filename}.")

    return combined_bssid_locations


def create_signal_heatmap(df, ssid_filter, df_name, title=None):
    
    df_filtered = df[df['ssid'] == ssid_filter]  # First filter by SSID
    # Filter by BSSID
    df_filtered = df_filtered[df_filtered['bssid'] == bssid_filter]
    df_filtered = df_filtered[df_filtered['signal'] < -10] 
    # Filter out erroneous RSSI values > -10 dBm 
    

    # Keep only max signal per (x, y, bssid)
    df_cleaned = df_filtered.loc[df_filtered.groupby(['x', 'y', 'bssid'])['signal'].idxmax()]
    # Sort by 'x' and 'y' to ensure the coordinates are in order
    df_cleaned = df_cleaned.sort_values(by=['x', 'y'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
        print(df_cleaned)
    # Convert RSSI to percentage
    df_cleaned['signal_percentage'] = df_cleaned['signal'].apply(lambda rssi: rssi_to_percentage(rssi, -100, -20))

    # Save the cleaned dataframe to a CSV file for review
    cleaned_filename = f"df_cleaned_{df_name}.csv"
    df_cleaned.to_csv(cleaned_filename, index=False)
    print(f"DataFrame after filtering saved as {cleaned_filename}.")
    
    # Initialize a grid to represent signal strengths for a 30x5 area (original grid size)
    grid_x = 30
    grid_y = 5
    grid_z = np.zeros((grid_x, grid_y))  # Initialize the grid with zero values


    # Populate the grid with signal percentages
    for index, row in df_cleaned.iterrows():
        x_pos = int(row['x']) - 1  # Adjust for 0-based indexing
        y_pos = int(row['y']) - 1  # Adjust for 0-based indexing
        grid_z[x_pos, y_pos] = row['signal_percentage']

    
    # Scale the heatmap to match the target image size (7500x675)
    figsize = (1500 / 300, 375 / 300)  # Convert pixels to inches at 300 dpi
    dpi = 300  # High quality
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    #grid_z_smoothed = gaussian_filter(grid_z, sigma=0.5)   
    # Plot the heatmap using Seaborn with the grid
    sns.heatmap(grid_z.T,  # Transpose to match x, y axes properly
                cmap='YlOrRd',  # Colormap for signal strength
                
                xticklabels=False,  # Hide x-axis labels (1 to 30)
                yticklabels=False,  # Hide y-axis labels (1 to 5)
                ax=ax,
                cbar=False)  # Remove colorbar initially for custom placement

    # Invert the Y-axis to match image coordinates
    ax.invert_yaxis()

    # Optionally, remove axis labels and title
    ax.axis('off')
    if title:
        ax.set_title(title)

    
    
    print("Heatmap created.")
    
    # Filenames for both images
    filename_with_labels = f"heatmap_with_labels_{df_name}.png"
    filename_scaled = f"heatmap_scaled_{df_name}.png"
    
    # Save the heatmap with labels first
    plt.savefig(filename_with_labels, bbox_inches="tight", pad_inches=0, transparent=True)
    print(f"Heatmap with labels saved as {filename_with_labels}.")
    
    # Now scale the image to 7500x675 pixels (without labels)
    img = Image.open(filename_with_labels)
    img = img.resize((1500, 375), Image.Resampling.LANCZOS)  # Resize to 7500x675 pixels
    img.save(filename_scaled)  # Save the resized image without labels
    print(f"Scaled heatmap saved as {filename_scaled}.")
    
    return fig
"""
def create_signal_heatmap(df,ssid_filter,df_name,title=None):
   
        
    
    if ssid_filter:
        print(f"Filtering data for SSID: {ssid_filter}")
        df = df[df['ssid'] == ssid_filter]
    
    if df.empty:
        print(f"No data found for SSID: {ssid_filter}")
        return
    
    print("DataFrame after filtering:")
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    #    print(df)
    print("Creating heatmap...")
    # Print the range of x and y coordinates
    print(f"x range: {df['x'].min()} - {df['x'].max()}")
    print(f"y range: {df['y'].min()} - {df['y'].max()}")
    # Apply Gaussian filter to smooth the signal values
    #df['smoothed_signal'] = gaussian_filter(df['signal'].values, sigma=4)
    
    # Create regular grid for interpolation
    grid_x, grid_y = np.mgrid[ 1:31:1, # 1 to 30 inclusive
                              1:6:1 # 1 to 5 inclusive
                              ]
    
    # Interpolate signal values
    points = df[['x', 'y']].values
    values = df['signal'].values
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic',fill_value=np.nan)
    
    # Ensure `grid_z` is not empty
    if grid_z is None or grid_z.size == 0:
        print("No valid data for heatmap.")
        return
    # Mask the NaN values 
    grid_z_masked = np.ma.masked_invalid(grid_z)
    # Create the heatmap
    figsize = (7500 / 300, 675 / 300)  # Convert pixels to inches at 300 dpi
    dpi = 300  # High quality
    fig, ax = plt.subplots(figsize=figsize,dpi=dpi) # Generate a figure and return figure and axis handle    
    
    #ax.set_title(title or f'Signal Strength Heatmap{" for " + ssid_filter if ssid_filter else ""}')
    
    # Higher percentage = stronger signal, so use regular colormap
    sns.heatmap(grid_z.T,
                cmap='YlOrRd',
                cbar_kws={'label': 'Signal Strength (%)'},
                xticklabels=False, # 1 to 30 inclusive 
                yticklabels=False, # 1 to 5 inclusive
                ax=ax,
                cbar=False)
    ax.invert_yaxis()
    #ax.set_xlabel('X Position')
    #ax.set_ylabel('Y Position')
    #ax.set_aspect('equal')
    
    print("Heatmap created.")
     # Filenames for both images
    filename_with_labels = f"heatmap_with_labels_{df_name}.png"
    filename_scaled = f"heatmap_scaled_{df_name}.png"
    
    # Save the heatmap with labels first
    plt.savefig(filename_with_labels, bbox_inches="tight", pad_inches=0, transparent=True)
    print(f"Heatmap with labels saved as {filename_with_labels}.")
    
    # Now scale the image to 7500x675 pixels (without labels)
    img = Image.open(filename_with_labels)
    img = img.resize((7500, 675), Image.Resampling.LANCZOS)  # Resize to 7500x675 pixels
    img.save(filename_scaled)  # Save the resized image without labels
    print(f"Scaled heatmap saved as {filename_scaled}.")

    #plt.show()  # Show the plot
    return fig
"""
def list_available_ssids(base_folder):
    """Print all available SSIDs in the dataset"""
    print("Listing available SSIDs...")
    vertical_df, _ = read_measurements(base_folder)
    if not vertical_df.empty:
        print("Available SSIDs in the dataset:")
        for ssid in sorted(vertical_df['ssid'].unique()):
            print(f"- {ssid}")
        return vertical_df['ssid'].unique()
    print("No SSIDs found.")
    return []

def collect_and_crop_images_with_folder_names(base_folder, output_folder, crop_box):
    """
    Collect all .png images from nested folders, save them with folder-based names,
    and crop each image to the specified crop box.

    Args:
    - base_folder (str): The base folder to search for .png files.
    - output_folder (str): The folder where cropped images will be saved.
    - crop_box (tuple): The region to crop (left, upper, right, lower).
    """
    print(f"Scanning for images in base folder: {base_folder}")
    os.makedirs(output_folder, exist_ok=True)
    
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)

        # Skip if not a directory
        if not os.path.isdir(folder_path):
            print(f"Skipping non-directory: {folder_path}")
            continue

        # Check for .png images in this folder
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith('.png'):
                output_path = os.path.join(output_folder, f"{folder_name}.png")
               
                # Skip if the output file already exists
                if (os.path.exists(output_path) or (output_path.endswith('cropped_images.png'))):
                    #print(f"Skipping existing cropped image: {output_path}")
                    continue
                
                file_path = os.path.join(folder_path, file_name)
                #print(f"Found image: {file_path}")
                
                # Open and crop the image
                with Image.open(file_path) as img:
                    cropped_img = img.crop(crop_box)
                    
                    # Save the cropped image to the output folder with folder-based name
                    output_path = os.path.join(output_folder, f"{folder_name}.png")
                    cropped_img.save(output_path)
                    print(f"Saved cropped image as: {output_path}")

def concatenate_images(image_folder):
    # Sort and group images by X value
    grouped_images = {}
    
    for file_name in os.listdir(image_folder):
        if file_name.endswith('.png'):
            # Extract X and Y from the file name
            x = int(file_name.split(',')[0].split('=')[1])
            y = int(file_name.split(',')[1].split('=')[1].split('.')[0])
            
            if x not in grouped_images:
                grouped_images[x] = []
            grouped_images[x].append((y, file_name))
    
    # Sort images by Y value for each X group
    for x in grouped_images:
        grouped_images[x].sort()  # Sorting by Y value (1 to 5)
    
    # List to store vertically stacked images for each X
    stacked_images = []

    # Process each X group
    for x in range(1, 31):  # X values from 1 to 30
        if x in grouped_images:
            # Load images for the current X
            images = []
            for _, file_name in grouped_images[x]:
                img = Image.open(os.path.join(image_folder, file_name))
                images.append(img)
            
            # Reverse the order of images so Y=1 is at the bottom
            images.reverse()
            
            # Vertically stack images for the same X value
            total_height = sum(img.height for img in images)
            max_width = max(img.width for img in images)
            
            stacked_image = Image.new('RGBA', (max_width, total_height))  # Create a new image with the correct mode
            y_offset = 0
            for img in images:
                stacked_image.paste(img, (0, y_offset))
                y_offset += img.height

            # Append the vertically stacked image for this X to the list
            stacked_images.append(stacked_image)
    
    # Now, concatenate the vertically stacked images horizontally
    total_width = sum(img.width for img in stacked_images)
    max_height = max(img.height for img in stacked_images)
    
    final_image = Image.new('RGBA', (total_width, max_height))  # Create final image with the correct mode
    
    x_offset = 0
    for img in stacked_images:
        final_image.paste(img, (x_offset, 0))
        x_offset += img.width
    final_image.save('final_image.png') # Save the final image
    return final_image

def overlay_heatmap_on_image(heatmap_path, final_image_path, output_path,transparency_factor=0.5):
    # Open the heatmap and final image
    heatmap = Image.open(heatmap_path).convert("RGBA")  # Ensure it's in RGBA (with alpha channel)
    final_image = Image.open(final_image_path).convert("RGBA")  # Same here for final image
    colorbar = Image.open("colorbar.png").convert("RGBA")  # Load the colorbar image
    # Apply Gaussian blur to the heatmap
    heatmap = heatmap.filter(ImageFilter.GaussianBlur(radius=2))
    # Ensure the heatmap and final image are the same size
    if heatmap.size != final_image.size:
        heatmap = heatmap.resize(final_image.size, Image.Resampling.LANCZOS)  # Resize heatmap to match
    
     # Resize colorbar to 375 height while maintaining aspect ratio
    colorbar_width = int(colorbar.width * (375 / colorbar.height))
    colorbar = colorbar.resize((colorbar_width, 375), Image.Resampling.LANCZOS)    
    
    # Adjust the alpha (transparency) of the heatmap
    heatmap_with_alpha = heatmap.split()
    alpha = heatmap_with_alpha[3]  # Get the alpha channel
    alpha = alpha.point(lambda p: p * transparency_factor)  # Adjust transparency
    heatmap_with_alpha = Image.merge("RGBA", (heatmap_with_alpha[0], heatmap_with_alpha[1], heatmap_with_alpha[2], alpha))

    # Combine the images by overlaying the heatmap with transparency
    combined_image = Image.alpha_composite(final_image, heatmap_with_alpha)
    
    # Combine with colorbar on the right side
    combined_width = combined_image.width + colorbar.width
    combined_height = max(combined_image.height, colorbar.height)
    final_combined = Image.new("RGBA", (combined_width, combined_height))

    # Paste the combined image and colorbar next to each other
    final_combined.paste(combined_image, (0, 0))
    final_combined.paste(colorbar, (combined_image.width, (combined_height - colorbar.height) // 2), colorbar)

    # Save the resulting image
    final_combined.save(output_path, "PNG")
    print(f"Overlayed image saved as {output_path} \n")

    # Show the combined image (optional)
    final_combined.show()

import numpy as np
import pandas as pd
from typing import Dict, List
from scipy.optimize import minimize

def analyze_wifi_signals(data: pd.DataFrame) -> Dict:
    """
    Analyze WiFi signal measurements from fixed position scanners.
    
    Args:
        data: DataFrame with columns ['x', 'y', 'ssid', 'bssid', 'signal']
             x, y: scanner coordinates
             signal: signal strength percentage (0-100)
             ssid: network name
             bssid: MAC address of access point
    
    Returns:
        Dictionary containing analysis results per BSSID
    """
    results = {}
    
    # Group by BSSID
    for bssid in data['bssid'].unique():
        bssid_data = data[data['bssid'] == bssid]
        #print("wow")
        
        #print(bssid_data)
        # Get SSID for this BSSID
        ssid = bssid_data['ssid'].iloc[0]
        
        # Find strongest signal value
        max_signal = bssid_data['signal'].max()
        
        # Identify all strong points with the maximum signal
        strong_points = bssid_data[bssid_data['signal'] == max_signal][['x', 'y', 'signal']].reset_index(drop=True)
        
        # Estimate source location
        estimated_location = estimate_source_location(bssid_data)
        
        # Calculate statistics
        stats = calculate_signal_stats(bssid_data)
        
        results[bssid] = {
            'ssid': ssid,
            'strongest_signal': strong_points,
            'estimated_location': estimated_location,
            'stats': stats
        }
    
    return results

def estimate_source_location(data: pd.DataFrame, sigma: float = 4) -> Dict[str, float]:
    """
    Estimate the likely source location using signal strengths as weights.
    RIGHT NOW ONLY USES VERTICALLY POLARISED ANTENNA, WE WANT TO USE BOTH POLARISATIONS ESPECIALLY IF Strongest Signal location
    is different than estimated location(for example let's see if its fine to use this if x of strongest signal is offset to est_x >=5 )
    
    Args:
        data: DataFrame with measurements for a single BSSID
    
    Returns:
        Dictionary with estimated x, y coordinates
    """
    
    # Use signal strength as weights
    #weights = data['signal'] / data['signal'].sum()  # Convert to 0-1 scale
     # Smooth signal strengths with Gaussian filter
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    #    print(data)
    data['smoothed_signal'] = gaussian_filter(data['signal'].values, sigma=sigma)
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    #    print(data)
    weights =  data['smoothed_signal'] / data['smoothed_signal'].sum()  # Normalize weights
    # Calculate weighted average of coordinates
    est_x = round(np.average(data['x'], weights=weights))
    est_y = round(np.average(data['y'], weights=weights))
    #rounded_y = round(est_y)
    # Find neighbors in y-1 and y+1
    neighbors = data[(data['x'] == est_x - 1) | (data['x'] == est_x) | (data['x'] == est_x + 1)]
    
    # Check for the point with the highest signal strength among neighbors
    if not neighbors.empty:
        best_neighbor = neighbors.loc[neighbors['signal'].idxmax()]
        print(f"But it might be x={best_neighbor['x']}, y={best_neighbor['y']} found with signal {best_neighbor['signal']}")
    #if not neighbors.empty:
    #    # Add smoothed signal to the neighbors DataFrame for comparison
    #    neighbors['smoothed_signal'] = gaussian_filter(neighbors['signal'].values, sigma=sigma)
    #    best_neighbor = neighbors.loc[neighbors['smoothed_signal'].idxmax()]
    #    print(f"Might be x={best_neighbor['x']}, y={best_neighbor['y']} found with smoothed signal {best_neighbor['smoothed_signal']}")
    # Step 2: Filter for high signal strength
    threshold = data['smoothed_signal'].mean()  # Define a threshold for high signal strength
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    #    print(data)
    high_signal_df = data[data['smoothed_signal'] >= threshold]

    # Step 3: Clustering using K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust n_clusters as needed
    #print("kmeans!!!!!")
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    #    print(high_signal_df)
    high_signal_df['cluster'] = kmeans.fit_predict(high_signal_df[['x', 'y', 'signal']])
    
    # Step 4: Calculate centroids of clusters
    centroids = high_signal_df.groupby('cluster').agg({'x': 'mean', 'y': 'mean', 'smoothed_signal': 'mean'}).reset_index()

    # Step 1: Identify the centroids with the highest signal values
    high_signal_centroids = centroids[centroids['smoothed_signal'] >= centroids['smoothed_signal'].mean()]

    # Step 2: Calculate the mean of the high signal centroids
    if not high_signal_centroids.empty:
        estimated_x = high_signal_centroids['x'].mean()
        estimated_y = high_signal_centroids['y'].mean()
        print(f"Estimated AP Location using mean of centroisd: x={round(estimated_x)}, y={round(estimated_y)}")
    else:
        print("No high signal centroids found.")   
        
        
    return {'x': est_x, 'y': est_y}

def calculate_signal_stats(data: pd.DataFrame) -> Dict:
    """
    Calculate signal statistics for a single BSSID.
    
    Args:
        data: DataFrame with measurements
    
    Returns:
        Dictionary with signal statistics
    """
    return {
        'mean_signal': data['signal'].mean(),
        'max_signal': data['signal'].max(),
        'min_signal': data['signal'].min(),
        'std_signal': data['signal'].std(),
        'num_measurements': len(data),
        'measurement_area': {
            'min_x': data['x'].min(),
            'max_x': data['x'].max(),
            'min_y': data['y'].min(),
            'max_y': data['y'].max()
        }
    }
def add_ssid_markers_to_image(csv_folder, image_path, output_image_path):
    # Load the final image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Set up font for SSID names (you may need to adjust this path)
    try:
        font = ImageFont.truetype("arial.ttf", 10)  # Or any font available on your system
    except IOError:
        font = ImageFont.load_default()  # Use default font if Arial is not available
    
    # Create a dictionary to store the first SSID per (x, y) grid cell
    grid_cell_ssid = {}

    # Loop through the CSV files in the specified folder
    for csv_file in os.listdir(csv_folder):
        if csv_file.endswith(".csv"):
            csv_path = os.path.join(csv_folder, csv_file)
            df = pd.read_csv(csv_path)
            
            # Loop through the DataFrame rows and place SSID names at the coordinates
            for _, row in df.iterrows():
                ssid = row['ssid']
                x_marker = row['x_marker']
                y_marker = row['y_marker']

                # Ensure that the grid cell (x_marker, y_marker) is not already occupied
                if (x_marker, y_marker) not in grid_cell_ssid:
                    # If not, assign the SSID to this grid cell
                    grid_cell_ssid[(x_marker, y_marker)] = ssid

    # Loop through the dictionary and draw the SSID names in the middle of each grid cell
    for (x_marker, y_marker), ssid in grid_cell_ssid.items():
        # Calculate the pixel coordinates for the center of the grid cell
        x_pixel = int((x_marker - 1) * (1500 / 30)) + int((1500 / 30) / 2)
        y_pixel = int((y_marker - 1) * (375 / 5)) + int((375 / 5) / 2)
        
        # Draw the SSID name at the calculated coordinates (centered)
        draw.text((x_pixel, y_pixel), ssid, fill="yellow", font=font, anchor="mm")
    
    # Save the output image
    img.save(output_image_path)
    print(f"Image saved with SSID markers as {output_image_path}")

# Example usage
if __name__ == "__main__":
    # Use the current directory as base folder
    base_folder = os.getcwd()
    
    #prints available ssids
    ssids = list_available_ssids(base_folder)
    
    if len(ssids) > 0:
        # specify which ssid to work on
        #B129,A0:F3:C1:6A:30:36
        #A_306,F8:D1:11:41:11:BC
        #B714,00:1F:CF:11:7F:56
        #DCTI-PUBLIC,E0:1C:FC:94:19:32
        #B029 A8:42:A1:69:A0:56
        #DecanatET_Guest ea:c3:2a:c9:28:ba
        #B329,E8:48:B8:64:B6:C3,-62,2.422
        #A304,98:DA:C4:3F:10:A7,-62,2.462
        ssid_filter = "A_306"  
        bssid_filter="F8:D1:11:41:11:BC"
        vertical_df, horizontal_df = read_measurements(base_folder)
        
        # Generate and show individual heatmaps
        
        create_signal_heatmap(vertical_df, ssid_filter,df_name="vertical")
        create_signal_heatmap(horizontal_df, ssid_filter,df_name="orizontal")
    else:
        print("No measurements found in the specified folder.")
        
    # Crop and concatenate images
    #crop_box = (200, 165, 450, 300) 
    crop_box = (290, 210, 340, 285)
    #or (290, 195, 340, 265)

    #collect_and_crop_images_with_folder_names(base_folder, 'cropped_images', crop_box)
    #concatenate_images('cropped_images').show() #COMMENTED BECAUSE ALREADY COLLECTED CROPPED IMAGES AND MADE FINAL IMAGE
    overlay_heatmap_on_image('heatmap_scaled_vertical.png', 'final_image.png', 'final_output_vertical.png')
    overlay_heatmap_on_image('heatmap_scaled_orizontal.png', 'final_image.png', 'final_output_horizontal.png')
    combined_bssid_locations_df = process_all_ssids(vertical_df,"vertical")
    combined_bssid_locations_df = process_all_ssids(horizontal_df,"horizontal")
    add_ssid_markers_to_image("horizontal","final_output_horizontal.png","horizontal_final_image_with_ssid_markers.png")
    add_ssid_markers_to_image("vertical","final_output_vertical.png","vertical_final_image_with_ssid_markers.png")
"""
vertical_df_ssid = vertical_df[vertical_df['ssid'] == ssid_filter]
results = analyze_wifi_signals(vertical_df_ssid)
print("Available BSSIDs in results:", results.keys())
bssid_info = results[bssid_filter]
print("Analysis for vertical polarization:")
print(f"SSID: {bssid_info['ssid']}")
print(f"Strongest Signal Points:")
print(bssid_info['strongest_signal'])
print(f"Estimated source at: ({bssid_info['estimated_location']['x']}, {bssid_info['estimated_location']['y']})using weighted average")
print(f"Signal statistics:")
print(f"Mean: {bssid_info['stats']['mean_signal']:.2f}")
#print(f"Max: {bssid_info['stats']['max_signal']:.2f}")
#print(f"Min: {bssid_info['stats']['min_signal']:.2f}")
print(f"Std: {bssid_info['stats']['std_signal']:.2f}")
#print(f"Number of measurements: {bssid_info['stats']['num_measurements']}")
#print(f"Measurement area: {bssid_info['stats']['measurement_area']}")

#with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
#        print(vertical_df_ssid)
        
horizontal_df_ssid = horizontal_df[horizontal_df['ssid'] == ssid_filter]
results = analyze_wifi_signals(horizontal_df_ssid)
bssid_info = results[bssid_filter]
print("Analysis for horizontal polarization:")
print(f"SSID: {bssid_info['ssid']}")
print(f"Strongest Signal Points: ({bssid_info['strongest_signal']})")
print(f"Estimated source at: ({bssid_info['estimated_location']['x']}, {bssid_info['estimated_location']['y']}) using weighted average")
print(f"Signal statistics:")
print(f"Mean: {bssid_info['stats']['mean_signal']:.2f}")
#print(f"Max: {bssid_info['stats']['max_signal']:.2f}")
#print(f"Min: {bssid_info['stats']['min_signal']:.2f}")
print(f"Std: {bssid_info['stats']['std_signal']:.2f}")
#print(f"Number of measurements: {bssid_info['stats']['num_measurements']}")
#print(f"Measurement area: {bssid_info['stats']['measurement_area']}")

"""
#with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
#        print(horizontal_df_ssid)



# Selecting RSSI and other relevant columns
#X = staffcm_df[['signal']].values

# Applying DBSCAN
#db = DBSCAN(eps=2, min_samples=3).fit(X)

# Adding the cluster labels to the DataFrame
#staffcm_df['cluster'] = db.labels_

# Find clusters with high RSSI
#high_rssi_clusters = staffcm_df[staffcm_df['signal'] > 70]  # Adjust threshold as needed
#print(high_rssi_clusters)

# Create a 3D plot for x, y, and RSSI
#fig = plt.figure(figsize=(10, 7))
#ax = fig.add_subplot(111, projection='3d')

# Scatter plot: x, y, RSSI with colors based on the clusters
#sc = ax.scatter(high_rssi_clusters['x'], high_rssi_clusters['y'], high_rssi_clusters['signal'], 
#                c=high_rssi_clusters['cluster'], cmap='viridis', s=50)

# Add labels and title
#ax.set_title("RSSI Clustering with DBSCAN - 3D Plot (X, Y, RSSI)")
#ax.set_xlabel("X Position")
#ax.set_ylabel("Y Position")
#ax.set_zlabel("Signal Strength (RSSI)")

# Add color bar to show RSSI cluster coloring
#cbar = plt.colorbar(sc)
#cbar.set_label('Cluster ID')

# Show the plot
#plt.show()