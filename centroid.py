import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
from PIL import Image

def find_star_centroids(image_path, threshold_factor=3, min_area=5, max_area=500):
    """
    Find weighted centroids of stars in an image.
    
    Parameters:
    - image_path: path to the star field image
    - threshold_factor: number of standard deviations above background for detection
    - min_area: minimum pixel area for a valid star
    - max_area: maximum pixel area to filter out artifacts
    
    Returns:
    - centroids: list of (x, y) centroid coordinates
    - intensities: list of total intensities for each star
    """
    
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=float)
    
    # Calculate background statistics
    background_mean = np.median(img_array)
    background_std = np.std(img_array)
    
    # Threshold: pixels above background + threshold_factor * std
    threshold = background_mean + threshold_factor * background_std
    binary_img = img_array > threshold
    
    # Label connected components
    labeled_img, num_features = ndimage.label(binary_img)
    
    centroids = []
    intensities = []
    
    # Process each detected blob
    for label_num in range(1, num_features + 1):
        # Get pixels belonging to this star
        star_mask = (labeled_img == label_num)
        area = np.sum(star_mask)
        
        # Filter by area to remove noise and artifacts
        if area < min_area or area > max_area:
            continue
        
        # Get coordinates and intensities of pixels in this blob
        y_coords, x_coords = np.where(star_mask)
        pixel_intensities = img_array[star_mask]
        
        # Calculate weighted centroid (center of mass)
        total_intensity = np.sum(pixel_intensities)
        x_centroid = np.sum(x_coords * pixel_intensities) / total_intensity
        y_centroid = np.sum(y_coords * pixel_intensities) / total_intensity
        
        centroids.append((x_centroid, y_centroid))
        intensities.append(total_intensity)
    
    return centroids, intensities, img_array

def visualize_results(img_array, centroids):
    """
    Visualize the image with detected centroids marked.
    """
    plt.figure(figsize=(12, 10))
    plt.imshow(img_array, cmap='gray')
    
    if centroids:
        x_coords, y_coords = zip(*centroids)
        plt.plot(x_coords, y_coords, 'r+', markersize=10, markeredgewidth=1.5)
    
    plt.title(f'Detected Stars: {len(centroids)} centroids')
    plt.colorbar(label='Intensity')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.show()

# Main execution
if __name__ == "__main__":
    image_path = "/Users/lundeencahilly/Desktop/github/samwise-startracker/photos/TIC308522539.01TOI5362.png"

    # Find centroids
    centroids, intensities, img_array = find_star_centroids(
        image_path,
        threshold_factor=5,
        min_area=5,
        max_area=500
    )
    
    print(f"Found {len(centroids)} stars")
    print("\nFirst 10 centroids (x, y) and their intensities:")
    for i, (centroid, intensity) in enumerate(zip(centroids[:10], intensities[:10])):
        print(f"Star {i+1}: x={centroid[0]:.2f}, y={centroid[1]:.2f}, intensity={intensity:.1f}")
    
    # Visualize
    visualize_results(img_array, centroids)
    
    # Optionally save centroids to file
    np.savetxt('star_centroids.txt', centroids, 
               header='x_centroid y_centroid', 
               fmt='%.3f')