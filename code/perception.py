import numpy as np
import cv2
import scipy.misc
import skimage.draw as draw
import math
from scipy import ndimage


r_navigable_thresh, g_navigable_thresh, b_navigable_thresh = 120, 60, 120
dst_size = 5
bottom_offset = 6
img_shape = (160, 320, 3)
scale = 2 * dst_size   
world_size = 200 
source_points = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination_points = np.float32([[img_shape[1]/2 - dst_size, img_shape[0] - bottom_offset],
                      [img_shape[1]/2 + dst_size, img_shape[0] - bottom_offset],
                      [img_shape[1]/2 + dst_size, img_shape[0] - 2*dst_size - bottom_offset], 
                      [img_shape[1]/2 - dst_size, img_shape[0] - 2*dst_size - bottom_offset],
                      ])

def perception_step(Rover):  

  warped_img = perspect_transform(Rover.img, source_points, destination_points)
  img = color_thresh(warped_img, rgb_thresh=(r_navigable_thresh, g_navigable_thresh, b_navigable_thresh), rgb_shape=True) * 105.0

  # Get threshold images in both binary and rgb
  rgb_threshed_img = color_thresh(warped_img, rgb_thresh=(r_navigable_thresh, g_navigable_thresh, b_navigable_thresh), rgb_shape=True) * 105.0
  binary_threshed_img = color_thresh(warped_img, rgb_thresh=(r_navigable_thresh, g_navigable_thresh, b_navigable_thresh))

  # Navigable points
  navigable_rover_x, navigable_rover_y = rover_coords(binary_threshed_img)
  navigable_world_x, navigable_world_y = pix_to_world(navigable_rover_x, navigable_rover_y, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
  Rover.worldmap[navigable_world_y, navigable_world_x, 2] += 1  

  # Obstacle points
  obstacle_rover_x, obstacle_rover_y = rover_coords(np.invert(binary_threshed_img))
  obstacle_world_x, obstacle_world_y = pix_to_world(obstacle_rover_x, obstacle_rover_y, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
  Rover.worldmap[obstacle_world_y, obstacle_world_x, 0] += 0.5
  # Removes a bit of navigable
  Rover.worldmap[obstacle_world_y, obstacle_world_x, 2] -= 0.3

  # Calculate coordinates
  distances, angles = to_polar_coords(navigable_rover_x, navigable_rover_y)
  avg_angle = np.mean(angles)  

  # Calculate angles
  Rover.nav_dists = distances**2
  Rover.nav_angles = angles
  
  steer_angle = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
  navigable_distance = np.mean(distances)
  direction_line_angle = math.radians(steer_angle) - math.pi/2 
  
  draw_direction_line(img, direction_line_angle, navigable_distance)

  Rover.vision_image = img  
  return Rover

def draw_direction_line(img, angle, distance):
  # Calculates the final x,y point 
  line_length = 160 - distance 
  trans_x = 159 + line_length * math.cos(angle) * -1
  trans_y = line_length * math.sin(angle) * -1
  
  # Do nothing if calculations failed
  if np.isnan(trans_x) or np.isnan(trans_y):
    return

  rr, cc, val = draw.line_aa(159, 160, np.int(trans_y), np.int(trans_x))
  img[rr, cc, 0] = val * 250
  img[rr, cc, 1:2] = 0    


def process_rocks():
  pass
  # rock_thresholds = rock_threshold(warped_img) > 0
  # rock_center_of_mass = ndimage.measurements.center_of_mass(rock_thresholds) # TODO: handle multiple rocks  

  # img[:, :, 0] = (rock_thresholds * 255.0)

  # Navigable rocks
  # rock_rover_coords_x, rock_rover_coords_y = rover_coords(rock_thresholds)
  # rock_x_world, rock_y_world = pix_to_world(rock_rover_coords_x, rock_rover_coords_y, Rover.pos[0], Rover.pos[1], Rover.yaw, 200, scale) 

  # Draw circle on rocks  
  # rr, cc = draw.circle(rock_center_of_mass[0], rock_center_of_mass[1], 5)  
  # img[rr, cc, 2] = 200.0

  #   # Calculate slope between Rover and rock
  #   slope = (rock_center_of_mass[1] - 159) / (rock_center_of_mass[0] - 160)  
  #   angle_to_rock = math.tan(slope)
  #   print('slope', slope, math.tan(slope), angle_to_rock)
  

def color_thresh(img, rgb_thresh=(160, 160, 160), rgb_shape=False):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    
    # If this flag is on, returns an RGB dimension image, instead of single channel
    if rgb_shape:
        color_select = np.stack((color_select,)*3, axis=-1)
        
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped

def rock_threshold(img, rgb_thresh=(160, 140, 90), rgb_shape=False):

    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2]) # ATTENTION: reversed comparison for blue
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    
    # If this flag is on, returns an RGB dimension image, instead of single channel
    if rgb_shape:
        color_select = np.stack((color_select,)*3, axis=-1)
        
    return color_select