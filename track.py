import cv2
import numpy as np
import pygame

class Track:
    def __init__(self, track_image_path='track.png', car_image_path='car.png', window_size=(800, 600)):
        self.window_size = window_size
        
        # Load the track image
        self.track_image = cv2.imread(track_image_path)
        if self.track_image is None:
            raise FileNotFoundError(f"Track image not found at {track_image_path}")
        
        # Load and resize the car image
        self.car_image = pygame.image.load(car_image_path)
        self.car_image = pygame.transform.scale(self.car_image, (32, 32))
        
        # Resize track image to fit the window
        self.track_image = cv2.resize(self.track_image, window_size)
        
        # Convert to grayscale and threshold to get a binary track
        self.gray = cv2.cvtColor(self.track_image, cv2.COLOR_BGR2GRAY)
        _, self.binary_track = cv2.threshold(self.gray, 127, 255, cv2.THRESH_BINARY)

        # Get track dimensions (should match the window size)
        self.height, self.width = self.binary_track.shape
        
        # Create Pygame surface for visualization
        self.surface = pygame.Surface(window_size)

        # Update the surface with the current track state
        self.update_surface()
        
        # Find start and finish positions
        self.start_pos = self._find_start_position()
        self.finish_pos = self._find_finish_position()
        
    def _find_start_position(self):
        """Find the starting position (green line)."""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(self.track_image, cv2.COLOR_BGR2HSV)
        # Green color range
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Find the first green pixel from top
        for y in range(self.height):
            for x in range(self.width):
                if green_mask[y, x] > 0:
                    return np.array([x, y])
        return np.array([0, 0])
    
    def _find_finish_position(self):
        """Find the finish position (red line)."""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(self.track_image, cv2.COLOR_BGR2HSV)
        # Red color range (red wraps around in HSV)
        lower_red1 = np.array([0, 40, 40])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 40, 40])
        upper_red2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Find the first red pixel from top
        for y in range(self.height):
            for x in range(self.width):
                if red_mask[y, x] > 0:
                    return np.array([x, y])
        return np.array([self.width-1, self.height-1])
    
    def update_surface(self):
        """Update the Pygame surface with the current track state."""
        # Ensure binary_track is uint8 (in case of data type issues)
        self.binary_track = self.binary_track.astype(np.uint8)

        # Convert the binary track to RGB (binary to color)
        track_rgb = cv2.cvtColor(self.binary_track, cv2.COLOR_GRAY2RGB)

        # Flip vertically (to match Pygame's coordinate system if needed)
        track_rgb = cv2.flip(track_rgb, 0)
        
        track_rgb = np.transpose(track_rgb, (1, 0, 2))

        # Ensure the surface is not locked (unlock if necessary)
        if self.surface.get_locked():
            self.surface.unlock()

        # Blit the RGB array onto the Pygame surface
        pygame.surfarray.blit_array(self.surface, track_rgb)
    
    def is_on_track(self, position):
        """Check if a position is on the track."""
        x, y = map(int, position)
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.binary_track[y, x] == 0
        return False

    def get_sensor_readings(self, position, angle, num_sensors=5):
        """Get sensor readings from the car's position."""
        readings = []
        sensor_angles = np.linspace(-np.pi/2, np.pi/2, num_sensors)
        
        for sensor_angle in sensor_angles:
            total_angle = angle + sensor_angle
            ray_length = 100  # Maximum ray length
            
            # Calculate ray end point (used for reference, but not in the loop)
            end_x = position[0] + ray_length * np.cos(total_angle)
            end_y = position[1] + ray_length * np.sin(total_angle)
            
            # Ensure current_pos is a float array
            current_pos = np.array(position, dtype=float)  # Explicitly use float
            
            step = 0.1  # Smaller step for more accurate detection
            
            # Ray-casting loop to find the boundary intersection
            while np.linalg.norm(current_pos - position) < ray_length:
                if not self.is_on_track(current_pos):
                    break
                # Move current position along the ray's direction
                current_pos += step * np.array([np.cos(total_angle), np.sin(total_angle)])
            
            # Calculate the actual distance to the boundary
            distance = np.linalg.norm(current_pos - position)
            readings.append(distance)
        
        return np.array(readings)
    
    def reset_car(self):
        """Reset car to starting position."""
        return self.start_pos.copy()
    
    def draw(self, screen, car_pos=None, car_angle=None):
        """Draw the track and optionally the car on the screen."""
        screen.blit(self.surface, (0, 0))
        
        # Draw start and finish lines
        pygame.draw.circle(screen, (0, 255, 0), (int(self.start_pos[0]), int(self.start_pos[1])), 5)
        pygame.draw.circle(screen, (255, 0, 0), (int(self.finish_pos[0]), int(self.finish_pos[1])), 5)
        
        if car_pos is not None and car_angle is not None:
            # Rotate the car image
            rotated_car = pygame.transform.rotate(self.car_image, -np.degrees(car_angle))
            
            # Get the rect of the rotated image
            car_rect = rotated_car.get_rect()
            
            # Set the center of the rect to the car's position
            car_rect.center = (int(car_pos[0]), int(car_pos[1]))
            
            # Draw the rotated car image
            screen.blit(rotated_car, car_rect) 