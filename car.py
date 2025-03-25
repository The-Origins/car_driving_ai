import numpy as np
import time

class Car:
    def __init__(self, track):
        self.track = track
        self.best_time = float('inf')  # Initialize best time as infinity
        self.start_time = 0  # Track when the episode starts
        self.reset()
        
    def reset(self):
        """Reset car to starting position."""
        self.position = self.track.reset_car()
        self.angle = 0  # Initial angle (in radians)
        self.speed = 0
        self.max_speed = 5
        self.acceleration = 0.2
        self.deceleration = 0.1
        self.turn_speed = 0.1
        self.crashed = False 
        self.distance_traveled = 0  # Track total distance traveled
        self.last_position = self.position.copy()  # For calculating distance
        self.start_time = time.time()  # Reset start time
        
    def get_state(self):
        """Get the current state of the car."""
        sensor_readings = self.track.get_sensor_readings(self.position, self.angle)
        return np.concatenate([
            sensor_readings,
            [self.speed],
            [np.sin(self.angle), np.cos(self.angle)]
        ])
    
    def step(self, action):
        """Execute one step of car movement based on the action."""
        # Store previous position for distance calculation
        self.last_position = self.position.copy()
        
        # Action: 0 = forward, 1 = left, 2 = right
        if action == 1:  # Turn left
            self.angle -= self.turn_speed
        elif action == 2:  # Turn right
            self.angle += self.turn_speed
            
        # Update speed based on action
        if action == 0:  # Forward
            self.speed = min(self.speed + self.acceleration, self.max_speed)
        else:
            self.speed = max(self.speed - self.deceleration, 0)
            
        # Update position
        self.position = np.array(self.position, dtype=float)
        self.position += self.speed * np.array([np.cos(self.angle), np.sin(self.angle)])
        
        # Calculate distance traveled
        distance_moved = np.linalg.norm(self.position - self.last_position)
        self.distance_traveled += distance_moved
        
        # Check if car is out of bounds
        if (self.position[0] < 0 or self.position[0] >= self.track.width or 
            self.position[1] < 0 or self.position[1] >= self.track.height):
            self.crashed = True
            # Calculate a severe negative reward that's proportional to the distance traveled
            # This ensures the car can't accumulate positive rewards by going out of bounds
            crash_reward = -50 - (self.distance_traveled * 0.1)  # Base penalty plus distance penalty
            return self.get_state(), crash_reward, True
        
        # Check if car is on track
        is_on_track = self.track.is_on_track(self.position)
        
        # Calculate reward
        reward = self._calculate_reward(is_on_track)
        
        # Check if finish line is reached
        done = self._check_finish_line() or self.crashed
        
        return self.get_state(), reward, done
    
    def _calculate_reward(self, is_on_track):
        """Calculate reward based on current state."""
        reward = 0
        
        # Base reward for staying on track (smaller now)
        if is_on_track:
            reward += 0.1
        else:
            reward -= 10
            
        # Reward for distance traveled (scaled by speed)
        if is_on_track:
            reward += self.speed * 0.5  # More reward for faster speed
            
        # Penalty for being too close to walls
        sensor_readings = self.track.get_sensor_readings(self.position, self.angle)
        min_distance = np.min(sensor_readings)
        if min_distance < 10:  # If any sensor is too close to a wall
            reward -= (10 - min_distance) * 0.5  # Penalty increases as distance decreases
            
        # Add penalty for exceeding best time
        current_time = time.time() - self.start_time
        if self.best_time != float('inf') and current_time > self.best_time:
            time_excess = current_time - self.best_time
            reward -= time_excess * 2  # Penalty of 2 points per second over best time
            
        return reward
    
    def _check_finish_line(self):
        """Check if the car has reached the finish line."""
        # Check if car is close enough to finish line
        distance_to_finish = np.linalg.norm(self.position - self.track.finish_pos)
        if distance_to_finish < 40:  # Within 20 pixels of finish line
            # Update best time if this is a new best
            current_time = time.time() - self.start_time
            if current_time < self.best_time:
                self.best_time = current_time
            return True
        return False 