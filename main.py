import pygame
import sys
import time
from track import Track
from car import Car
from ai_agent import DQNAgent

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = (800, 600)
FPS = 60
BATCH_SIZE = 32
EPISODES = 1000
SAVE_INTERVAL = 100
MAX_TIME = 60  # Maximum time allowed for each episode

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

def calculate_time_reward(time_taken):
    """Calculate reward based on time taken to reach finish."""
    if time_taken > MAX_TIME:
        return -10  # Penalty for taking too long
    return 100 - time_taken  # Higher reward for faster times

def main():
    # Create window
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("AI Car Racing")
    clock = pygame.time.Clock()
    
    # Initialize environment
    track = Track(window_size=WINDOW_SIZE)
    car = Car(track)
    
    # Initialize AI agent
    state_size = len(car.get_state())
    action_size = 3  # forward, left, right
    agent = DQNAgent(state_size, action_size)
    
    # Training variables
    best_time = float('inf')
    episode_rewards = []
    
    # Main training loop
    for episode in range(EPISODES):
        car.reset()
        state = car.get_state()
        total_reward = 0
        start_time = time.time()
        finished = False
        
        while True:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
            
            # Get action from agent
            action = agent.act(state)
            
            # Take action and get new state
            next_state, reward, done = car.step(action)
            total_reward += reward
            
            # Store experience in memory
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            agent.replay(BATCH_SIZE)
            
            # Update state
            state = next_state
            
            # Check if episode is finished
            if done:
                time_taken = time.time() - start_time
                # Add time-based reward if finished successfully
                if not car.crashed:
                    time_reward = calculate_time_reward(time_taken)
                    total_reward += time_reward
                    if time_taken < best_time:
                        best_time = time_taken
                        agent.save('best_model.pth')
                finished = not car.crashed
                break
            
            # Draw everything
            screen.fill(WHITE)
            track.draw(screen, car.position, car.angle)
            
            # Draw UI
            font = pygame.font.Font(None, 20)
            episode_text = font.render(f'Episode: {episode + 1}/{EPISODES}', True, BLACK)
            reward_text = font.render(f'Reward: {total_reward:.2f}', True, BLACK)
            epsilon_text = font.render(f'Epsilon: {agent.epsilon:.2f}', True, BLACK)
            best_time_text = font.render(f'Best Time: {best_time:.2f}s', True, BLACK)
            distance_text = font.render(f'Distance: {car.distance_traveled:.1f}', True, BLACK)
            current_time = time.time() - start_time
            time_text = font.render(f'Time: {current_time:.1f}s', True, BLACK)
            
            screen.blit(episode_text, (10, 10))
            screen.blit(reward_text, (140, 10))
            screen.blit(epsilon_text, (260, 10))
            screen.blit(best_time_text, (380, 10))
            screen.blit(distance_text, (520, 10))
            screen.blit(time_text, (650, 10))
            
            pygame.display.flip()
            clock.tick(FPS)
        
        # Episode finished
        episode_rewards.append(total_reward)
        
        # Save model periodically
        if (episode + 1) % SAVE_INTERVAL == 0:
            agent.save(f'model_episode_{episode + 1}.pth')
        
        # Print episode summary
        print(f"Episode: {episode + 1}/{EPISODES}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Finished: {finished}")
        if finished:
            print(f"Time: {time_taken:.2f}s")
        print(f"Best Time: {best_time:.2f}s")
        print(f"Distance Traveled: {car.distance_traveled:.1f}")
        print(f"Epsilon: {agent.epsilon:.2f}")
        print("-" * 50)

if __name__ == "__main__":
    main() 