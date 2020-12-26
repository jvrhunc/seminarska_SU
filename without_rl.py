import numpy as np
import gym
from IPython.display import clear_output

def get_env():
    env = gym.make('Taxi-v3')
    return env

def without_reinforcement_learning():
    
    env = get_env() # Pridobimo okolje za igranje igre

    timesteps, penalties, rewards = 0, 0, 0 # Za shranjevanje stevila korakov, kazni in nagrad

    frames = [] # Za shranjevanje vseh "okvirjev" oz. stanje kjer se bo taksi nahajal

    done = False # Pogoj, ce smo koncali eno "epizodo"

    state = env.reset() # Nastavimo prvotno stanje na poljubno
    
    while not done: # Delamo dokler nismo nasli pravilne poti
        
        action = env.action_space.sample() # Izbira naslednje akcije
        
        state, reward, done, _ = env.step(action) # Evalvacije akcije za pridobivanje novega stanja, nagrade in pogoja za končanje
        
        rewards += reward # Sestevamo skupne neagrade
            
        if reward == -10: # Stejemo koliko kazni je dobil nas agent (narobe storjena akcija)
            penalties += 1
        
        frames.append({ # Vsak okvir damo v list "frames"
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'rewards': rewards
            }
        )
        
        timesteps += 1 # Pristejem stevilo korakov
        
    return frames, timesteps, penalties, rewards