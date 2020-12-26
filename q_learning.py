import numpy as np
import gym
from IPython.display import clear_output
import random

def get_env():
    env = gym.make('Taxi-v3')
    return env

def q_learning(q_table):
    
    env = get_env() # Pridobimo okolje za igranje igre

    timesteps, penalties, rewards = 0, 0, 0 # Za shranjevanje stevila korakov, kazni in nagrad

    frames = [] # Za shranjevanje vseh "okvirjev" oz. stanje kjer se bo taksi nahajal

    done = False # Pogoj, ce smo koncali eno epizodo

    state = env.reset() # Nastavimo prvotno stanje na poljubno
    
    while not done: # Delamo dokler nismo nasli pravilne poti
        
        action = np.argmax(q_table[state]) # Izbira naslednje akcije glede na q_table

        state, reward, done, _ = env.step(action) # Evalvacije akcije za pridobivanje novega stanja, nagrade in pogoja za končanje
        
        rewards += reward # Sestevamo skupne neagrade
        
        if reward == -10: # Stejemo koliko kazni je dobil nas agent (narobe storjena akcija)
            penalties += 1

        frames.append({ # Vsak okvir damo v list
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'rewards': rewards
            }
        )
        
        timesteps += 1 # Pristejem stevilo korakov
    
    return frames, timesteps, penalties, rewards

def train_q_learning(n):
    
    env = get_env()
    
    q_table = np.zeros([env.nS, env.nA]) # Kreiramo q table velikosti (stevilo stanje) * (stevilo_akcij) 
    
    # Parametri za racunanje nove vrednosti v tabeli Q
    alpha = 0.1
    gamma = 1.0
    epsilon = 0.1
    
    for i in range(1, n+1):
        
        state = env.reset() # Izberemo poljubno zacetno stanje
        
        done = False # Pogoj ali se je ena epizoda ze koncala

        while not done:
            
            # Raziskujemo poljubno stanje
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            # Raziskujemo nauceno stanje iz Q tabele
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, _ = env.step(action) # Katero je naslednje stanje glede na podano akcijo
            
            old_value = q_table[state, action] # Vrednost v Q tabeli za prejsno stanje in akcijo
            next_max = np.max(q_table[next_state]) # Maksimalna vrednost v Q tabeli za naslednje stanje

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max) # Vrednost v Q tabeli izracunamo na podlagi podane formule
            q_table[state, action] = new_value
            
            state = next_state

        if i % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")
            
    print("Training finished.\n")
    return q_table