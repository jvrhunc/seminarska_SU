import numpy as np
import gym
from IPython.display import clear_output

def get_env():
    env = gym.make('Taxi-v3')
    return env

def value_iteration_algorithm(policy):
    
    env = get_env()
    
    timesteps, penalties, rewards = 0, 0, 0 # Za shranjevanje stevila korakov, kazni in nagrad
    
    frames = []  # Za shranjevanje vseh "okvirjev" oz. stanje kjer se bo taksi nahajal

    done = False # Pogoj, ce smo koncali eno epizodo

    state = env.reset() # Nastavimo prvotno stanje na poljubno
    
    while not done: # Delamo dokler nismo nasli pravilne poti
        
        action = np.argmax(policy[state]) # Izbira naslednje akcije glede na q_table
        
        state, reward, done, _ = env.step(action) # Evalvacije akcije za pridobivanje novega stanja, nagrade in pogoja za končanje
        
        rewards += reward # Sestevamo skupne neagrade
        
        if reward == -10: # Stejemo koliko kazni je dobil nas agent (narobe storjena akcija)
            penalties += 1
        
        # Stejemo koliko kazni je dobil nas agent (narobe storjena akcija)
        if reward == -10:
            penalties += 1

        frames.append({ # Vsak okvir damo v list
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            }
        )
        
        timesteps += 1 # Pristejem stevilo korakov
    
    return frames, timesteps, penalties, rewards
    
def value_iteration(theta=0.0001, discount_factor=1.0, max_iterations=10):
    
    env = get_env() # Vkljucimo nase okolje
    
    V = np.zeros(env.nS) # Kreiramo zacetno vrednost nase funkcijske vrednosti "V"
    i = 0
    while True:
        
        delta = 0 #Delto uporabimo za ustavitveni pogoj (ko bo manjsa kot nasa "theta")
        
        for s in range(env.nS): # Posodobimo vsako stanje
            
            A = one_step_lookahead(s, V, env, discount_factor) # Pogledamo en korak naprej, da najdemo najboljso mozno akcijo
            best_action_value = np.max(A) # Izberemo najboljso mozno akcijo iz "akcij naslednjega koraka"
            
            delta = max(delta, np.abs(best_action_value - V[s])) # Izracunamo delto za vsak stanja, ki smo jih do sedaj videli
            
            V[s] = best_action_value # Posodobimo naso funkcijsko vrednost "V"
        
        
        i += 1 # Ustavitveni pogoj, ce nasa funkcija ne konvergira
        if i > max_iterations:
            break
        
        if delta < theta: # Ustavitveni pogoj
            break
    
    policy = np.zeros([env.nS, env.nA]) # Create a deterministic policy using the optimal value function
    for s in range(env.nS):
        A = one_step_lookahead(s, V, env, discount_factor) # Pogledamo en korak naprej, da najdemo najboljso mozno akcijo
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0
    
    print("Policy for Value Iteration Algorithm created.\n")
    return policy, V

# Pomozna funkcija, ki izracuna vse vrednosti akcij v danem stanju
def one_step_lookahead(state, V, env, discount_factor):
    A = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[state][a]:
            A[a] += prob * (reward + discount_factor * V[next_state])
    return A