import numpy as np
import gym
from IPython.display import clear_output

def get_env():
    env = gym.make('Taxi-v3')
    return env

def policy_iteration_algorithm(policy):
    
    env = get_env()
    
    timesteps, penalties, rewards = 0, 0, 0 # Za shranjevanje stevila korakov, kazni in nagrad
    
    frames = []  # Za shranjevanje vseh "okvirjev" oz. stanje kjer se bo taksi nahajal

    done = False # Pogoj, ce smo koncali eno epizodo

    state = env.reset() # Nastavimo prvotno stanje na poljubno
    
    while not done: # Delamo dokler nismo nasli pravilne poti
        
        # Izbira naslednje akcije glede na q_table
        action = np.argmax(policy[state])
        
        # Evalvacije akcije za pridobivanje novega stanja, nagrade in pogoja za končanje
        state, reward, done, _ = env.step(action)
        
        # Sestevamo skupne neagrade
        rewards += reward
        
        if reward == -10: # Stejemo koliko kazni je dobil nas agent (narobe storjena akcija)
            penalties += 1
        
        # Stejemo koliko kazni je dobil nas agent (narobe storjena akcija)
        if reward == -10:
            penalties += 1

        # Vsak okvir damo v list
        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'rewards': rewards
            }
        )
        
        # Pristejem stevilo korakov
        timesteps += 1
    
    return frames, timesteps, penalties, rewards

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001, max_iterations=10):
    
    V = np.zeros(env.nS) # Inicializiramo naso funkcijsko vrednost "V"
    
    i = 0
    while True: # Dokler je vrednost "delta" manjsa od vrednosti "theta"
        
        delta = 0 # Sledenje posodobitvam v probramu
        
        for s in range(env.nS): # Za vsako stanje pogledamo naprej en korak v vse mozne akcije in naslednjo stanje
            v = 0
            
            for a, action_prob in enumerate(policy[s]): 
                 
                for prob, next_state, reward, done in env.P[s][a]:
                    # Izracunamo vrednost na podalig formule
                    v += action_prob * prob * (reward + discount_factor * V[next_state]) # P[s, a, s']*(R(s,a,s')+γV[s'])
                    
            delta = max(delta, np.abs(v - V[s])) # Pogledamo koliko se je vrednost spremenila
            V[s] = v # Posodobimo trenutno stanje
        
        i += 1
        if i > max_iterations: # Ce nasa funkcija ne konvergira jo ustavimo z maksimalnim stevilom iteracij
            break
    
        if delta < theta: # Ustavitveni pogoj
            break
    return np.array(V)

def policy_iteration(policy_eval_fn=policy_eval, discount_factor=1.0):
    
    env = get_env()
    
    policy = np.ones([env.nS, env.nA]) / env.nA # Nastavimo nasto prvotno politiko
    
    while True:
        V = policy_eval_fn(policy, env, discount_factor) # Zracunamo funkcijsko vrednost "V" glede na trenutno politiko
        
        policy_stable = True # Ce bomo kasneje politiko posodobili ga bomo postavili na False
        
        for s in range(env.nS): # Posodobimo politiko za vsako stanje
            chosen_a = np.argmax(policy[s]) # Izberemo najboljso akcijo trenutne politike
            
            action_values = np.zeros(env.nA) # Najdemo najboljso akcijo glede za en korak naprej
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            best_a = np.argmax(action_values)
            
            if chosen_a != best_a: # Ce smo posodobili politiko
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]
        
        if policy_stable: # Ko politike nismo vec spremnijali
            print("Policy for Policy Iteration Algorithm created.\n")
            return policy, V