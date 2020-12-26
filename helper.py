import gym
from IPython.display import clear_output
from time import sleep

def play_animation(frames, sleep_time):
        for i, frame in enumerate(frames):
            clear_output(wait=True)
            print(frame['frame'])
            print(f"Timestep: {i + 1}")
            print(f"State: {frame['state']}")
            print(f"Action: {frame['action']}")
            sleep(sleep_time)