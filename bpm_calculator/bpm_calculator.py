import random
from typing import List
import argparse

def next_bpm_calculator(bpm: float, n: int) -> float: 
    """
    A function that takes in a BPM and randomly decides whether to add or subtract from it
    """
    if random.random() > 0.5:
        return bpm + n
    return bpm - n

def run_bpm_calculator_n_times(starting_bpm: float, number_of_times: int) -> List[float]:
    bpm_list = [starting_bpm]
    for i in range(number_of_times):
        starting_bpm = next_bpm_calculator(starting_bpm, 5)
        bpm_list.append(starting_bpm)
    return bpm_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPM calculator script")
    parser.add_argument('--starting_bpm', type=int, help='A float argument for the starting bpm', default=120)
    parser.add_argument('--num_samples', type=int, help='An int argument for the number of songs to generate', default=10)
    args = parser.parse_args()
    bpm_list = run_bpm_calculator_n_times(args.starting_bpm, args.num_samples)
    print("List of BPMs", bpm_list)


    
