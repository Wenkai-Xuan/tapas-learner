import matplotlib.pyplot as plt
import os
import json

freq = []

path = '/home/tapasdeveloper/build_playground/tapas-learner_3/tapas-learner/multi-agent-tamp-solver/24-data-gen/out/sequence_plan_20241208_223814_conveyor'

for (root,dirs,files) in os.walk(path, topdown=True):
    for file in files:
        if file.startswith('metadata.json'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r') as f:
                data = json.load(f)
                # print(data['metadata']['makespan'])
                freq.append(data['metadata']['makespan'])

# random old plan length: (186+1)
freq_random = [x + 0 for x in freq]

# print(freq)
plt.hist(freq_random, bins=len(freq_random))

plt.show()