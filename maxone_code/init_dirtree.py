import os

TEMP_DIRS=[
    'config/{}',
    'data/{}/scan',
    'data/{}/selectelectrode/raw',
    'data/{}/selectelectrode/spike',
    'data/{}/raw',
    'data/{}/spike',
    'data/{}/dataset',
]

def makedirs_(path):
    if not os.path.exists(path):
        os.makedirs(path)

def render(exps):
    for exp in exps:
        for dr in TEMP_DIRS:
            path = dr.format(exp)
            makedirs_(path)