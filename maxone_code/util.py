import math

SIZE_CHANNEL=1024
SAMPLING_FREQ=20000
RECORD_GAIN=512


import requests

def pushSlack(text='hello'):
    url = 'https://hooks.slack.com/services/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'
    headers = {'Content-type': 'application/json'}
    payload = {'text': text}
    r = requests.post(url, headers=headers, data=json.dumps(payload))

def elecid2posi(elecid, is_uM=False):
    y,x = divmod(elecid,220)
    if is_uM is True:
        x *= 17.5
        y *= 17.5
    return y,x

def posi2elecid(y,x, is_uM=False):
    if is_uM is True:
        x = round(x /17.5)
        y = round(y /17.5)
    return x+y*220

def distElecId(id1,id2, is_uM=False):
    x1,y1=elecid2posi(id1)
    x2,y2=elecid2posi(id2)
    dist=math.sqrt((x1-x2)**2+(y1-y2)**2)
    if is_uM is True:
        dist *= 17.5
    return dist