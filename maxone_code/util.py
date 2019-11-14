
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

