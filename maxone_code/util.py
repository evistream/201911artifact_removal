
SIZE_CHANNEL=1024
SAMPLING_FREQ=20000
RECORD_GAIN=512


import requests

def pushSlack(text='hello'):
    url = 'https://hooks.slack.com/services/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'
    headers = {'Content-type': 'application/json'}
    payload = {'text': text}
    r = requests.post(url, headers=headers, data=json.dumps(payload))