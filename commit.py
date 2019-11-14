import subprocess

def add():
    subprocess.call(['git', 'reset'])
    subprocess.call(['git', 'add', '.'])
    subprocess.call(['git', 'status'])


def commit(text=None):
    if text is None:
        text='.'
    subprocess.call(['git', 'commit', '-m', text])
    # subprocess.call(['git', 'push'])
    # subprocess.call(['ssh', 'n1', 'cd ~/Documents/sakurayama_workspace/201910; git pull'])
