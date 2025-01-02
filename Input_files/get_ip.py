import subprocess

def get_private_ip():
    result = subprocess.run(['ip', 'addr'], stdout=subprocess.PIPE, text=True)
    for line in result.stdout.split('\n'):
        line = line.strip()
        if line.startswith('inet') and '127.0.0.1' not in line:
            ip = line.split()[1].split('/')[0]
            return ip
    return None

print(get_private_ip())