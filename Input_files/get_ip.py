import subprocess

# def get_private_ip():
#     result = subprocess.run(['ip', 'addr'], stdout=subprocess.PIPE, text=True)
#     for line in result.stdout.split('\n'):
#         line = line.strip()
#         if line.startswith('inet') and '127.0.0.1' not in line:
#             ip = line.split()[1].split('/')[0]
#             return ip
#     return None

import socket

def get_private_ip():
    try:
        # Create a socket and connect to an external address
        # Use an address like 8.8.8.8 (Google's public DNS) to determine the network interface
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            private_ip = s.getsockname()[0]
        return private_ip
    except Exception as e:
        return f"Error occurred: {e}"

print(get_private_ip())