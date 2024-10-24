import socket
import struct

def handle_audio_data(data):
    # Process audio data
    print(f"Processing audio data")#: {data}")

def handle_animation_data(data):
    # Process animation data
    print(f"Processing animation data: {data.decode('ascii')}")

def start_server(host='localhost', port=12130):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        print(f"Server listening on {host}:{port}")

        while True:
            client_socket, client_address = server_socket.accept()
            with client_socket:
                print(f"Connection from {client_address}")

                while True:
                    # Read the length prefix (8 bytes)
                    length_prefix = client_socket.recv(8)
                    if not length_prefix:
                        print("Connection closed.")
                        break

                    # Unpack the length prefix
                    data_length = struct.unpack("!Q", length_prefix)[0]
                    
                    # Read the actual data based on the length prefix
                    data = b''
                    while len(data) < data_length:
                        chunk = client_socket.recv(data_length - len(data))
                        if not chunk:
                            print("Connection closed prematurely.")
                            break
                        data += chunk

                    if not data:
                        print("Connection closed.")
                        break

                    # Read the prefix (1 byte)
                    if len(data) < 1:
                        print("Received incomplete data.")
                        continue
                    
                    prefix = data[0:1]
                    payload = data[1:]

                    # Handle data based on prefix
                    if prefix == b'\x00':
                        print("Either header or eos data.")
                    elif prefix == b'\x01':
                        handle_audio_data(payload)
                    elif prefix == b'\x02':
                        handle_animation_data(payload)
                    else:
                        print("Unknown data type prefix received.")
                    
                    # If you expect to handle multiple messages in one connection
                    # You might want to decide if you continue to the next message
                    # or break if this is a one-time communication

if __name__ == "__main__":
    start_server()
