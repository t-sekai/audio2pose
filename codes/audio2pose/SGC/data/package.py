import socket
import struct

class SGCSocket():
    def __init__(self, remote_address: str, port: int) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (remote_address, port)
        print(f"connecting to {server_address[0]} port {server_address[1]}")
        self.socket.connect(server_address)

    def send_with_validation(self, raw_data, ascii_convert: bool, prefix: bytes = b'\x00'): #\x00 for header or eos, \x01 for audio, \x02 for anime, \x03 for text, \x04 for action
            assert len(prefix) == 1
            verify_size = struct.pack("!Q", len(prefix) + len(raw_data))
            send_data = verify_size + prefix
            if ascii_convert:
                send_data += bytes(raw_data, "ascii")
            else:
                send_data += raw_data
            try:
                self.socket.send(send_data)
            except (socket.error, BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
                 print(f"Socket erorr: {e}")
                 raise

    def send_eos(self):
        eos_symbol = f"EOS"
        self.send_with_validation(eos_symbol, True)

    def close(self):
         self.socket.close()