import socket
import select
import sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('127.0.0.1', 8000))

while True:
    read, write, fail = select.select((s,), (), ())

    for desc in read:
        if desc == s:
            data = s.recv(4096)
            print(data.decode())
        else:
            msg = desc.readline()
            s.send(msg.encode())
