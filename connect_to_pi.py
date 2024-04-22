import paramiko

class Server:
    def __init__(self) -> None:
        self.client = paramiko.SSHClient()

    def connect(self):
        self.client.load_system_host_keys()

        print("connecting to server")
        self.client.connect("raspberrypi", username="pi", key_filename="C:\\Users\\rbalh\\.ssh\\id_rsa")
        print("connected")

    def close_conn(self):
        self.client.close()

if __name__ == "__main__":
    s = Server()
    s.connect()
    s.close_conn()
    