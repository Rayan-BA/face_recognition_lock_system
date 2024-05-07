import paramiko
from os import listdir

import paramiko.ssh_exception

class RPi:
    def __init__(self, hostname = "raspberrypi", username = "pi") -> None:
        self.hostname = hostname
        self.username = username
        self.known_hosts = "C:\\Users\\rbalh\\.ssh\\known_hosts"
        self.remote_working_dir = "ef-access-control"
        self.ssh = paramiko.SSHClient()

    def connect(self):
        try:
            self.ssh.load_host_keys(self.known_hosts)
        except paramiko.ssh_exception.SSHException:
            with open("known_hosts", "w") as file:
                file.write(f"{self.username}@{self.hostname}")

        try:
            print(f"connecting to {self.username}@{self.hostname}...")
            self.ssh.connect(self.hostname, username=self.username)
            self.ssh.exec_command(f"cd ef-access-control;pwd")
            self.sftp = self.ssh.open_sftp()
            self.sftp.chdir(self.remote_working_dir)
            print("connected.")
        except Exception as e:
            print(e)
    
    def close(self):
        try:
            self.ssh.close()
            self.sftp.close()
            print("connection closed.")
        except Exception as e:
            print(e)
    
    def send(self, local_dir):
        print("sending...")
        try:
            for file in listdir(local_dir):
                self.sftp.put(localpath=f"{local_dir}/{file}", remotepath=file)
        except Exception as e:
            print(e)

    def receive(self, local_dir="./entries.json"):
        print("receiving...")
        try:
            self.sftp.get(remotepath="entries.json", localpath=local_dir)
        except Exception as e:
            print(e)
    
    def is_connected(self):
        if self.ssh.get_transport() is not None:
            return self.ssh.get_transport().is_active()
        else: return False
    
    def stat(self, file):
        try:
            return self.sftp.stat(file)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    s = RPi()
    s.connect()
    # s.send("models")
    # s.receive("./entries.json")
    print(s.stat("entries.json").st_size) # if size changes receive updated files
    s.close()
    