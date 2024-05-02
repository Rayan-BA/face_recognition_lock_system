import paramiko
from os import listdir

import paramiko.ssh_exception

class RPi:
    def __init__(self, hostname = "raspberrypi", username = "pi") -> None:
        self.hostname = hostname
        self.username = username
        self.remote_working_dir = "ef-access-control"
        self.ssh = paramiko.SSHClient()

    def connect(self):
        try:
            self.ssh.load_host_keys("C:\\Users\\rbalh\\.ssh\\known_hosts")
        except paramiko.ssh_exception.SSHException:
            with open("known_hosts", "w") as file:
                file.write(f"{self.username}@{self.hostname}")

        print(f"connecting to {self.username}@{self.hostname}...")
        self.ssh.connect(self.hostname, username=self.username)
        self.ssh.exec_command(f"cd ef-access-control;pwd")
        self.sftp = self.ssh.open_sftp()
        self.sftp.chdir(self.remote_working_dir)
        print("connected.")
    
    def close(self):
        self.ssh.close()
        self.sftp.close()
        print("connection closed.")
    
    def send(self, local_dir):
        try:
            for file in listdir(local_dir):
                print(file)
                self.sftp.put(localpath=f"{local_dir}/{file}", remotepath=file)
        except Exception as e:
            print(e)

    def receive(self, local_dir):
        # get log file at fixed time intervals?
        # if there is a way for remote to trigger and event on local then do this
        self.sftp.get(remotepath="log.txt", localpath=local_dir)
    
    def is_connected(self):
        if self.ssh.get_transport() is not None:
            return self.ssh.get_transport().is_active()
        else: return False


if __name__ == "__main__":
    s = RPi()
    s.connect()
    s.send("models")
    s.receive("./log.txt")
    s.close()
    