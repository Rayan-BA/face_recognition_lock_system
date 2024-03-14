#acount private info code
class account:
    def __init__(self):
        self.__private_username = "admin"
        self.__private_password = "admin"

    def get_username(self):
        return self.__private_username

    def update_account(self, username, password):
        self.__private_username = username
        self.__private_password = password

    def check_account(self, username, password):
        if self.__private_username == username:
            if self.__private_password == password:
                return True
        else:
            return False