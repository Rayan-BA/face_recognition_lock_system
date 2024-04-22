from flask_sqlalchemy import SQLAlchemy
import datetime
import bcrypt

db = SQLAlchemy()

class Users(db.Model):
    user_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(32))
    image = db.Column(db.LargeBinary)
    created_at = db.Column(db.DateTime, default=datetime.datetime.now(datetime.UTC))

    def __init__(self, name, image):
        self.name = name
        self.image = image

class Account(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    password_hash = db.Column(db.String(128), nullable=False)

    def __init__(self, password):
        self.set_password(password)

    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

class Entries(db.Model):
    entry_number = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.DateTime, default=datetime.datetime.now(datetime.UTC))
    accepted = db.Column(db.Boolean, default=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    user = db.relationship('Users', backref=db.backref('entries', lazy=True))

    def __init__(self, user_id):
        self.user_id = user_id
    