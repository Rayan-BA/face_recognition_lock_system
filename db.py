from flask_sqlalchemy import SQLAlchemy
import datetime, bcrypt

db = SQLAlchemy()

class Users(db.Model):
    def __init__(self, name, image):
        self.name = name
        self.image = image
    
    user_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(32), unique=True)
    image = db.Column(db.LargeBinary)
    created_at = db.Column(db.DateTime, default=datetime.datetime.now(datetime.UTC))

class Account(db.Model):
    def __init__(self, password):
        self.set_password(password)
    
    id = db.Column(db.Integer, primary_key=True)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

class Entries(db.Model):
    def __init__(self, entry_id, name, time , accepted, image, reject_reason):
        self.entry_id = entry_id
        self.name = name
        self.time = time
        self.accepted = accepted
        self.image = image
        self.reject_reason = reject_reason
    
    entry_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(32))
    time = db.Column(db.DateTime)
    accepted = db.Column(db.Boolean)
    image = db.Column(db.LargeBinary)
    reject_reason = db.Column(db.String(32))
    # user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    # user = db.relationship('Users', backref=db.backref('entries', lazy=True))
