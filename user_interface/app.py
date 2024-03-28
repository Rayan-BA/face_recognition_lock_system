#graduatioation project: Enter Face
#semester 452
from flask import Flask, render_template, url_for, request, redirect,session,send_file,Response,flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from io import BytesIO
import cv2
from forms import UserForm, AccountForm
import bcrypt
from bcrypt import checkpw


app = Flask(__name__)
camera=cv2.VideoCapture(0)
app.secret_key = "graduate"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.permanent_session_lifetime = timedelta(hours=1) #keep session for one day

############################(DATABASE CODE)####################################
db = SQLAlchemy(app)
class users(db.Model):
    user_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(32))
    image = db.Column(db.LargeBinary)
    created_at = db.Column(db.DateTime, default=datetime.now)


    def __init__(self, name, image):
        self.name = name
        self.image = image

class account(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    password_hash = db.Column(db.String(128), nullable=False)

    def __init__(self, password):
        self.set_password(password)# use set_password to hash the password enterd by the user

    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))
class entries(db.Model):
    entrie_number = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.DateTime, default=datetime.now)
    accepted= db.Column(db.Boolean, default=True)
    user_id = db.Column( db.Integer, db.ForeignKey('users.user_id'))

    def __init__(self, user_id):
        self.user_id = user_id

############################(CAMERA CODE)####################################

def generate_frames():
    while True:
        success,frame=camera.read() #reads camera frame
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)#incode image into memory buffer
            frame=buffer.tobytes()#convert buffer to frames
        yield(b' -- frame\r\n'
                    b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n')#we use yield instade of return becuse return will end the loop
############################(APP CODE)####################################
@app.route("/")
def index():
    return redirect(url_for("login"))  # Go to login by default

@app.route("/login", methods=["POST","GET"])
def login():
    found_account = account.query.all()
    if not found_account:
        return redirect(url_for("register"))  # if there is no account go register
    elif "user" in session:
        return redirect(url_for("dashboard"))  # if user already in Session go to Dashboard
    elif request.method == "POST":
        form = AccountForm(request.form)
        if form.validate():  # to validate input
            password_input = request.form["password"]
            user_account = account.query.first()
            if user_account and checkpw(password_input.encode('utf-8'), user_account.password_hash.encode('utf-8')):
                session["user"] = "admin"  # save in session
                flash("you have logged-in successfuly")
                return redirect(url_for("dashboard"))  # if user was found go to Dashboard
            else:
                flash("wrong password", "error")
                return redirect(url_for("login"))
        else:
            # Flash all validation errors
            for field, errors in form.errors.items():
                for error in errors:
                    flash(f"{getattr(form, field).label.text}: {error}", "error")
            return redirect(url_for("login"))
    else:
        return render_template("login.html")

    
@app.route("/register", methods=["POST","GET"])
def register():
    found_account =  account.query.all()
    if found_account:
        return redirect(url_for("login")) #if there is account go login   
    elif "user" in session:
        return redirect(url_for("dashboard")) #if user already in Session go to Dashboared
    elif request.method == "POST":
        form = AccountForm(request.form)
        if form.validate():#to validate input
            password_input = request.form["password"]
            new_password = account(password_input)
            db.session.add(new_password)
            db.session.commit()
            session["user"]="admin" #save in session 
            flash("you have registered successfuly")
            return redirect(url_for("dashboard")) #if user was found go to Dashboared
        else:
            # Flash all validation errors
            for field, errors in form.errors.items():
                for error in errors:
                    flash(f"{getattr(form, field).label.text}: {error}", "error")
            return redirect(url_for("register"))
    else:
        return render_template("register.html")
    
@app.route("/dashboard")
def dashboard():
    if "user" in session: 
        return render_template("dashboard.html", num_of_users = users.query.count(), num_of_entries = entries.query.count(), num_of_alerts = entries.query.filter_by(accepted=False).count())
    else:
        return redirect(url_for("login"))

@app.route("/registerdUsers")
def registerdUsers():
    if "user" in session: 
        return render_template("registerdUsers.html",values = users.query.all())
    else:
        return redirect(url_for("login"))

@app.route("/history")
def history():
    if "user" in session:
        return render_template("history.html",values = entries.query.all())
    else:
        return redirect(url_for("login"))
    
@app.route("/newUser",methods=["POST","GET"])
def newUser():
    if "user" in session:
        if request.method == "POST":
            form = UserForm(request.form)
            if form.validate():#to validate input
                user = request.form["username"]
                file = request.files["file"]
                image= file.read()

                found_user =  users.query.filter_by(name=user).first() 
                if found_user:
                    flash("user already exists", "info")
                else:
                    new_user = users(user, image)
                    db.session.add(new_user)
                    db.session.commit() #if user not found then add new user to data base db
                    flash("user  have been added successfuly")
                
                return redirect(url_for("newUser"))
            else:
                # Flash all validation errors
                for field, errors in form.errors.items():
                    for error in errors:
                        flash(f"{getattr(form, field).label.text}: {error}", "error")
                return redirect(url_for("newUser"))
        else:
            return render_template("newUser.html") 
    else:
        return redirect(url_for("login"))
    
@app.route("/video") 
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/logout") #Delete username from Session then go to login
def logout():
    if "user" in session:
        flash("you have been logged-out successfuly")
    session.pop("user",None)
    return redirect(url_for("login"))

@app.route("/deleteUser", methods=["POST", "GET"]) #Delete username and password from db then go to registerdUserss
def deleteUser():
    if request.method == "POST":
        user = request.form["delete"]
        user_to_delete = users.query.filter_by(name=user).first()
        if user_to_delete:
            db.session.delete(user_to_delete)
            db.session.commit()
        return redirect(url_for("registerdUsers"))
    
@app.route('/download/<upload_id>') #to return image from data base
def download(upload_id):
    upload = users.query.filter_by(user_id=upload_id).first()
    return send_file(BytesIO(upload.image), mimetype='image/jpg')

@app.route("/updateSettings", methods=["POST", "GET"]) #update username and password 
def updateSettings():
    if "user" in session:
        if "theme" not in session:
            session['theme'] = "light" #light theme is default
        if request.method == "POST":
            new_theme = request.form["theme"]
            if new_theme != session['theme']: 
                return redirect(url_for("setTheme",set_theme=new_theme))
            form = AccountForm(request.form)
            if form.validate():#to validate input
                new_password = request.form["password"]
                check_password=account.query.filter_by(password=new_password).first() 
                if check_password:
                    flash("you cant use the same password","info")
                    return redirect(request.referrer)
                old_password=account.query.filter_by(id=1).first()
                old_password.password = new_password
                db.session.commit()
                flash("Password have been updated successfuly")
                session.pop("user",None)#exit session
                return redirect(url_for("login"))
            else:
                # Flash all validation errors
                for field, errors in form.errors.items():
                    for error in errors:
                        flash(f"{getattr(form, field).label.text}: {error}", "error")
                return redirect(request.referrer)
    else:
        return redirect(url_for("login"))
    
@app.route('/setTheme/<set_theme>')
def setTheme(set_theme):
    session['theme'] = set_theme 
    return redirect(request.referrer)
    
        
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)