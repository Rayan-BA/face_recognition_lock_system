#graduation project: Enter Face
#semester 452
from flask import Flask, render_template, url_for, request, redirect, session, send_file, Response, flash
from flask_cors import CORS, cross_origin
from flask_apscheduler import APScheduler
from datetime import timedelta
from io import BytesIO
from forms import UserForm, AccountForm, AccountFormUpdate
from bcrypt import checkpw
from db import *
from FaceEmbeddingGenerator import FaceEmbeddingGenerator
from SVC import mySVC
import base64, shutil, json
from rpi import RPi
from pathlib import Path

app = Flask(__name__)
cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True
db.init_app(app)
app.secret_key = "graduate"
app.permanent_session_lifetime = timedelta(hours=3) #keep session for one day

scheduler = APScheduler() # run periodic RPI.stat() to check for new entrie attempts
scheduler.api_enabled = True
scheduler.init_app(app)
scheduler.start()

@app.route("/")
def index():
    return redirect(url_for("login"))  # Go to login by default

@app.route("/login", methods=["POST","GET"])
def login():
    found_account = Account.query.all()
    if not found_account:
        return redirect(url_for("register"))  # if there is no account go register
    elif "user" in session:
        return redirect(url_for("dashboard"))  # if user already in Session go to Dashboard
    elif request.method == "POST":
        form = AccountForm(request.form)
        if form.validate():  # to validate input
            password_input = request.form["password"]
            user_account = Account.query.first()
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
    found_account =  Account.query.all()
    if found_account:
        return redirect(url_for("login")) #if there is account go login   
    elif "user" in session:
        return redirect(url_for("dashboard")) #if user already in Session go to Dashboared
    elif request.method == "POST":
        form = AccountForm(request.form)
        if form.validate():#to validate input
            password_input = request.form["password"]
            new_password = Account(password_input)
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
        return render_template("dashboard.html", num_of_users = Users.query.count(), num_of_entries = Entries.query.count(), num_of_alerts = Entries.query.filter_by(accepted=False).count())
    else:
        return redirect(url_for("login"))

@app.route("/registerdUsers")
def registerdUsers():
    if "user" in session: 
        return render_template("registerdUsers.html", values = Users.query.all())
    else:
        return redirect(url_for("login"))

@app.route("/history")
def history():
    if "user" in session:
        return render_template("history.html", values = Entries.query.order_by(Entries.time.desc()).all())
    else:
        return redirect(url_for("login"))

@app.route("/entryImage/<entry_id>")
def entryImage(entry_id):
    upload = Entries.query.filter_by(entry_id=entry_id).first()
    return send_file(BytesIO(upload.image), mimetype='image/jpg')

@app.route("/newUser",methods=["POST","GET"])
@cross_origin()
def newUser():
    if "user" in session:
        if request.method == "POST":
            values = []
            for value in request.form.values():
                values.append(value)
            images = values[1:]

            form = UserForm(request.form)
            if form.validate(): #to validate input
                username = request.form["username"]
                found_user =  Users.query.filter_by(name=username).first() 
                if found_user:
                    flash("user already exists", "info")
                    print("user already exists")
                else:
                    for i, img in enumerate(images):
                        try:
                            Path(f"tmp/{username}").mkdir(parents=True, exist_ok=True)

                            with open(f"tmp/{username}/{i}.jpg", "wb") as file:
                                file.write(base64.b64decode(img))
                        except Exception as e:
                            print(e)
                    
                    try:
                        FaceEmbeddingGenerator(dataset="./tmp").create_embeddings()
                        mySVC().train()

                        ssh = RPi()
                        ssh.connect()
                        ssh.send("./models")
                        ssh.close()
                    
                        with open(f"tmp/{username}/0.jpg", "rb") as file:
                            image = file.read()
                            new_user = Users(username, image)
                            db.session.add(new_user)
                            db.session.commit() #if user not found then add new user to data base db
                            flash("user  have been added successfuly")
                    except Exception as e:
                        print(e)
                    
                    
                
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
        user_to_delete = Users.query.filter_by(name=user).first()
        if user_to_delete:
            FaceEmbeddingGenerator(dataset="./tmp").remove_embed(user)
            shutil.rmtree(Path(f"tmp/{user}"))
            db.session.delete(user_to_delete)
            db.session.commit()
        return redirect(url_for("registerdUsers"))
    
@app.route('/download/<upload_id>') #to return image from data base
def download(upload_id):
    upload = Users.query.filter_by(user_id=upload_id).first()
    return send_file(BytesIO(upload.image), mimetype='image/jpg')

@app.route("/updateSettings", methods=["POST", "GET"]) #update username and password 
def updateSettings():
    if "user" in session:
            form = AccountFormUpdate(request.form)
            if form.validate():#to validate input
                current_password= request.form["current_password"]
                user_account = Account.query.first()
                if user_account and checkpw(current_password.encode('utf-8'), user_account.password_hash.encode('utf-8')):
                    new_password_input = request.form["new_password"]
                    new_password = Account(new_password_input)
                    if user_account and checkpw(new_password_input.encode('utf-8'), user_account.password_hash.encode('utf-8')):
                        flash("you cant use the same password","info")
                        return redirect(request.referrer)
                    else:
                        db.session.delete(user_account)
                        db.session.add(new_password)
                        db.session.commit()
                        flash("Password have been updated successfuly")
                        session.pop("user",None)#exit session
                        return redirect(url_for("login"))
                else:
                    flash("invalid current password","error")
                    return redirect(request.referrer)
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

@scheduler.task(trigger="interval", id="entries", seconds=30, next_run_time=datetime.datetime.now())
def updateEntries():
    ssh = RPi()
    ssh.connect()
    if not ssh.is_connected():
        print("RPI is not connected.")
        # return
    entries_file = "entries.json"
    with app.app_context():
        stat = ssh.stat(entries_file)
        if stat:
            current_file_size = stat.st_size
        else:
            current_file_size = ""
        
        if not Path("entries_prev_size.txt").exists():
            with open("entries_prev_size.txt", "w") as file:
                file.write(current_file_size)
        with open("entries_prev_size.txt", "r") as file:
            prev_file_size = file.read()
        
        print(prev_file_size, current_file_size)
        
        if int(prev_file_size) != int(current_file_size) and current_file_size != "":
            ssh.receive()
            if Path(entries_file).exists():
                with open(entries_file, "r") as file:
                    entries = json.load(file)
                    for entry in entries:
                        id = int(entry["id"])
                        try:
                            saved_entry = Entries.query.filter_by(entry_id=id).first()
                        except Exception as e:
                            print(e)
                            saved_entry = None
                        if saved_entry is not None:
                            print("skipping")
                            continue
                        try:
                            image = base64.b64decode(entry["image"])
                        except Exception as e:
                            print(e)
                            continue
                        time = datetime.datetime.strptime(entry["time"], "%Y-%m-%d %H:%M:%S.%f%z")
                        new_entry = Entries(entry_id=id, name=entry["name"], time=time, accepted=entry["accepted"], image=image, reject_reason=entry["reject_reason"])
                        db.session.add(new_entry)
                        db.session.commit()
                with open("entries_prev_size.txt", "w") as file:
                    file.write(str(current_file_size))
                Path.unlink(entries_file)
        
    ssh.close()
        
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, use_reloader=False)