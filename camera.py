import cv2

class Camera:
    def __init__(self):
        self.vid_cap = cv2.VideoCapture(0)
        self.toggle = True

    def generate_frames(self):
        while True:
            success, frame = self.vid_cap.read() #reads camera frame
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame) #incode image into memory buffer
                frame = buffer.tobytes() #convert buffer to frames
            yield(b' -- frame\r\n'
                        b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n') #we use yield instade of return becuse return will end the loop
    
    def toggle_vid(self):
        self.toggle = not self.toggle
    
    def release(self):
        self.vid_cap.release()