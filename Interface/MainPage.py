from flask import Flask, request, redirect, url_for
from flask import render_template
import os
from werkzeug import secure_filename
import sai
import base64
UPLOAD_FOLDER = 'upload/'
ALLOWED_EXTENSIONS = set(['mp3','txt'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
@app.route('/analysis/<filename>')
def analysis(filename):
	opendata=sai.Opendata(filename)
	speaker1,speaker2,other=sai.SeparateSpeakers(opendata)
	countspeak1,FreqPlot1,countspeak2,FreqPlot2,Adjadv1,Adjadv2,Aggresult1,Aggresult2,SentSpeaker1,SentSpeaker2,Speak1Result,Sentplot1,Speak2Result,Sentplot2=sai.PreprocessData(speaker1,speaker2,other)
	return render_template("test.html",countspeak1=countspeak1,freqplot1=FreqPlot1,adjadv1=Adjadv1,aggresult=Aggresult1,sentspeak1=SentSpeaker1,sentplot=Sentplot1,countspeak2=countspeak2,freqplot2=FreqPlot2,adjadv2=Adjadv2,aggresult2=Aggresult2,sentspeak2=SentSpeaker2,sentplot2=Sentplot2)

@app.route("/upload",methods=['GET','POST'])
def upload():
	if request.method == 'POST':
		file = request.files['file']
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	return redirect(url_for('analysis',filename=filename))

@app.route("/", methods=['GET', 'POST'])
def index():
	return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)