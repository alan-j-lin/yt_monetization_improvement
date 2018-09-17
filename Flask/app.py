import os
from flask import Flask, render_template, jsonify, request, abort
from api import get_classification

# Initialization of the Video Classification App
app = Flask('YTextension')

# Post method for video classification
@app.route('/classify', methods = ['POST'])
def classify_video():

    # check to see if what is being passed is a JSON
    if not request.json:
        abort(400)
    # check to see that the necessary entries are in the JSON
    elif 'url' not in request.json:
        abort(400)

    # using get_classification function from api to return video label
    else:
        data = request.json
        response = get_classification(data)
        return jsonify(response)

        # return jsonify(response)



# Running Video Classification App
if __name__ == '__main__':
  port = int(os.environ.get('PORT', 5000))
  app.run(host='0.0.0.0', port=port, debug=True)
