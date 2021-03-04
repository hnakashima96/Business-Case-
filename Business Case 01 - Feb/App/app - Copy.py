import os
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load

PROJECT_ROOT = "C:/Users/putos/OneDrive/Documents/GitHub/Business-Case-"
#C:/Users/putos/OneDrive/Documents/GitHub/Business-Case-

app = Flask(__name__)  # Initialize the flask App
model = load(os.path.join(PROJECT_ROOT,  'best_decision_tree.joblib'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = list(request.form.values())[0].split(',')
    final_features = [np.array(int_features)]
    prediction = model.predict_proba(final_features)

    output = np.argmax(prediction[0])

    if output == 0:
        return render_template('index.html', prediction_text='This current customer has great chances to belong cluster: {}. This cluster is nominated as “Unusual Drinker”,\
                                                              those clients that are attracted by promotions and usually do online shops. They are composed by the younger people from\
                                                              de database and the lower income. The Sweetred, Sweetwh, Dessert, Exotic are the wine’s types that that group\
                                                              mostly buy.'.format(output))
    elif output ==1:
        return render_template('index.html', prediction_text='This current customer has great chances to belong cluster: {}. This cluster is nominated as “New Drinker”, are the\
                                                             clients who seems begin their wine journey. They are recently clients and usually visit the website and a\
                                                             great conversion rate, although promotions are not decision make to a purchase. They prefer Dryred wines and presents\
                                                             a relevant interest on accessories.'.format(output))
    elif output == 2:
        return render_template('index.html', prediction_text='This current customer has great chances to belong cluster: {}. This cluster is nominated as “Pro Drinkers”, They compose\
                                                             the elderly clients with the highest income. They hardly buy on internet neither visit the site. Also, this cluster do\
                                                             not present an interest on discounts. They are attracted by Dryred wines but seems to be interest on others\
                                                             type options. '.format(output))
    else:
        return render_template('index.html', prediction_text='This current customer has great chances to belong cluster: {}. This cluster is nominated as “Elite Drinkers”, has a high\
                                                             similarity to the cluster 1 (“New Drinker”). We nominated as Elite due to the difference with the cluster 1, \
                                                             which are senior clients.'.format(output))

if __name__ == "__main__":
    app.run(debug=True)