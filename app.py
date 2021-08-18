{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "# 1. Library imports\n",
    "import uvicorn\n",
    "from fastapi import FastAPI\n",
    "from BankNotes import BankNote\n",
    "import numpy as np\n",
    "import pickle\n",
    "import pandas as pd\n",
    "# 2. Create the app object\n",
    "app = FastAPI()\n",
    "pickle_in = open(\"classifier.pkl\",\"rb\")\n",
    "classifier=pickle.load(pickle_in)\n",
    "\n",
    "# 3. Index route, opens automatically on http://127.0.0.1:8000\n",
    "@app.get('/')\n",
    "def index():\n",
    "    return {'message': 'Hello, World'}\n",
    "\n",
    "# 4. Route with a single parameter, returns the parameter within a message\n",
    "#    Located at: http://127.0.0.1:8000/AnyNameHere\n",
    "@app.get('/{name}')\n",
    "def get_name(name: str):\n",
    "    return {'Welcome': f'{name}'}\n",
    "\n",
    "# 3. Expose the prediction functionality, make a prediction from the passed\n",
    "#    JSON data and return the predicted Bank Note with the confidence\n",
    "@app.post('/predict')\n",
    "def predict_banknote(data:BankNote):\n",
    "    data = data.dict()\n",
    "    variance=data['variance']\n",
    "    skewness=data['skewness']\n",
    "    curtosis=data['curtosis']\n",
    "    entropy=data['entropy']\n",
    "   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))\n",
    "    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])\n",
    "    if(prediction[0]>0.5):\n",
    "        prediction=\"Fake note\"\n",
    "    else:\n",
    "        prediction=\"Its a Bank note\"\n",
    "    return {\n",
    "        'prediction': prediction\n",
    "    }\n",
    "\n",
    "# 5. Run the API with uvicorn\n",
    "#    Will run on http://127.0.0.1:8000\n",
    "if __name__ == '__main__':\n",
    "    uvicorn.run(app, host='127.0.0.1', port=8000)\n",
    "    \n",
    "#uvicorn app:app --reload"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
