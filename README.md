# Medical-chatbot

# How to run?
### STEPS:

Clone the repository

```bash
git clone https://github.com/ilyass-hm-04/Medical-chatbot.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

### STEP 03- Setup Environment Variables
Create a `.env` file in the root directory and add the following keys:

```bash
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENROUTER_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### STEP 04- Run the application
```bash
python app.py
```
