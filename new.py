from tkinter import *
from tkinter import messagebox
import numpy as np
import pandas as pd

# List of symptoms and diseases
l1 = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
      'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 
      'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 
      'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 
      'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 
      'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 
      'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 
      'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 
      'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 
      'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 
      'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 
      'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 
      'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 
      'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 
      'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 
      'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 
      'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 
      'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 
      'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 
      'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 
      'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 
      'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 
      'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 
      'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 
      'red_sore_around_nose', 'yellow_crust_ooze']

disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 
           'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension', 'Migraine', 'Cervical spondylosis', 
           'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A', 
           'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis', 
           'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins', 'Hypothyroidism', 
           'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis', 'Arthritis', '(vertigo) Paroymsal Positional Vertigo', 
           'Acne', 'Urinary tract infection', 'Psoriasis', 'Impetigo']

l2 = [0] * len(l1)

# Read and preprocess testing data
tr = pd.read_csv("Testing.csv")
tr.replace({'prognosis': dict(zip(disease, range(len(disease))))}, inplace=True)
X_test = tr[l1]
y_test = tr[["prognosis"]].values.ravel()

# Read and preprocess training data
df = pd.read_csv("Training.csv")
df.replace({'prognosis': dict(zip(disease, range(len(disease))))}, inplace=True)
df = df.infer_objects(copy=False)
X = df[l1]
y = df[["prognosis"]].values.ravel()

# Function to handle the prediction
def message():
    if all(symptom.get() == "None" for symptom in [Symptom1, Symptom2, Symptom3, Symptom4, Symptom5]):
        messagebox.showinfo("OPPS!!", "ENTER SYMPTOMS PLEASE")
    else:
        NaiveBayes()

def NaiveBayes():
    from sklearn.naive_bayes import MultinomialNB
    gnb = MultinomialNB()
    gnb.fit(X, y)
    
    from sklearn.metrics import accuracy_score
    y_pred = gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred, normalize=False))

    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    inputtest = [[1 if symptom in psymptoms else 0 for symptom in l1]]
    predicted = gnb.predict(inputtest)[0]

    t3.delete("1.0", END)
    t3.insert(END, disease[predicted])

# GUI Initialization
root = Tk()
root.title("Disease Prediction From Symptoms")

Symptom1 = StringVar()
Symptom1.set("None")
Symptom2 = StringVar()
Symptom2.set("None")
Symptom3 = StringVar()
Symptom3.set("None")
Symptom4 = StringVar()
Symptom4.set("None")
Symptom5 = StringVar()
Symptom5.set("None")

Label(root, text="Disease Prediction From Symptoms", font=("Elephant", 30)).grid(row=1, column=0, columnspan=2, padx=100)

Label(root, text="", font=("Elephant", 20)).grid(row=5, column=1, pady=10, sticky=W)

Label(root, text="Symptom 1", font=("Elephant", 15)).grid(row=7, column=1, pady=10, sticky=W)
Label(root, text="Symptom 2", font=("Elephant", 15)).grid(row=8, column=1, pady=10, sticky=W)
Label(root, text="Symptom 3", font=("Elephant", 15)).grid(row=9, column=1, pady=10, sticky=W)
Label(root, text="Symptom 4", font=("Elephant", 15)).grid(row=10, column=1, pady=10, sticky=W)
Label(root, text="Symptom 5", font=("Elephant", 15)).grid(row=11, column=1, pady=10, sticky=W)

Button(root, text="Predict", height=2, width=20, command=message, font=("Elephant", 15)).grid(row=15, column=1, pady=20)

OPTIONS = sorted(l1)
