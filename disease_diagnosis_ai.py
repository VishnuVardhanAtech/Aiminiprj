import pandas as pd
from math import prod

def create_disease_db_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    disease_db = {}
    for _, row in df.iterrows():
        disease = str(row['Name']).strip()
        symptoms = str(row['Symptoms']).split(',') if pd.notna(row['Symptoms']) else []
        symptoms = [s.strip().lower() for s in symptoms if s.strip()]
        prior = 0.01  # Assign default prior or estimate better if possible
        symptom_probs = {symptom: 0.8 for symptom in symptoms}  # heuristic likelihoods
        
        treatments = str(row['Treatments']) if pd.notna(row['Treatments']) else "Consult a healthcare provider for diagnosis and treatment."
        contagious = str(row['Contagious']) if 'Contagious' in row and pd.notna(row['Contagious']) else "N/A"
        chronic = str(row['Chronic']) if 'Chronic' in row and pd.notna(row['Chronic']) else "N/A"
        
        disease_db[disease] = {
            "symptoms": symptom_probs,
            "prior": prior,
            "treatments": treatments,
            "contagious": contagious,
            "chronic": chronic
        }
    return disease_db

def bayes_diagnose(user_positive, disease_db):
    all_symptoms = set()
    for data in disease_db.values():
        all_symptoms.update(data["symptoms"].keys())

    positive = [s.lower() for s in user_positive if s.lower() in all_symptoms]

    posterior = {}
    for disease, data in disease_db.items():
        prior = data["prior"]
        likelihoods = [data["symptoms"].get(symptom, 0.1) for symptom in positive]
        if not likelihoods:
            likelihoods.append(1)  # no symptoms input means no info
        posterior[disease] = prior * prod(likelihoods)

    total = sum(posterior.values())
    for disease in posterior:
        posterior[disease] = round(posterior[disease]/total, 4) if total else 0.0

    return sorted(posterior.items(), key=lambda x: x[1], reverse=True)[:5]

def print_results_line_by_line(top5_results, disease_db):
    for disease, prob in top5_results:
        print(f"Disease: {disease}")
        print(f"Probability: {prob*100:.2f}%")
        print(f"Treatments: {disease_db[disease]['treatments']}")
        print(f"Contagious: {disease_db[disease]['contagious']}")
        print(f"Chronic: {disease_db[disease]['chronic']}")
        print("-"*60)

def main():
    csv_path = "Diseases_Symptoms.csv"
    disease_db = create_disease_db_from_csv(csv_path)
    
    print("Enter your symptoms you HAVE (comma separated):")
    symptoms_input = input().strip()
    user_positive = [s.strip() for s in symptoms_input.split(",") if s.strip()]

    if not user_positive:
        print("No symptoms entered. Exiting.")
        return

    top5 = bayes_diagnose(user_positive, disease_db)

    print("\nTop 5 probable diseases with details:\n")
    print_results_line_by_line(top5, disease_db)

if __name__ == "__main__":
    main()
