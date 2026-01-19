import requests

# SET OUR RENDER URL
# the link below from the Render Dashboard
URL = "https://breast-cancer-mlzoomcamp.onrender.com/predict"

# CREATE A MOCK PATIENT DATA
# These are 16 features based on your 'simplified' feature list
# We've used values typical for a Malignant (M) case
patient_data = {
    "features": [
        1001.0,  # area_mean
        0.1184,  # smoothness_mean
        0.1471,  # concavity_mean
        0.0787,  # fractal_dimension_mean
        153.4,   # area_se
        0.0063,  # smoothness_se
        0.0490,  # concavity_se
        0.0030,  # fractal_dimension_se
        0.0196,  # texture_mean
        0.1622,  # smoothness_worst
        0.6656,  # concavity_worst
        0.0711,  # fractal_dimension_worst
        0.0265,  # symmetry_mean
        0.0591,  # symmetry_se
        0.1189,  # symmetry_worst
        0.4500   # (Dummy value for 16th feature if needed)
    ]
}

def test_cloud_api():
    print(f"📡 Sending request to: {URL}")
    
    try:
        # Send the POST request to Render
        response = requests.post(URL, json=patient_data)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            print("--- Full Response from Server ---")
            print(result)  # This will show us exactly what keys exist
            
            # Use the correct key based on what we see above
            # If the server sends 'diagnosis', use result['diagnosis']
            # If the server sends 'prediction', use result['prediction']
        else:
            print(f"❌ Failed! Status Code: {response.status_code}")
            print(f"Error Message: {response.text}")
            
    except Exception as e:
        print(f"🔌 Connection Error: {e}")

if __name__ == "__main__":
    test_cloud_api()
