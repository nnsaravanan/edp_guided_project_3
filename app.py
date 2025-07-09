from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)
features = ['homeworld_Alderaan', 'homeworld_Aleen Minor', 'homeworld_Bestine IV',
       'homeworld_Cerea', 'homeworld_Champala', 'homeworld_Chandrila',
       'homeworld_Concord Dawn', 'homeworld_Corellia', 'homeworld_Dagobah',
       'homeworld_Dathomir', 'homeworld_Dorin', 'homeworld_Eriadu',
       'homeworld_Glee Anselm', 'homeworld_Haruun Kal', 'homeworld_Iktotch',
       'homeworld_Iridonia', 'homeworld_Kalee', 'homeworld_Kashyyyk',
       'homeworld_Malastare', 'homeworld_Mirial', 'homeworld_Mon Cala',
       'homeworld_Muunilinst', 'homeworld_Naboo', 'homeworld_Ojom',
       'homeworld_Quermia', 'homeworld_Rodia', 'homeworld_Ryloth',
       'homeworld_Serenno', 'homeworld_Shili', 'homeworld_Skako',
       'homeworld_Socorro', 'homeworld_Stewjon', 'homeworld_Sullust',
       'homeworld_Tatooine', 'homeworld_Tholoth', 'homeworld_Toydaria',
       'homeworld_Trandosha', 'homeworld_Troiken', 'homeworld_Tund',
       'homeworld_Umbara', 'homeworld_Vulpter', 'homeworld_Zolan',
       'unit_type_at-at', 'unit_type_at-st', 'unit_type_resistance_soldier',
       'unit_type_stormtrooper', 'unit_type_tie_fighter',
       'unit_type_tie_silencer', 'unit_type_unknown', 'unit_type_x-wing']

def convert_to_dataframe(features, homeworld_input, unit_type_input):
    df = pd.DataFrame(0, index=[0], columns=features)
    print(df)

    # Update Homeworld
    homeworld_columns = [col for col in df.columns if col.startswith('homeworld_')]
    df.loc[0, homeworld_columns] = 0  
    df.loc[0, str('homeworld_') + homeworld_input] = 1    

    # Update UnitType
    unit_type_columns = [col for col in df.columns if col.startswith('unit_type')]
    df.loc[0, unit_type_columns] = 0  
    df.loc[0, str('unit_type_') + unit_type_input] = 1    
    return df

# Load the model from disk
with open('./model/decision_tree_pipeline.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/api/predict', methods=['POST'])
def predict():
    # Get the request data
    data = request.get_json(force=True)

     # Ensure the data is a list (even if it's just one dictionary)
    if isinstance(data, dict):
        data = [data]
    # print(data)
    prediction = convert_to_dataframe(features, data[0]['homeworld'], data[0]['unit_type'])
    print(type(model))
    # Make a prediction
    output = model.predict(prediction)
    return jsonify(output.tolist())

@app.route('/api/predict', methods=['GET'])
def prediction():
    dict2 = {"name": "g"}
    return jsonify(dict2)

if __name__ == '__main__':
    app.run(port=5050)