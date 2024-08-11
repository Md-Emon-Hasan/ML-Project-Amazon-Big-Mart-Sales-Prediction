from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and encoders
def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

model = load_pickle('models/model.pkl')
fat_content_label_encoder_1 = load_pickle('models/Fat_Content_label_encoder_1.pkl')
item_type_label_encoder_2 = load_pickle('models/Item_Type_label_encoder_2.pkl')
identifier_label_encoder_3 = load_pickle('models/Identifier_label_encoder_3.pkl')
establishment_Year_encoder_4 = load_pickle('models/Establishment_Year_encoder_4.pkl')
size_encoder_5 = load_pickle('models/Size_encoder_5.pkl')
location_Type_encoder_6 = load_pickle('models/Location_Type_encoder_6.pkl')
outlet_Type_encoder_7 = load_pickle('models/Outlet_Type_encoder_7.pkl')

# Define dropdown options
Item_Fat_Content_Var = ['Low Fat', 'Regular', 'low fat', 'LF', 'reg']
Item_Type_Var = ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads', 'Starchy Foods', 'Others', 'Seafood']
Outlet_Identifier_Var = ['OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027', 'OUT045', 'OUT017', 'OUT046', 'OUT035', 'OUT019']
Outlet_Establishment_Year_Var = [1999, 2009, 1998, 1987, 1985, 2002, 2007, 1997, 2004]
Outlet_Size_Var = ['Medium', 'High', 'Small']
Outlet_Location_Type_Var = ['Tier 1', 'Tier 3', 'Tier 2']
Outlet_Type_Var = ['Supermarket Type1', 'Supermarket Type2', 'Grocery Store', 'Supermarket Type3']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data
            Item_Fat_Get = request.form.get('Item_Fat')
            Item_Type_Get = request.form.get('Item_Type')
            Outlet_Identifier_Get = request.form.get('Outlet_Identifier')
            Outlet_Establishment_Year_Get = int(request.form.get('Outlet_Establishment_Year'))
            Outlet_Size_Get = request.form.get('Outlet_Size')
            Outlet_Location_Type_Get = request.form.get('Outlet_Location_Type')
            Outlet_Type_Get = request.form.get('Outlet_Type')

            # Debugging prints
            print("Item_Fat_Get:", Item_Fat_Get)
            print("Item_Type_Get:", Item_Type_Get)
            print("Outlet_Identifier_Get:", Outlet_Identifier_Get)
            print("Outlet_Establishment_Year_Get:", Outlet_Establishment_Year_Get)
            print("Outlet_Size_Get:", Outlet_Size_Get)
            print("Outlet_Location_Type_Get:", Outlet_Location_Type_Get)
            print("Outlet_Type_Get:", Outlet_Type_Get)

            # Encode all labels
            Item_Fat_Content_Encoded = fat_content_label_encoder_1.transform([Item_Fat_Get])[0]
            Item_Type_Encoded = item_type_label_encoder_2.transform([Item_Type_Get])[0]
            Outlet_Identifier_Encoded = identifier_label_encoder_3.transform([Outlet_Identifier_Get])[0]
            Outlet_Establishment_Year_Encoded = establishment_Year_encoder_4.transform([Outlet_Establishment_Year_Get])[0]
            Outlet_Size_Encoded = size_encoder_5.transform([Outlet_Size_Get])[0]
            Outlet_Location_Type_Encoded = location_Type_encoder_6.transform([Outlet_Location_Type_Get])[0]
            Outlet_Type_Encoded = outlet_Type_encoder_7.transform([Outlet_Type_Get])[0]

            # Predict sales
            prediction = model.predict([[Item_Fat_Content_Encoded, Item_Type_Encoded, Outlet_Identifier_Encoded, Outlet_Establishment_Year_Encoded, Outlet_Size_Encoded, Outlet_Location_Type_Encoded, Outlet_Type_Encoded]])
            return render_template('result.html', sales=prediction[0])

        except Exception as e:
            # Log the error and show a user-friendly message
            print(f"Error: {str(e)}")
            return "An error occurred while processing your request."

    # Sort dropdown values
    Item_Fat_Content_Var.sort()
    Item_Type_Var.sort()
    Outlet_Identifier_Var.sort()
    Outlet_Establishment_Year_Var.sort()
    Outlet_Size_Var.sort()
    Outlet_Location_Type_Var.sort()
    Outlet_Type_Var.sort()

    return render_template('index.html',
                           Item_Fats=Item_Fat_Content_Var,
                           Item_Types=Item_Type_Var, 
                           Outlet_Identifiers=Outlet_Identifier_Var, 
                           Outlet_Establishment_Years=Outlet_Establishment_Year_Var, 
                           Outlet_Sizes=Outlet_Size_Var, 
                           Outlet_Location_Types=Outlet_Location_Type_Var, 
                           Outlet_Types=Outlet_Type_Var)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
