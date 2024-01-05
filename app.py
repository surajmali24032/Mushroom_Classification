from flask import Flask, request, render_template
from src.pipelines.prediciton_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict_datapoints():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            
        cap_shape               =request.form.get('cap_shape'),
        cap_surface             =request.form.get('cap_surface'),
        cap_color               =request.form.get('cap_color'),
        bruises                 =request.form.get('bruises'),
        odor                    =request.form.get('odor'),
        gill_attachment         =request.form.get('gill_attachment'),
        gill_spacing            =request.form.get('gill_spacing'),
        gill_size               =request.form.get('gill_size'),
        gill_color              =request.form.get('gill_color'),
        stalk_shape             =request.form.get('stalk_shape'),
        stalk_root              =request.form.get('stalk_root'),
        stalk_surface_above_ring=request.form.get('stalk_surface_above_ring'),
        stalk_surface_below_ring=request.form.get('stalk_surface_below_ring'),
        stalk_color_above_ring  =request.form.get('stalk_color_above_ring'),
        stalk_color_below_ring  =request.form.get('stalk_color_below_ring'),
        veil_color              =request.form.get('veil_color'),
        ring_number             =request.form.get('ring_number'),
        ring_type               =request.form.get('ring_type'),
        spore_print_color       =request.form.get('spore_print_color'),
        population              =request.form.get('population'),
        habitat                 =request.form.get('habitat')

        )

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        result = pred[0]
        outpot = ""
        if result == 0:
            output = "Delicious Mushroom, You can Eat"
        else:
            output = "Don't Eat, it's Poisonous"

        return render_template('result.html', final_result = output)


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5001)
