
# Dashboard Application

This README file provides detailed instructions on how to run the dashboard application, add new plots, and understand the structure and workflow of the application.

## Table of Contents
1. Project Structure
2. Installation and Setup
3. Running the Dashboard
4. Adding New Plots
5. Application Workflow

## Project Structure

The directory structure of the application is as follows:
```
app/
│
├── data/
│   └── curated_datasets/
│       ├── water_consumption_curated.parquet
│       └── forecasting_dataset.parquet
│
├── models/
│   ├── forecaster.pkl
│   ├── forecaster_with_weather.pkl
│   └── input_flow_forecaster.pkl
│
├── templates/
│   ├── index.html
│   ├── static_plots.html
│   ├── forecasting_plots.html
│   └── model_performance.html
│
├── plots_functions/
│   ├── static_plots.py
│   ├── forecasting_plots.py
│   └── model_performance.py
│
├── utils/
│   └── scripts/
│       └── functions/
│           └── main_functions.py
│
├── run.py
└── README.md
```

## Installation and Setup

1. **Clone the repository:**

```bash
   git clone <repository_url>
   cd <repository_name>
```

2. **Create a virtual environment and activate it:**

```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```

3. **Install the required packages:**

```bash
   pip install -r requirements.txt
```

## Running the Dashboard

1. **Navigate to the `app` directory:**

```bash
   cd app
```

2. **Run the Flask application:**

```bash
   python run.py
```

3. **Access the dashboard:**
Open your web browser and go to `http://localhost:3000`.

## Adding New Plots

### Step-by-Step Guide

1. **Create the Plot Function:**
   - Navigate to the `plots_functions` directory.
   - Identify the appropriate file for your new plot function based on the HTML file that will display the plot (e.g., `static_plots.py` for `static_plots.html`).
   - Add your plot function in the identified file. The function should accept the necessary dataframe(s) and return the plot in HTML format.

   Example:

```python
   # app/plots_functions/static_plots.py
   import plotly.graph_objects as go

   def generate_new_plot(data):
       fig = go.Figure()
       fig.add_trace(go.Scatter(x=data['x'], y=data['y'], mode='lines'))
       fig.update_layout(title='New Plot')
       return fig.to_html(full_html=False)
```

2. **Import the Plot Function:**

   - Open `run.py`.
   - Import the newly created plot function at the top of the file.

Example:

```python
   # app/run.py
   from plots_functions.static_plots import generate_new_plot
```

3. **Apply the Plot Function:**

   - Within the appropriate route function in `run.py`, apply the plot function to the dataframe and store the returned HTML.

Example:

```python
   # app/run.py
   @app.route('/static-plots')
   def static_plots():
       data = load_data_function()  # Replace with actual data loading function
       new_plot_html = generate_new_plot(data)
       return render_template('static_plots.html', plot_html=question_1_plot, plot_2_html=question_2_plot, plot_3_html=question_3_plot, plot_4_html=new_plot_html)
```

4. **Update the HTML Template:**

   - Open the corresponding HTML file in the `templates` directory (e.g., `static_plots.html`).
   - Add the new plot to the template by including the parameter passed from `run.py`.

   Example:

```html
   <!-- app/templates/static_plots.html -->
   <div class="plot-container">
       <div>
           <div class="plot-title">Column 1 Title</div>
           <div>{{ plot_html | safe }}</div>
           <div>{{ plot_2_html | safe }}</div>
       </div>
       <div>
           <div class="plot-title">Column 2 Title</div>
           <div>{{ plot_3_html | safe }}</div>
           <div>{{ plot_4_html | safe }}</div>
       </div>
       <div>
           <div class="plot-title">New Plot Title</div>
           <div>{{ new_plot_html | safe }}</div>
       </div>
   </div>
```

## Application Workflow

1. **Data Loading:**
   - Data is loaded in `run.py` from the `data/curated_datasets` directory.

2. **Plot Functions:**
   - Plot functions are defined in the `plots_functions` directory.
   - Each function accepts the necessary dataframe(s) and returns the plot in HTML format.

3. **Rendering Templates:**
   - Routes defined in `run.py` render HTML templates from the `templates` directory.
   - Plots in HTML format are passed as parameters to the `render_template` function.

4. **Adding New Plots:**
   - Create the plot function in the appropriate file in the `plots_functions` directory.
   - Import the plot function into `run.py`.
   - Apply the plot function to the dataframe in the correct route.
   - Pass the plot in HTML format as a parameter to the `render_template` function.
   - Update the HTML file to include the new plot.

By following these steps, you can efficiently add new plots and extend the functionality of your dashboard application. If you have any questions or need further assistance, please refer to the code comments and examples provided.
