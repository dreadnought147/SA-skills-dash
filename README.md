# SA-skills-dash
skills-dashboard
# Skills Analytics & Unemployment Forecasting Dashboard

An interactive dashboard for analyzing workforce skills, gaps, and unemployment trends across South African provinces.  
**Note:** The dashboard currently uses simulated data, as real market endpoints are not available./

Watch demo [here](https://drive.google.com/file/d/1umITO0KS6nOWbx4h0RxmUhbhbzcq-i-M/view?usp=sharing)

## Features

- Skills gap analysis and shortage visualization
- Critical skills distribution by province
- Machine learning predictions for entrepreneurship interest
- Neural network employment forecasting
- Unemployment rate simulation and impact analysis
- AI-powered insights and recommendations

## Technologies Used

- [Streamlit](https://streamlit.io/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [matplotlib](https://matplotlib.org/)
- [plotly](https://plotly.com/python/)
- [TensorFlow](https://www.tensorflow.org/)

## Setup Instructions

1. **Clone the repository**
   ```bash
   cd skills-analytics-dashboard
   #Create a Python virtual environment (recommended Python 3.10)
   python -m venv venv
   #Activate the virtual environment
   windows: venv\Scripts\activate
   macOS/lnux: source venv/bin/activate
   #install the requirements
   pip install streamlit pandas scikit-learn matplotlib plotly tensorflow


   #run the dash
   streamlit run dashboard.py
## final comments
consnidering that this is a dashboard that gives insight on data we need to create a platform that we can collect this data from since not many sites exist that contain this information for us .
to levergae this i made SPA that collect information on south african who are unemployed and what skills and qualification they may have check this out here [EmployementInitiative](https://github.com/dreadnought147/skills-collect)


