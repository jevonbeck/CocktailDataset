

source ~/Anaconda3/Scripts/activate
conda create --prefix .conda/ python=3.8 pip
conda activate .conda/
pip install setuptools wheel toml
python setup.py install


Datasets:
all_drinks.csv - https://www.kaggle.com/ai-first/cocktail-ingredients
mr-boston-flattened.csv - https://www.kaggle.com/jenlooper/mr-boston-cocktail-dataset/