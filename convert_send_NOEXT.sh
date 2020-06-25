jupyter nbconvert --to script "$1.ipynb"
scp "$1.py" kb-server:~/Documents/jigsaw
