```
brew install cmake boost boost-python sdl2 swig wget

git clone https://github.com/tanguy-s/aml-cw03.git
pip3 install virtualenv
virtualenv venv --system-site-package -p python3
source venv/bin/activate

pip install -r aml-cw03/requirements.txt

cd aml-cw03/code/p2

```

Start environment with rendering:

```
python main.py -e pong --test --render
python main.py -e pacman --test --render
python main.py -e boxing --test --render
```

