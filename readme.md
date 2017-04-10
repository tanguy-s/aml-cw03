Install environment:

```
brew install cmake boost boost-python sdl2 swig wget

git clone https://github.com/tanguy-s/aml-cw03.git
pip3 install virtualenv
virtualenv venv --system-site-package -p python3
source venv/bin/activate

pip install -r aml-cw03/requirements.txt

```

## Part A: Cart pole

```
cd aml-cw03/code/p1
``

Training:
```
python main.py -m A31 --training
python main.py -m A32 --training
python main.py -m A4 --training
python main.py -m A51 --training
python main.py -m A52 --training
python main.py -m A6 --training
python main.py -m A7 --training
python main.py -m A8 --training
```

Testing:
```
python main.py -m A31 --test --lr {1,2,3,4,5,6} (--render)[optional]
python main.py -m A32 --test --lr {1,2,3,4,5,6} (--render)[optional]
python main.py -m A4 --test (--render)[optional]
python main.py -m A51 --test (--render)[optional]
python main.py -m A52 --test (--render)[optional]
python main.py -m A6 --test (--render)[optional]
python main.py -m A7 --test (--render)[optional]
python main.py -m A8 --test (--render)[optional]
```


## Part B: Atari Games

```
cd aml-cw03/code/p2
``

Training:

```
python main.py -e pong --training
python main.py -e pacman --training
python main.py -e boxing --training
```

Evaluate trained agents:

```
python main.py -e pong --test (--episodes [1, 100]) (--render)[optional]
python main.py -e pacman --test (--episodes [1, 100]) (--render)[optional]
python main.py -e boxing --test (--episodes [1, 100]) (--render)[optional]
```

Evaluate random policy:

```
python main.py -e pong --random (--render)[optional]
python main.py -e pacman --random (--render)[optional]
python main.py -e boxing --random (--render)[optional]
```

Evaluate policy on initialized but untrained Q-net:

```
python main.py -e pong --notraining (--render)[optional]
python main.py -e pacman --notraining (--render)[optional]
python main.py -e boxing --notraining (--render)[optional]
```





